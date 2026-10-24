"""Generic humanoid simulator: any MJCF, a plugin policy, then PD.

``Robot.drive`` stores a body velocity. On a link whose protocol is not
``simulated``, :class:`~ohho.safety.SafetyGate` clamps that command before it
reaches the transport. This backend's protocol is ``simulated``, so the gate's
existing rule leaves it ungated. The transport then does the rest:

* a locomotion policy, ONNX or TorchScript, at a plugin rate (default 50 Hz)
* a PD torque on every model step, ``tau = kp (q_target - q) - kd dq``

``prepare`` holds ``default_q``. ``damp`` sets ``kp`` to 0. ``custom`` copies
joint targets straight into the PD setpoint. None of those three load a policy.
The MJCF, joint order, gains, observation layout and checkpoint come from
:class:`~ohho.humanoid_sim.HumanoidSimConfig`.

MuJoCo is imported on ``connect``, not when this module loads.

A passive viewer is optional. ``attach_viewer`` stores a
``mujoco.viewer.launch_passive`` handle, and each :meth:`step` syncs it.
The live ``start_sim`` loop calls :meth:`step`, so the window follows that
loop. Headless operation never opens one.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..adapters.errors import AdapterUnavailable
from ..humanoid import HumanoidMode
from ..humanoid_sim import (
    DEFAULT_POLICY_RATE_HZ,
    HumanoidSimConfig,
    ObservationExtra,
    ObservationTerm,
    PolicyLayout,
    append_observation_frame,
    build_observation,
    clip_vector,
    default_observation_terms,
    fall_state_from_tilt,
    flatten_history,
    policy_stride,
    projected_gravity,
    quat_wxyz_to_rpy,
    resolve_sim_config,
    rpy_to_quat_wxyz,
)
from ..humanoid_telemetry import apply_humanoid_telemetry, build_humanoid_telemetry
from ..policy_runner import LocomotionPolicy
from ..registry import RobotSpec
from ..schema import (
    ConnectionState,
    JointReading,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ..transport import BaseTransport


def _policy_scales(value: object, n: int) -> tuple[float, ...]:
    if isinstance(value, tuple) and len(value) == n:
        return tuple(float(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return tuple(float(value) for _ in range(n))
    raise ValueError("action_scale does not match the policy joints")


def _as_gain_tuple(value: object, n: int, label: str) -> tuple[float, ...]:
    """PD gains after config validation: one finite float per joint."""
    if isinstance(value, tuple) and len(value) == n:
        return tuple(float(item) for item in value)
    raise ValueError(f"{label} length does not match joint_names")


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise AdapterUnavailable(
            "mujoco is not installed; pip install 'ohho-os[humanoid-sim]'"
        ) from exc
    return mujoco


class HumanoidMujocoTransport(BaseTransport):
    """MuJoCo humanoid link. ``step`` is deterministic; ``start_sim`` is live."""

    protocol = "simulated"

    def __init__(
        self,
        spec: RobotSpec,
        address: str = "",
        runtime: object = None,
        config: HumanoidSimConfig | None = None,
    ) -> None:
        super().__init__()
        del runtime
        self.spec = spec
        self.address = address
        self._config = config
        self._mj: Any = None
        self._model: Any = None
        self._data: Any = None
        self._mode = HumanoidMode.OFF
        self._cmd = Velocity()
        self._estopped = False
        self._state = ConnectionState.IDLE
        self._connected_since: float | None = None
        self._joint_names: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self._qadr: tuple[int, ...] = ()
        self._dadr: tuple[int, ...] = ()
        self._actuators: tuple[int | None, ...] = ()
        self._kp: tuple[float, ...] = ()
        self._kd: tuple[float, ...] = ()
        self._default_q: tuple[float, ...] = ()
        self._q_target: list[float] = []
        self._kp_enabled = False
        self._policy: LocomotionPolicy | None = None
        self._policy_joints: tuple[str, ...] = ()
        self._terms: tuple[ObservationTerm, ...] = ()
        self._action_scale: tuple[float, ...] = ()
        self._q_relative = False
        self._rate_hz = DEFAULT_POLICY_RATE_HZ
        self._history_length = 1
        self._history_init = "zeros"
        self._clip_observation: float | None = None
        self._clip_action: float | None = None
        self._extra: ObservationExtra | None = None
        self._history: tuple[tuple[float, ...], ...] | None = None
        self._walk_steps = 0
        self._last_action: tuple[float, ...] = ()
        self._base_body = 0
        self._free_qadr: int | None = None
        self._push_force = (0.0, 0.0, 0.0)
        self._push_left = 0.0
        self.pd_steps = 0
        self.last_tau: tuple[float, ...] = ()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._viewer: object | None = None
        self._viewer_lock = threading.Lock()

    @property
    def mode(self) -> HumanoidMode:
        """Generic posture the backend is in."""
        return self._mode

    @property
    def command(self) -> Velocity:
        """Last body-velocity command. Zero after an e-stop."""
        return self._cmd

    @property
    def q_target(self) -> tuple[float, ...]:
        """PD position setpoints, in the plugin's joint order."""
        return tuple(self._q_target)

    @property
    def last_action(self) -> tuple[float, ...]:
        """Previous policy output. Zeros until the policy has run."""
        return self._last_action

    @property
    def timestep(self) -> float:
        """MuJoCo ``opt.timestep``. PD runs once per this interval."""
        self._require_model()
        return float(self._model.opt.timestep)

    @property
    def policy_calls(self) -> int:
        """How many times the loaded policy has been evaluated."""
        if self._policy is None:
            return 0
        return self._policy.calls

    @property
    def policy_rate_hz(self) -> float:
        """Configured policy rate. 50 Hz when no policy is loaded."""
        return self._rate_hz

    @property
    def base_position(self) -> tuple[float, float, float]:
        """World position of the floating base, ``(x, y, z)`` in metres."""
        self._require_model()
        pos = self._data.xpos[self._base_body]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    @property
    def mujoco_model(self) -> Any:
        """Live MuJoCo model, or ``None`` before :meth:`connect`."""
        return self._model

    @property
    def mujoco_data(self) -> Any:
        """Live MuJoCo data. The sim loop is the only writer."""
        return self._data

    def attach_viewer(self, handle: object) -> None:
        """Show ``handle`` and sync it from :meth:`step`.

        ``handle`` is the object returned by ``mujoco.viewer.launch_passive``.
        A previous handle is closed.
        """
        with self._viewer_lock:
            previous = self._viewer
            self._viewer = handle
        if previous is not None and previous is not handle:
            from ..mujoco_viewer import close_viewer

            close_viewer(previous)

    def viewer_is_running(self) -> bool:
        """True while an attached passive viewer still has its window."""
        with self._viewer_lock:
            handle = self._viewer
        if handle is None:
            return False
        running = getattr(handle, "is_running", None)
        if not callable(running):
            return True
        try:
            return bool(running())
        except Exception:
            return False

    def policy_stride_steps(self) -> int:
        """Model steps between policy evaluations at the current rate."""
        return policy_stride(self.timestep, self._rate_hz)

    def effective_kp(self) -> tuple[float, ...]:
        """PD stiffness in use. All zeros in ``damp`` and before ``prepare``."""
        if not self._kp_enabled:
            return tuple(0.0 for _ in self._kp)
        return self._kp

    def actuator_index(self, joint_name: str) -> int | None:
        """MuJoCo actuator slot for ``joint_name``, or ``None`` when torque is applied in joint space."""
        self._require_model()
        slot = self._index.get(joint_name)
        if slot is None:
            raise KeyError(joint_name)
        return self._actuators[slot]

    def connect(self) -> TransportStatus:
        mj = _import_mujoco()
        config = (
            self._config
            if self._config is not None
            else resolve_sim_config(self.spec, self.address)
        )
        if self.address.strip().lower().endswith(".xml"):
            config = replace(config, mjcf=self.address.strip())
        self._config = config
        self._mj = mj
        try:
            self._model = mj.MjModel.from_xml_path(config.mjcf)
        except Exception as exc:
            raise AdapterUnavailable(
                f"could not load MJCF {config.mjcf!r}: {exc}"
            ) from exc
        self._data = mj.MjData(self._model)
        if config.timestep is not None:
            self._model.opt.timestep = config.timestep
        requested = self._mode
        self._bind(config)
        self._spawn(config)
        self._restore_mode(requested)
        mj.mj_forward(self._model, self._data)
        self._state = ConnectionState.CONNECTED
        self._connected_since = time.time()
        status = self.status()
        self._emit_status(status)
        return status

    def disconnect(self) -> None:
        self.stop_sim()
        self._close_viewer()
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"humanoid-mujoco · {self.spec.id}",
            connected_since=self._connected_since,
        )

    def send_velocity(self, vel: Velocity) -> None:
        if self._estopped:
            self._cmd = Velocity()
            return
        self._cmd = vel

    def send_joint_command(self, name: str, position: float) -> None:
        """Pass one joint target through in ``custom``. Other modes ignore it."""
        if self._mode is not HumanoidMode.CUSTOM:
            return
        slot = self._index.get(name)
        if slot is None:
            return
        self._q_target[slot] = float(position)

    def send_humanoid(self, action: str, *args: object, **fields: object) -> None:
        """Posture and custom joint targets. No sport-mode call."""
        name = action.strip().lower()
        if name == "prepare":
            self._enter_prepare()
        elif name == "walk_mode":
            self._enter_walk()
        elif name == "damp":
            self._mode = HumanoidMode.DAMPING
            self._kp_enabled = False
        elif name == "custom":
            self._mode = HumanoidMode.CUSTOM
            self._kp_enabled = True
        elif name == "move_joints":
            self._move_joints(args, fields)
        elif name == "get_up":
            self._enter_prepare()
        elif name == "lie_down":
            self._mode = HumanoidMode.DAMPING
            self._kp_enabled = False
        elif name in {"look", "move_ee", "gripper", "hand"}:
            return
        else:
            raise ValueError(f"unknown humanoid action {action!r}")

    def emergency_stop(self) -> None:
        """Zero the velocity command and hold the current joint targets.

        The policy stops updating. The mode is left as it is. Damping is a
        separate ``damp`` call.
        """
        self._estopped = True
        self._cmd = Velocity()
        if self._model is not None and self._q_target:
            q, _dq = self._joint_state()
            self._q_target = list(q)

    def release_stop(self) -> None:
        self._estopped = False

    def inject_push(self, force: Sequence[float], duration_s: float = 0.0) -> None:
        """Apply a world-frame force on the floating base for conformance tests.

        ``force`` is ``(fx, fy, fz)`` in newtons. ``duration_s`` is sim time.
        Zero holds the force for one model step.
        """
        self._require_model()
        if isinstance(force, (str, bytes)) or len(force) != 3:
            raise ValueError("push force must be fx, fy, fz")
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        self._push_force = (float(force[0]), float(force[1]), float(force[2]))
        self._push_left = self.timestep if duration_s == 0 else float(duration_s)

    def set_base_pose(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 1.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        """Place the floating base. Conformance tests use this for tilt and odom."""
        self._require_model()
        if self._free_qadr is None:
            raise ValueError("this MJCF has no free joint to pose")
        adr = self._free_qadr
        quat = rpy_to_quat_wxyz(roll, pitch, yaw)
        self._data.qpos[adr : adr + 7] = (x, y, z, quat[0], quat[1], quat[2], quat[3])
        self._mj.mj_forward(self._model, self._data)

    def read(self) -> Telemetry:
        self._require_model()
        self._mj.mj_forward(self._model, self._data)
        return self._snapshot()

    def step(self, dt: float | None = None) -> Telemetry:
        """Advance one model step, or ``round(dt / timestep)`` model steps.

        Each model step updates PD, then calls ``mj_step``. The policy runs
        only in ``walk`` mode, once per :meth:`policy_stride_steps`.
        """
        self._require_model()
        timestep = self.timestep
        if dt is None:
            count = 1
        else:
            if dt <= 0:
                raise ValueError("dt must be > 0")
            count = max(1, int(round(dt / timestep)))
        with self._viewer_physics_guard():
            for _ in range(count):
                self._substep()
            snap = self.read()
        self._emit_telemetry(snap)
        self._sync_viewer()
        return snap

    def start_sim(self, hz: float | None = None) -> None:
        """Background loop at the model rate. Tests call :meth:`step` instead.

        Each iteration calls :meth:`step`, which syncs an attached passive
        viewer. The window is not opened here.
        """
        if self._thread is not None:
            return
        self._require_model()
        self._stop.clear()
        period = self.timestep if hz is None else 1.0 / hz

        def loop() -> None:
            while not self._stop.is_set():
                self.step(period)
                time.sleep(period)

        self._thread = threading.Thread(
            target=loop, name="ohho-humanoid-mujoco", daemon=True
        )
        self._thread.start()

    def stop_sim(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None

    def _viewer_physics_guard(self) -> contextlib.AbstractContextManager[object]:
        """Lock the passive viewer around ``mj_step``.

        The live loop steps on a background thread while the viewer renders
        on its own thread. MuJoCo's handle lock keeps those two off the same
        ``mjData``. ``sync`` takes the lock itself, so it stays outside this
        guard. No viewer means the step runs as it always has.
        """
        with self._viewer_lock:
            handle = self._viewer
        if handle is None:
            return contextlib.nullcontext()
        lock = getattr(handle, "lock", None)
        if not callable(lock):
            return contextlib.nullcontext()
        try:
            guard = lock()
        except Exception:
            return contextlib.nullcontext()
        if not hasattr(guard, "__enter__"):
            return contextlib.nullcontext()
        return guard

    def _sync_viewer(self) -> None:
        with self._viewer_lock:
            handle = self._viewer
        if handle is None:
            return
        from ..mujoco_viewer import sync_viewer

        sync_viewer(handle)

    def _close_viewer(self) -> None:
        with self._viewer_lock:
            handle = self._viewer
            self._viewer = None
        if handle is None:
            return
        from ..mujoco_viewer import close_viewer

        close_viewer(handle)

    def _restore_mode(self, requested: HumanoidMode) -> None:
        if requested is HumanoidMode.PREPARE:
            self._enter_prepare()
        elif requested is HumanoidMode.WALK:
            self._enter_walk()
        elif requested is HumanoidMode.DAMPING:
            self._mode = HumanoidMode.DAMPING
            self._kp_enabled = False
        elif requested is HumanoidMode.CUSTOM:
            self._mode = HumanoidMode.CUSTOM
            self._kp_enabled = True

    def _enter_prepare(self) -> None:
        self._mode = HumanoidMode.PREPARE
        self._kp_enabled = True
        if self._default_q:
            self._q_target = list(self._default_q)

    def _enter_walk(self) -> None:
        self._mode = HumanoidMode.WALK
        self._kp_enabled = True
        self._reset_policy_memory()

    def _reset_policy_memory(self) -> None:
        self._walk_steps = 0
        self._history = None
        if self._extra is not None:
            self._extra.reset()
        self._seed_last_action()

    def _seed_last_action(self) -> None:
        count = len(self._policy_joints)
        if count == 0 or not self._q_relative or self._model is None:
            self._last_action = tuple(0.0 for _ in range(count))
            return
        positions, _velocities = self._joint_state()
        by_name = dict(zip(self._joint_names, positions))
        default_by_name = dict(zip(self._joint_names, self._default_q))
        seeded: list[float] = []
        for index, name in enumerate(self._policy_joints):
            scale = self._action_scale[index]
            if scale == 0.0:
                seeded.append(0.0)
                continue
            seeded.append((by_name[name] - default_by_name[name]) / scale)
        self._last_action = tuple(seeded)

    def _spawn(self, config: HumanoidSimConfig) -> None:
        if config.base_pos is None:
            return
        if self._free_qadr is None:
            raise ValueError("base_pos needs a free joint")
        x, y, z = config.base_pos
        adr = self._free_qadr
        self._data.qpos[adr] = x
        self._data.qpos[adr + 1] = y
        self._data.qpos[adr + 2] = z
        for index, qadr in enumerate(self._qadr):
            self._data.qpos[qadr] = self._default_q[index]
        self._data.qvel[:] = 0.0
        self._q_target = list(self._default_q)

    def _move_joints(
        self, args: tuple[object, ...], fields: Mapping[str, object]
    ) -> None:
        if self._mode is not HumanoidMode.CUSTOM:
            return
        if args:
            group = args[0]
            targets = args[1] if len(args) > 1 else None
        else:
            group = fields.get("group")
            targets = fields.get("q")
        if not isinstance(group, str) or group.strip() == "":
            raise ValueError("move_joints needs a group name")
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise TypeError("move_joints q must be a sequence of positions")
        names = self._group_names(group.strip())
        if len(targets) != len(names):
            raise ValueError(
                f"group {group!r} has {len(names)} joints, got {len(targets)} targets"
            )
        for name, value in zip(names, targets):
            slot = self._index.get(name)
            if slot is None:
                raise KeyError(name)
            self._q_target[slot] = float(value)  # type: ignore[arg-type]

    def _group_names(self, group: str) -> tuple[str, ...]:
        groups = self.spec.joint_groups
        if isinstance(groups, Mapping) and group in groups:
            return tuple(str(name) for name in groups[group])
        if group in self._index:
            return (group,)
        raise ValueError(f"unknown joint group {group!r}")

    def _bind(self, config: HumanoidSimConfig) -> None:
        names = config.joint_names or self._discover_joints()
        if not names:
            raise AdapterUnavailable("MJCF has no joints for the PD loop")
        n = len(names)
        if config.joint_names:
            kp = _as_gain_tuple(config.kp, n, "kp")
            kd = _as_gain_tuple(config.kd, n, "kd")
            default_q = _as_gain_tuple(config.default_q, n, "default_q")
        else:
            kp = tuple(20.0 for _ in range(n))
            kd = tuple(1.0 for _ in range(n))
            default_q = tuple(0.0 for _ in range(n))
        self._joint_names = names
        self._index = {name: i for i, name in enumerate(names)}
        self._kp = kp
        self._kd = kd
        self._default_q = default_q
        self._q_target = list(default_q)
        self._qadr, self._dadr, self._actuators = self._map_joints(names)
        self._base_body, self._free_qadr = self._find_base(config.base_body)
        self._install_policy(config.policy, names)
        self._kp_enabled = False
        self._mode = HumanoidMode.OFF

    def _install_policy(
        self, policy: PolicyLayout | None, joints: tuple[str, ...]
    ) -> None:
        self._policy = None
        self._policy_joints = joints
        self._terms = default_observation_terms()
        self._action_scale = tuple(1.0 for _ in joints)
        self._q_relative = False
        self._rate_hz = DEFAULT_POLICY_RATE_HZ
        self._history_length = 1
        self._history_init = "zeros"
        self._clip_observation = None
        self._clip_action = None
        self._extra = None
        self._history = None
        self._last_action = tuple(0.0 for _ in joints)
        if policy is None or not policy.checkpoint:
            return
        order = policy.joint_names or joints
        missing = [name for name in order if name not in self._index]
        if missing:
            raise ValueError(f"policy joints are not in the model: {missing}")
        self._policy_joints = order
        self._terms = policy.terms
        self._action_scale = _policy_scales(policy.action_scale, len(order))
        self._q_relative = policy.q_relative
        self._rate_hz = policy.rate_hz
        self._history_length = policy.history_length
        self._history_init = policy.history_init
        self._clip_observation = policy.clip_observation
        self._clip_action = policy.clip_action
        self._extra = None if policy.extra_factory is None else policy.extra_factory()
        self._last_action = tuple(0.0 for _ in order)
        self._policy = LocomotionPolicy(policy.checkpoint, rate_hz=policy.rate_hz)

    def _discover_joints(self) -> tuple[str, ...]:
        mj = self._mj
        model = self._model
        names: list[str] = []
        for i in range(int(model.nu)):
            if int(model.actuator_trntype[i]) != int(mj.mjtTrn.mjTRN_JOINT):
                continue
            joint_id = int(model.actuator_trnid[i, 0])
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if name and name not in names:
                names.append(name)
        return tuple(names)

    def _map_joints(
        self, names: tuple[str, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int | None, ...]]:
        mj = self._mj
        model = self._model
        qadr: list[int] = []
        dadr: list[int] = []
        actuators: list[int | None] = []
        hinge = int(mj.mjtJoint.mjJNT_HINGE)
        slide = int(mj.mjtJoint.mjJNT_SLIDE)
        for name in names:
            joint_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ValueError(f"MJCF has no joint {name!r}")
            kind = int(model.jnt_type[joint_id])
            if kind not in (hinge, slide):
                raise ValueError(f"joint {name!r} must be a hinge or a slide")
            qadr.append(int(model.jnt_qposadr[joint_id]))
            dadr.append(int(model.jnt_dofadr[joint_id]))
            actuators.append(self._actuator_for(joint_id))
        return tuple(qadr), tuple(dadr), tuple(actuators)

    def _actuator_for(self, joint_id: int) -> int | None:
        mj = self._mj
        model = self._model
        joint_trn = int(mj.mjtTrn.mjTRN_JOINT)
        for i in range(int(model.nu)):
            if int(model.actuator_trntype[i]) != joint_trn:
                continue
            if int(model.actuator_trnid[i, 0]) == joint_id:
                return i
        return None

    def _find_base(self, body_name: str) -> tuple[int, int | None]:
        mj = self._mj
        model = self._model
        free_qadr: int | None = None
        free_body = 0
        free = int(mj.mjtJoint.mjJNT_FREE)
        for i in range(int(model.njnt)):
            if int(model.jnt_type[i]) == free:
                free_body = int(model.jnt_bodyid[i])
                free_qadr = int(model.jnt_qposadr[i])
                break
        if body_name:
            named = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
            if named < 0:
                raise ValueError(f"MJCF has no body {body_name!r}")
            return named, free_qadr
        if free_body:
            return free_body, free_qadr
        return (1 if int(model.nbody) > 1 else 0), None

    def _substep(self) -> None:
        self._apply_push()
        self._maybe_policy()
        q, dq = self._joint_state()
        tau = self._pd(q, dq)
        self.last_tau = tuple(tau)
        self._write_tau(tau)
        self.pd_steps += 1
        self._mj.mj_step(self._model, self._data)

    def _apply_push(self) -> None:
        body = self._base_body
        self._data.xfrc_applied[body, :] = 0
        if self._push_left <= 0:
            return
        fx, fy, fz = self._push_force
        self._data.xfrc_applied[body, 0] = fx
        self._data.xfrc_applied[body, 1] = fy
        self._data.xfrc_applied[body, 2] = fz
        self._push_left -= self.timestep
        if self._push_left < 1e-12:
            self._push_left = 0.0

    def _maybe_policy(self) -> None:
        if (
            self._policy is None
            or self._mode is not HumanoidMode.WALK
            or self._estopped
        ):
            return
        self._walk_steps += 1
        if self._walk_steps % self.policy_stride_steps() != 0:
            return
        self._apply_policy()

    def _apply_policy(self) -> None:
        assert self._policy is not None
        q, dq = self._joint_state()
        by_name = dict(zip(self._joint_names, q))
        dq_by_name = dict(zip(self._joint_names, dq))
        default_by_name = dict(zip(self._joint_names, self._default_q))
        ordered_q = []
        ordered_dq = []
        for name in self._policy_joints:
            position = by_name[name]
            if self._q_relative:
                position -= default_by_name[name]
            ordered_q.append(position)
            ordered_dq.append(dq_by_name[name])
        _roll, _pitch, yaw, ang, grav = self._base_signals()
        command = (self._cmd.linear_x, self._cmd.linear_y, self._cmd.angular_z)
        frame = build_observation(
            base_ang_vel=ang,
            projected_gravity=grav,
            command=command,
            q=ordered_q,
            dq=ordered_dq,
            last_action=self._last_action,
            terms=self._terms,
        )
        if self._extra is not None:
            frame = frame + tuple(
                self._extra.extend(yaw=yaw, command=command, dt=1.0 / self._rate_hz)
            )
        frame = clip_vector(frame, self._clip_observation)
        self._history = append_observation_frame(
            self._history,
            frame,
            length=self._history_length,
            init=self._history_init,
        )
        action = clip_vector(
            self._policy.act(flatten_history(self._history)), self._clip_action
        )
        if len(action) != len(self._policy_joints):
            raise ValueError(
                f"policy returned {len(action)} actions for {len(self._policy_joints)} joints"
            )
        self._last_action = action
        for index, (name, value) in enumerate(zip(self._policy_joints, action)):
            slot = self._index[name]
            self._q_target[slot] = (
                self._default_q[slot] + value * self._action_scale[index]
            )

    def _pd(self, q: Sequence[float], dq: Sequence[float]) -> list[float]:
        if self._mode is HumanoidMode.OFF:
            return [0.0 for _ in q]
        tau: list[float] = []
        for i, (position, velocity) in enumerate(zip(q, dq)):
            kp = self._kp[i] if self._kp_enabled else 0.0
            tau.append(kp * (self._q_target[i] - position) - self._kd[i] * velocity)
        return tau

    def _write_tau(self, tau: Sequence[float]) -> None:
        self._data.qfrc_applied[:] = 0
        for i, value in enumerate(tau):
            actuator = self._actuators[i]
            if actuator is None:
                self._data.qfrc_applied[self._dadr[i]] = value
            else:
                self._data.ctrl[actuator] = value

    def _joint_state(self) -> tuple[list[float], list[float]]:
        q = [float(self._data.qpos[adr]) for adr in self._qadr]
        dq = [float(self._data.qvel[adr]) for adr in self._dadr]
        return q, dq

    def set_joint_velocity(self, name: str, velocity: float) -> None:
        """Set one joint's velocity. Conformance tests use this for the PD law."""
        self._require_model()
        slot = self._index.get(name)
        if slot is None:
            raise KeyError(name)
        self._data.qvel[self._dadr[slot]] = float(velocity)

    def _base_signals(
        self,
    ) -> tuple[
        float, float, float, tuple[float, float, float], tuple[float, float, float]
    ]:
        import numpy as np

        mj = self._mj
        body = self._base_body
        quat = self._data.xquat[body]
        roll, pitch, yaw = quat_wxyz_to_rpy(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        )
        xmat = [float(item) for item in self._data.xmat[body]]
        grav = projected_gravity(xmat)
        vel = np.zeros(6)
        mj.mj_objectVelocity(
            self._model, self._data, int(mj.mjtObj.mjOBJ_BODY), body, vel, 1
        )
        ang = (float(vel[0]), float(vel[1]), float(vel[2]))
        return roll, pitch, yaw, ang, grav

    def _snapshot(self) -> Telemetry:
        q, dq = self._joint_state()
        roll, pitch, yaw, ang, _grav = self._base_signals()
        pos = self._data.xpos[self._base_body]
        joints = {
            name: JointReading(name=name, position=q[i], velocity=dq[i])
            for i, name in enumerate(self._joint_names)
        }
        block = build_humanoid_telemetry(
            mode=self._mode,
            vendor_mode="",
            fall_state=fall_state_from_tilt(roll, pitch),
            base_rpy=(roll, pitch, yaw),
            base_ang_vel=ang,
            joints=self._group_samples(q, dq),
            odom={"x": float(pos[0]), "y": float(pos[1]), "yaw": yaw},
            battery={"fraction": 1.0, "voltage": 0.0},
            experimental=True,
            spec=self.spec if self._spec_groups() else None,
        )
        telemetry = Telemetry(
            joints=[joints[name] for name in self._joint_names if name in joints],
            battery=1.0,
            custom={},
        )
        return apply_humanoid_telemetry(telemetry, block)

    def _spec_groups(self) -> bool:
        groups = self.spec.joint_groups
        return isinstance(groups, Mapping) and len(groups) > 0

    def _group_samples(
        self, q: Sequence[float], dq: Sequence[float]
    ) -> dict[str, dict[str, list[float]]]:
        by_q = dict(zip(self._joint_names, q))
        by_dq = dict(zip(self._joint_names, dq))
        by_tau = dict(zip(self._joint_names, self.last_tau or (0.0 for _ in q)))
        groups: Mapping[str, Sequence[str]]
        if self._spec_groups():
            groups = self.spec.joint_groups  # type: ignore[assignment]
        else:
            groups = {"joints": self._joint_names}
        samples: dict[str, dict[str, list[float]]] = {}
        for group, names in groups.items():
            samples[str(group)] = {
                "q": [float(by_q[str(name)]) for name in names],
                "dq": [float(by_dq[str(name)]) for name in names],
                "tau": [float(by_tau.get(str(name), 0.0)) for name in names],
            }
        return samples

    def _require_model(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                "connect() the humanoid-mujoco transport before stepping"
            )


__all__ = ["HumanoidMujocoTransport"]
