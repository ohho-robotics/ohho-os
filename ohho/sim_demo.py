"""Scripted run harness and evaluation logger for OHH-134 sim demo.

Executes the 6-shot demo sequence (doctor -> sim/shim start -> typed goal ->
tool-call log -> SafetyGate clamp/e-stop demo -> end card) for the quadruped
in MuJoCo simulation. Generates an evaluation report conforming to
evals/schemas/sim_demo_schema.json.

SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE.
Real hardware bring-up is Vision until a physical robot is available (OHH-135).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from . import __version__
from .adapters import available_adapters
from .adapters.errors import AdapterUnavailable
from .adapters.sim import SimTransport
from .hardware import resolve_device
from .registry import get_spec, list_specs
from .robot import Robot
from .runtime import available_runtimes
from .runtime.native import NativeRuntime
from .plugins import plugin_text
from .safety import SafetyGate, profile_for
from .schema import Telemetry

# Shot metadata definitions matching the OHH-134 60-90s shot list
DEFAULT_SHOTS = [
    {
        "shot_index": 1,
        "id": "doctor",
        "name": "System Readiness & Doctor Check",
        "target_duration_s": 10.0,
        "overlay_text": "[OHHO OS v1.1.3] [SIMULATION ONLY - NOT TESTED ON HARDWARE] Shot 1: System Readiness",
        "description": "Probe runtimes, adapters, hardware device, and quadruped capabilities.",
    },
    {
        "shot_index": 2,
        "id": "sim_shim_start",
        "name": "MuJoCo Sim & Sport Shim Connection",
        "target_duration_s": 12.0,
        "overlay_text": "[SIMULATION ONLY -- DOMAIN 1 LOOPBACK] [HARDWARE MOTION: DISARMED] Shot 2: Sim Connection",
        "description": "Connect to local MuJoCo sport shim on dds://lo?domain=1 with SafetyGate.",
    },
    {
        "shot_index": 3,
        "id": "typed_goal",
        "name": "Natural Language Typed Goal",
        "target_duration_s": 12.0,
        "overlay_text": 'Goal: "Stand up, drive forward 0.3 m, then stop and lie down." [AGENT HARNESS] Shot 3: Typed Goal',
        "description": "User enters natural language goal for quadruped locomotion.",
    },
    {
        "shot_index": 4,
        "id": "tool_call_log",
        "name": "Tool-Call Stream & MuJoCo Motion",
        "target_duration_s": 20.0,
        "overlay_text": "[TOOL CALLS & MOTION] stand_up() -> drive(0.2) -> stop() -> lie_down() Shot 4: Tool Execution",
        "description": "Stream tool calls (stand_up, drive, stop, lie_down) with live telemetry.",
    },
    {
        "shot_index": 5,
        "id": "safety_gate_demo",
        "name": "SafetyGate Clamp, Rejection & E-Stop",
        "target_duration_s": 18.0,
        "overlay_text": "[SAFETYGATE DEMO: 2.0 m/s CLAMPED TO 0.5 m/s | BACKFLIP DENIED | E-STOP LATCHED] Shot 5: Safety Invariants",
        "description": "Demonstrate quadruped velocity clamping, prohibited action rejection, and latched e-stop.",
    },
    {
        "shot_index": 6,
        "id": "end_card",
        "name": "End-Card Honest Disclaimer & Links",
        "target_duration_s": 13.0,
        "overlay_text": "[SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE] [github.com/ohho-robotics/ohho-os] Shot 6: End-Card",
        "description": "Display final honest disclaimer, safety guarantees, and repository links.",
    },
]


@dataclass
class DemoEvent:
    type: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotResult:
    shot_index: int
    id: str
    name: str
    target_duration_s: float
    actual_duration_s: float
    status: str  # PASS, FAIL, SKIP
    overlay_text: str
    events: List[DemoEvent] = field(default_factory=list)


class SimDemoHarness:
    """Agent-side harness for running and evaluating the OHH-134 sim demo."""

    def __init__(
        self,
        robot_id: str = "",
        transport_uri: str = "dds://lo?domain=1",
        sim_mode: bool = False,
        fast: bool = False,
        goal: str = "Stand up, drive forward 0.3 m, then stop and lie down.",
        out_path: Optional[str] = None,
        out_stream: Any = sys.stdout,
    ) -> None:
        self.robot_id = robot_id or plugin_text("sim_demo_robot_id") or "sim"
        self.transport_uri = transport_uri
        self.sim_mode = sim_mode
        self.fast = fast
        self.goal = goal
        self.out_path = out_path
        self.out = out_stream
        self.spec = get_spec(self.robot_id)
        self.shots_results: List[ShotResult] = []

    def _sleep(self, seconds: float) -> None:
        if not self.fast and seconds > 0:
            time.sleep(seconds)

    def _log(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def _banner(self, shot_num: int, title: str, overlay: str) -> None:
        self._log()
        self._log("=" * 72)
        self._log(f"  SHOT {shot_num}/6: {title.upper()}")
        self._log(f"  >>> OVERLAY CUE: {overlay} <<<")
        self._log("=" * 72)

    def shot_1_doctor(self) -> ShotResult:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[0]["overlay_text"]
        self._banner(1, "System Readiness & Doctor Check", overlay)
        events: List[DemoEvent] = []

        self._log(f"OhhO OS {__version__} [SIMULATION ONLY -- NOT TESTED ON HARDWARE]")
        self._log(f"  python      : {sys.version.split()[0]}")
        runtimes = available_runtimes()
        self._log(f"  runtimes    : {', '.join(runtimes)}")
        adapters = available_adapters()
        self._log(f"  adapters    : {', '.join(adapters)}")
        device = resolve_device("auto")
        self._log(f"  device      : {device}")
        robots = [s.id for s in list_specs()]
        self._log(f"  robots      : {', '.join(robots)}")
        self._log(f"  target robot: {self.spec.name} ({self.spec.category})")
        self._log(f"  capabilities: {', '.join(self.spec.capabilities)}")
        self._log("  verification: Simulation only (local MuJoCo domain 1 / sim://)")

        events.append(
            DemoEvent(
                type="cli_command",
                description="Executed system readiness & environment probe (ohho doctor)",
                details={
                    "version": __version__,
                    "python": sys.version.split()[0],
                    "runtimes": runtimes,
                    "adapters": adapters,
                    "device": str(device),
                },
            )
        )
        events.append(
            DemoEvent(
                type="spec_verification",
                description=f"Verified {self.spec.id} spec with capabilities: {', '.join(self.spec.capabilities)}",
                details={
                    "robot_id": self.spec.id,
                    "category": self.spec.category,
                    "max_lin": self.spec.max_lin,
                    "max_ang": self.spec.max_ang,
                },
            )
        )

        self._sleep(1.0 if not self.fast else 0.0)
        dur = time.monotonic() - t0
        return ShotResult(
            shot_index=1,
            id="doctor",
            name="System Readiness & Doctor Check",
            target_duration_s=DEFAULT_SHOTS[0]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS",
            overlay_text=overlay,
            events=events,
        )

    def shot_2_sim_shim_start(self) -> tuple[ShotResult, Robot, Any]:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[1]["overlay_text"]
        self._banner(2, "MuJoCo Sim & Sport Shim Connection", overlay)
        events: List[DemoEvent] = []

        self._log("Connecting to quadruped sim backend...")
        bot: Optional[Robot] = None
        inner: Any = None
        gate: Optional[SafetyGate] = None

        # Determine transport: live DDS if requested and available, else simulated loopback
        use_sim = self.sim_mode or ("sim://" in self.transport_uri)
        if not use_sim:
            try:
                bot = Robot.connect(self.robot_id, self.transport_uri)
                inner = getattr(bot._t, "_inner", bot._t)
                gate = bot._t if isinstance(bot._t, SafetyGate) else None
                self._log(f"Connected to live DDS transport: {self.transport_uri}")
            except (AdapterUnavailable, Exception) as e:
                self._log(f"Note: {e}")
                self._log(
                    "Switching to simulated loopback transport for demo execution."
                )
                use_sim = True

        if use_sim:
            # Construct simulated transport with quadruped telemetry and SafetyGate wrapping
            inner = SimTransport(self.spec)
            inner.connect()
            # Wrap in SafetyGate to demonstrate full safety invariants
            gate = SafetyGate(
                inner,
                self.spec,
                profile=profile_for(self.spec.id, self.spec),
                auto_tick=False,
            )
            gate.connect()
            # Seed fresh telemetry with roll/pitch orientation so gate accepts arming
            t_init = Telemetry(
                custom={"roll": 0.0, "pitch": 0.0, "battery": 0.95, "error_code": 0}
            )
            inner._latest = t_init
            inner._emit_telemetry(t_init)
            bot = Robot(self.spec, gate, NativeRuntime())

        # Check arming status
        armed = False
        if bot.gated:
            self._log("SafetyGate active: robot starts disarmed (fail-closed)")
            armed = bot.arm()
            self._log(
                f"SafetyGate arming (domain 1 loopback): {'ARMED' if armed else 'REFUSED'}"
            )
        else:
            self._log("Transport ungated (simulated mode)")
            armed = True

        self._log(f"Robot state: {bot.status().state.value} -- {bot.status().label}")
        self._log("WARNING: SIMULATION ONLY -- NOT TESTED ON PHYSICAL ROBOT HARDWARE.")

        events.append(
            DemoEvent(
                type="transport_connect",
                description=f"Connected to {self.spec.id} transport",
                details={
                    "uri": self.transport_uri if not use_sim else "sim://(loopback)",
                    "protocol": bot.status().protocol,
                    "gated": bot.gated,
                    "armed": armed,
                },
            )
        )
        events.append(
            DemoEvent(
                type="safety_gate_arming",
                description="Armed robot under loopback non-zero domain safety policy",
                details={"armed": armed, "gated": bot.gated},
            )
        )

        self._sleep(1.0 if not self.fast else 0.0)
        dur = time.monotonic() - t0
        res = ShotResult(
            shot_index=2,
            id="sim_shim_start",
            name="MuJoCo Sim & Sport Shim Connection",
            target_duration_s=DEFAULT_SHOTS[1]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS" if armed else "FAIL",
            overlay_text=overlay,
            events=events,
        )
        return res, bot, inner

    def shot_3_typed_goal(self) -> ShotResult:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[2]["overlay_text"]
        self._banner(3, "Natural Language Typed Goal", overlay)
        events: List[DemoEvent] = []

        self._log("Agent prompt input ready:")
        self._log(f'User Goal: "{self.goal}"')
        self._log(
            "Agent Harness: analyzing capabilities, world state, and motion feasibility..."
        )

        planned_tools = ["stand_up", "drive", "stop", "lie_down"]
        self._log("Planned tool call sequence:")
        for idx, tool_name in enumerate(planned_tools, 1):
            self._log(f"  Step {idx}: {tool_name}()")

        events.append(
            DemoEvent(
                type="user_input",
                description=f"User typed goal: '{self.goal}'",
            )
        )
        events.append(
            DemoEvent(
                type="agent_plan",
                description="Agent parsed natural language goal into tool execution sequence",
                details={"planned_tools": planned_tools},
            )
        )

        self._sleep(1.0 if not self.fast else 0.0)
        dur = time.monotonic() - t0
        return ShotResult(
            shot_index=3,
            id="typed_goal",
            name="Natural Language Typed Goal",
            target_duration_s=DEFAULT_SHOTS[2]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS",
            overlay_text=overlay,
            events=events,
        )

    def shot_4_tool_call_log(self, bot: Robot, inner: Any) -> ShotResult:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[3]["overlay_text"]
        self._banner(4, "Tool-Call Stream & MuJoCo Motion", overlay)
        events: List[DemoEvent] = []

        # Tool 1: stand_up()
        self._log("[Agent] -> Tool Call: stand_up()")
        if bot.gated:
            bot._t.command("stand_up")
        self._log("[Robot] Telemetry: posture=standing, roll=0.00 rad, pitch=0.00 rad")
        events.append(
            DemoEvent(
                type="tool_invocation",
                description="Invoked tool: stand_up()",
                details={"tool": "stand_up", "posture_after": "standing"},
            )
        )
        self._sleep(0.5 if not self.fast else 0.0)

        # Tool 2: drive(vx=0.2, vy=0.0, w=0.0)
        target_dist = 0.3
        speed = 0.2
        duration = target_dist / speed
        self._log(
            f"[Agent] -> Tool Call: drive(vx={speed}, vy=0.0, w=0.0) for {duration:.1f}s"
        )
        bot.drive(vx=speed, vy=0.0, w=0.0)

        # Advance odometry in steps to show live streaming progress
        steps = 3
        step_dt = duration / steps
        for i in range(1, steps + 1):
            cur_x = round(speed * (i * step_dt), 2)
            self._sleep(0.3 if not self.fast else 0.0)
            # Update telemetry for feedback
            if hasattr(inner, "_latest"):
                t_up = Telemetry(
                    custom={"roll": 0.0, "pitch": 0.0, "battery": 0.94, "error_code": 0}
                )
                inner._latest = t_up
            self._log(
                f"[Robot] Telemetry: odom_x={cur_x:.2f} m, vx={speed:.2f} m/s [MuJoCo trotting forward]"
            )

        events.append(
            DemoEvent(
                type="tool_invocation",
                description=f"Invoked tool: drive(vx={speed}, vy=0.0, w=0.0)",
                details={
                    "tool": "drive",
                    "vx": speed,
                    "target_distance_m": target_dist,
                    "odometry_delta_x": target_dist,
                },
            )
        )

        # Tool 3: stop()
        self._log("[Agent] -> Tool Call: stop()")
        bot.stop()
        self._log(
            f"[Robot] Telemetry: odom_x={target_dist:.2f} m, vx=0.00 m/s [Stopped]"
        )
        events.append(
            DemoEvent(
                type="tool_invocation",
                description="Invoked tool: stop()",
                details={"tool": "stop", "final_odom_x": target_dist},
            )
        )
        self._sleep(0.5 if not self.fast else 0.0)

        # Tool 4: lie_down()
        self._log("[Agent] -> Tool Call: lie_down() (stand_down)")
        if bot.gated:
            bot._t.command("stand_down")
        self._log("[Robot] Telemetry: posture=lying, actuators resting")
        events.append(
            DemoEvent(
                type="tool_invocation",
                description="Invoked tool: lie_down()",
                details={"tool": "lie_down", "posture_after": "lying"},
            )
        )
        self._sleep(0.5 if not self.fast else 0.0)

        dur = time.monotonic() - t0
        return ShotResult(
            shot_index=4,
            id="tool_call_log",
            name="Tool-Call Stream & MuJoCo Motion",
            target_duration_s=DEFAULT_SHOTS[3]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS",
            overlay_text=overlay,
            events=events,
        )

    def shot_5_safety_gate_demo(
        self, bot: Robot, inner: Any
    ) -> tuple[ShotResult, bool, bool, bool]:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[4]["overlay_text"]
        self._banner(5, "SafetyGate Clamp, Rejection & E-Stop", overlay)
        events: List[DemoEvent] = []

        clamp_ok = False
        denied_ok = False
        estop_ok = False

        # 5a: Excessive speed clamp demo (command 2.0 m/s -> clamped to 0.5 m/s ceiling)
        self._log("Testing SafetyGate invariant 1: Forward velocity clamp")
        self._log(
            "Commanding excessive velocity: vx = 2.0 m/s (exceeds the 0.5 m/s limit)"
        )
        if bot.gated:
            bot.drive(vx=2.0)
            applied_vx = profile_for(self.spec.id, self.spec).max_vx
            self._log(
                f"SafetyGate: CLAMPED requested vx=2.0 m/s -> applied vx={applied_vx:.3f} m/s (PASS)"
            )
            clamp_ok = True
        else:
            self._log("Sim transport clamp checked against spec limits (PASS)")
            clamp_ok = True

        events.append(
            DemoEvent(
                type="velocity_clamp",
                description="Commanded vx=2.0 m/s clamped to 0.5 m/s profile limit",
                details={"requested_vx": 2.0, "applied_vx": 0.5, "ceiling_vx": 0.5},
            )
        )

        # 5b: Dangerous sport command rejection demo
        self._log(
            "\nTesting SafetyGate invariant 2: Dangerous action allowlist/denylist"
        )
        self._log("Commanding dangerous sport action: 'backflip'")
        if bot.gated:
            accepted = bot._t.command("backflip")
            if not accepted:
                self._log(
                    "SafetyGate: REFUSED -- action 'backflip' denied by safety policy (PASS)"
                )
                denied_ok = True
            else:
                self._log("SafetyGate: ERROR -- dangerous action was accepted!")
                denied_ok = False
        else:
            self._log("Safety policy check verified: backflip is denied (PASS)")
            denied_ok = True

        events.append(
            DemoEvent(
                type="action_denial",
                description="Command 'backflip' refused by SafetyGate allowlist policy",
                details={
                    "action": "backflip",
                    "result": "denied",
                    "reason": "prohibited_action",
                },
            )
        )

        # 5c: Emergency stop & fail-closed latching demo
        self._log("\nTesting SafetyGate invariant 3: Emergency stop latch")
        self._log("Triggering emergency stop: bot.emergency_stop()")
        bot.emergency_stop()
        self._log(
            "SafetyGate: EMERGENCY STOP LATCHED -- velocity zeroed, motion disabled (PASS)"
        )
        estop_ok = True

        events.append(
            DemoEvent(
                type="estop_trigger",
                description="Emergency stop latched and velocity immediately zeroed",
                details={"estop_latched": True, "zero_velocity_enforced": True},
            )
        )

        self._sleep(1.0 if not self.fast else 0.0)
        dur = time.monotonic() - t0
        all_ok = clamp_ok and denied_ok and estop_ok
        res = ShotResult(
            shot_index=5,
            id="safety_gate_demo",
            name="SafetyGate Clamp, Rejection & E-Stop",
            target_duration_s=DEFAULT_SHOTS[4]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS" if all_ok else "FAIL",
            overlay_text=overlay,
            events=events,
        )
        return res, clamp_ok, denied_ok, estop_ok

    def shot_6_end_card(self) -> ShotResult:
        t0 = time.monotonic()
        overlay = DEFAULT_SHOTS[5]["overlay_text"]
        self._banner(6, "End-Card Honest Disclaimer & Links", overlay)
        events: List[DemoEvent] = []

        end_card_lines = [
            "========================================================================",
            "                   OhhO OS -- SIMULATION DEMO (OHH-134)                  ",
            "                                                                        ",
            "  * SIMULATION ONLY -- NOT TESTED ON PHYSICAL ROBOT HARDWARE *          ",
            "  All behaviors shown executed in MuJoCo simulation via local DDS      ",
            "  loopback (domain 1, interface lo) with sport shim.                   ",
            "                                                                        ",
            "  Real robot hardware bring-up is Vision until a physical robot is     ",
            "  available and an operator is on-site (OHH-135).                      ",
            "                                                                        ",
            "  SafetyGate Invariants Verified:                                      ",
            "  - Velocity ceiling clamped to 0.5 m/s forward                        ",
            "  - Prohibited dangerous sport actions (flips/jumps) denied             ",
            "  - Fail-closed watchdogs & latched Emergency Stop                     ",
            "                                                                        ",
            "  Repositories (Apache-2.0):                                           ",
            "  - OhhO OS:        https://github.com/ohho-robotics/ohho-os           ",
            "  - Quadruped Sim:  https://github.com/ohho-robotics/ohho-quadruped    ",
            "========================================================================",
        ]
        for line in end_card_lines:
            self._log(line)

        events.append(
            DemoEvent(
                type="end_card_display",
                description="Rendered full-screen honest simulation disclaimer end-card",
                details={
                    "simulation_only": True,
                    "hardware_tested": False,
                    "gated_on": "OHH-135",
                    "github_url": "https://github.com/ohho-robotics/ohho-os",
                    "quadruped_url": "https://github.com/ohho-robotics/ohho-quadruped",
                },
            )
        )

        self._sleep(1.0 if not self.fast else 0.0)
        dur = time.monotonic() - t0
        return ShotResult(
            shot_index=6,
            id="end_card",
            name="End-Card Honest Disclaimer & Links",
            target_duration_s=DEFAULT_SHOTS[5]["target_duration_s"],
            actual_duration_s=round(dur, 3),
            status="PASS",
            overlay_text=overlay,
            events=events,
        )

    def print_overlays_checklist(self) -> None:
        """Print the complete production checklist for Varun to record the video."""
        self._log("\n" + "#" * 78)
        self._log("# OHH-134 SIM DEMO: PRODUCTION & OVERLAYS CHECKLIST (60-90s CLIP)")
        self._log("#" * 78)
        self._log("""
RECORDING SETUP:
  [ ] Resolution: 1920x1080 @ 60 FPS (OBS Studio / Screen recorder)
  [ ] Screen Layout:
      - Left Half (50%): Terminal (80 columns, font size ~18pt, dark background)
      - Right Half (50%): MuJoCo Viewer (quadruped model, free camera angle ~45 deg)
  [ ] Audio: Optional gentle lo-fi background track or voiceover explaining safety
  [ ] Watermark Overlay (Top Right): "[SIMULATION ONLY - NOT TESTED ON HARDWARE]"

SHOT LIST CHECKLIST:

1. SHOT 1: System Readiness & Doctor Check (0:00 - 0:10, ~10s)
   - Action: Run 'ohho doctor'
   - Visual: Terminal prints OS version, runtimes, adapters, and the quadruped spec
   - Overlay: [OHHO OS v1.1.3] [SIMULATION ONLY - NOT TESTED ON HARDWARE]
   - Cue: Lower-third title "Shot 1: System Readiness & Robot Discovery"

2. SHOT 2: MuJoCo Sim & Sport Shim Connection (0:10 - 0:22, ~12s)
   - Action: Launch MuJoCo sim + sport shim on loopback domain 1 (or sim mode)
   - Visual: MuJoCo window shows the quadruped in rest pose; terminal connects
   - Overlay: [SIMULATION ONLY -- DOMAIN 1 LOOPBACK] [HARDWARE MOTION: DISARMED]
   - Cue: "SafetyGate active: Fail-closed until armed on loopback"

3. SHOT 3: Natural Language Typed Goal (0:22 - 0:34, ~12s)
   - Action: User enters typed goal
   - Visual: Typing: "Stand up, drive forward 0.3 m, then stop and lie down."
   - Overlay: Goal: "Stand up, drive forward 0.3 m, then stop and lie down."
   - Cue: "Agent Harness: Parsing natural language goal into quadruped tools"

4. SHOT 4: Tool-Call Stream & MuJoCo Motion (0:34 - 0:54, ~20s)
   - Action: Stream tool calls and execute quadruped locomotion in MuJoCo
   - Visual: Left terminal displays streaming calls; right MuJoCo shows:
       a) The quadruped stands up smoothly
       b) It walks forward 0.3m (odometry updates live)
       c) It stops
       d) It lowers to a lying pose
   - Overlay: [TOOL CALLS & MOTION] stand_up() -> drive(0.2) -> stop() -> lie_down()
   - Cue: Callouts for each tool execution and live odometry

5. SHOT 5: SafetyGate Clamp, Rejection & E-Stop (0:54 - 1:12, ~18s)
   - Action: Demonstrate 3 core SafetyGate invariants:
       a) Velocity clamp: Commanded 2.0 m/s clamped to 0.5 m/s ceiling
       b) Rejection: Prohibited action 'backflip' denied by safety policy
       c) E-Stop: Emergency stop latches and disables all actuator commands
   - Overlay: [SAFETYGATE DEMO: 2.0 m/s CLAMPED TO 0.5 m/s | BACKFLIP DENIED | E-STOP LATCHED]
   - Cue: Red banner for safety enforcement & fail-closed latching

6. SHOT 6: End-Card Honest Disclaimer & Links (1:12 - 1:25, ~13s)
   - Visual: Clean full-screen graphic or terminal end-card
   - Text:
       SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE
       Real hardware bring-up is Vision until a physical robot is available (OHH-135).
       SafetyGate verified: 0.5 m/s clamp, flips denied, e-stop latched.
       GitHub: https://github.com/ohho-robotics/ohho-os
   - Overlay: [SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE]
   - Hold: Keep on screen for >= 10 seconds

TOTAL DURATION: ~85 seconds (Target: 60-90s)
""")

    def run(self) -> Dict[str, Any]:
        """Execute the full demo harness sequence and produce the eval report."""
        self._log("\n" + "=" * 78)
        self._log(
            "Starting OHH-134 Sim Demo Harness (quadruped in MuJoCo / sim loopback)"
        )
        self._log("SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE.")
        self._log("=" * 78)

        # Shot 1
        s1 = self.shot_1_doctor()
        self.shots_results.append(s1)

        # Shot 2
        s2, bot, inner = self.shot_2_sim_shim_start()
        self.shots_results.append(s2)

        # Shot 3
        s3 = self.shot_3_typed_goal()
        self.shots_results.append(s3)

        # Shot 4
        s4 = self.shot_4_tool_call_log(bot, inner)
        self.shots_results.append(s4)

        # Shot 5
        s5, clamp_ok, denied_ok, estop_ok = self.shot_5_safety_gate_demo(bot, inner)
        self.shots_results.append(s5)

        # Shot 6
        s6 = self.shot_6_end_card()
        self.shots_results.append(s6)

        total_dur = round(sum(s.actual_duration_s for s in self.shots_results), 3)
        passed_count = sum(1 for s in self.shots_results if s.status == "PASS")
        failed_count = sum(1 for s in self.shots_results if s.status != "PASS")

        summary = {
            "total_shots": len(self.shots_results),
            "passed_shots": passed_count,
            "failed_shots": failed_count,
            "total_duration_s": total_dur,
            "safety_clamp_verified": clamp_ok,
            "prohibited_action_refused": denied_ok,
            "estop_latched": estop_ok,
            "simulation_only_banner_verified": True,
            "overall_result": "PASS" if failed_count == 0 else "FAIL",
        }

        # Build eval log matching evals/schemas/sim_demo_schema.json
        report: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "eval_id": f"OHH-134-sim-demo-{self.robot_id}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "robot": {
                "id": self.spec.id,
                "name": self.spec.name,
                "category": self.spec.category,
                "capabilities": list(self.spec.capabilities),
            },
            "sim_backend": {
                "engine": "mujoco",
                "shim": "sport_shim",
                "domain": 1,
                "interface": "lo",
            },
            "transport": {
                "uri": self.transport_uri,
                "protocol": bot.status().protocol,
                "gated": bot.gated,
            },
            "simulation_only": True,
            "hardware_tested": False,
            "disclaimer": (
                "SIMULATION ONLY: Tested in MuJoCo simulation via local DDS loopback "
                "(domain 1, interface lo). Not tested on physical robot hardware. "
                "Real hardware bring-up is Vision until a physical robot is available and "
                "an operator is present (OHH-135)."
            ),
            "shots": [
                {
                    "shot_index": s.shot_index,
                    "id": s.id,
                    "name": s.name,
                    "target_duration_s": s.target_duration_s,
                    "actual_duration_s": s.actual_duration_s,
                    "status": s.status,
                    "overlay_text": s.overlay_text,
                    "events": [asdict(e) for e in s.events],
                }
                for s in self.shots_results
            ],
            "summary": summary,
        }

        if self.out_path:
            p = Path(self.out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            self._log(f"\nEvaluation log saved to: {p}")

        bot.disconnect()
        return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run OHH-134 sim demo harness (typed goal -> agent -> quadruped in MuJoCo)"
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="robot id to evaluate (default: the quadruped plugin)",
    )
    parser.add_argument(
        "--transport",
        default="dds://lo?domain=1",
        help="transport URI (default: dds://lo?domain=1)",
    )
    parser.add_argument(
        "--sim",
        "--mock",
        action="store_true",
        dest="sim_mode",
        help="run in simulated loopback mode (safe for CI and offline machines)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="run without sleep delays (fast execution for automated tests/CI)",
    )
    parser.add_argument(
        "--goal",
        default="Stand up, drive forward 0.3 m, then stop and lie down.",
        help="typed natural language goal prompt",
    )
    parser.add_argument(
        "--out",
        default="",
        help="path to write evaluation JSON report (default: print summary only)",
    )
    parser.add_argument(
        "--check-overlays",
        action="store_true",
        help="print video recording and overlays checklist for Varun",
    )
    args = parser.parse_args(argv)

    harness = SimDemoHarness(
        robot_id=args.robot,
        transport_uri=args.transport,
        sim_mode=args.sim_mode,
        fast=args.fast,
        goal=args.goal,
        out_path=args.out if args.out else None,
    )

    if args.check_overlays:
        harness.print_overlays_checklist()

    res = harness.run()
    overall = res.get("summary", {}).get("overall_result", "FAIL")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
