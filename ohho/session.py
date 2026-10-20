"""MCP robot session bound to one resolved robot id.

Console and VR pass only the robot id. When ``robot_id`` is omitted the session
uses :func:`ohho.config.resolve_robot_id` (flag, then ``OHHO_ROBOT``, then
``~/.ohho/config.toml``, then ``sim``) and that robot's :class:`RobotProfile`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import RobotProfile, load_robot_profile, resolve_robot_id
from .robot import Robot


class RobotSession:
    """One MCP session for one robot.

    An explicit ``robot_id`` is the flag and wins over the environment and
    config file. Omit it to use the shared switch point. ``home``, ``cwd`` and
    ``config_path`` are injectable so tests do not read the real ``~/.ohho``.
    """

    def __init__(
        self,
        robot_id: str | None = None,
        *,
        transport: str | None = None,
        runtime: str = "auto",
        home: Path | str | None = None,
        cwd: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        config_path: Path | str | None = None,
        safety_config: str | None = None,
        estop_damp: bool = False,
    ) -> None:
        explicit = isinstance(robot_id, str) and bool(robot_id.strip())
        self.resolved = resolve_robot_id(
            robot_id if explicit else None,
            home=home,
            environ=environ,
            config_path=config_path,
        )
        if explicit and home is None and cwd is None and config_path is None:
            self.profile = RobotProfile(self.resolved.robot_id)
        else:
            self.profile = load_robot_profile(
                self.resolved.robot_id, home=home, cwd=cwd
            )
        self.transport = transport if transport is not None else self.profile.transport
        self.runtime = runtime
        self.safety_config = safety_config
        self.estop_damp = estop_damp

    @property
    def robot_id(self) -> str:
        return self.resolved.robot_id

    def connect(self) -> Robot:
        """Connect with the profile's transport, sim backend and tighter caps.

        Does not arm the safety gate. A simulated transport is used when the
        profile does not name one and the caller did not pass ``transport``.
        """
        extra: dict[str, Any] = {}
        if self.profile.sim_backend:
            extra["sim_backend"] = self.profile.sim_backend
        if self.profile.asset_path:
            extra["asset_path"] = self.profile.asset_path
        if self.profile.safety_overrides:
            extra["safety_overrides"] = dict(self.profile.safety_overrides)
        return Robot.connect(
            self.robot_id,
            transport=self.transport,
            runtime=self.runtime,
            safety_config=self.safety_config,
            estop_damp=self.estop_damp,
            **extra,
        )


__all__ = ["RobotSession"]
