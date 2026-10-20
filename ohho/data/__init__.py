"""Data collection — record episodes from a live robot to LeRobot format.

``Recorder`` wraps a :class:`~ohho.robot.Robot`, intercepts its commands to
capture actions, polls telemetry for state, and writes episodes to disk in
LeRobot v2.1 layout (Parquet + meta JSON). Humanoid state and action names
come from the spec's joint groups. The same dataset feeds :mod:`ohho.train`
and :mod:`ohho.serve`.

Heavy deps (``pyarrow`` for Parquet, ``av`` for video) are lazy: without the
``[data]`` extra the writer falls back to JSON Lines so the record→inspect loop
still works with zero external dependencies.
"""

from __future__ import annotations

from .humanoid_teleop import record_humanoid_teleop_episode
from .reader import DatasetReader
from .recorder import Episode, Recorder
from .teleop import record_go2_teleop_episodes

__all__ = [
    "Recorder",
    "Episode",
    "DatasetReader",
    "record_go2_teleop_episodes",
    "record_humanoid_teleop_episode",
]
