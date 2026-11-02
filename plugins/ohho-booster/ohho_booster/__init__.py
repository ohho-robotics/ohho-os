"""Booster humanoid plugin for OhhO OS.

Importing this package registers nothing by itself. The ``ohho.robots`` entry
point loads :func:`plugin`. A vendor SDK is not imported here.
"""

from __future__ import annotations

from .plugin import plugin

__all__ = ["plugin"]
