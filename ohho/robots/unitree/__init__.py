"""Built-in quadruped plugin.

The ``ohho.robots`` entry point loads :func:`plugin`. Importing this package
does not import a vendor SDK.
"""

from __future__ import annotations

from .plugin import plugin

__all__ = ["plugin"]
