"""Construct the built-in sensor adapters.

Each adapter module exposes ``source()``. This registry does not name them.
Detection stays inside the adapter.
"""

from __future__ import annotations

import importlib
import pkgutil

from ohho.bench.sensors.base import SensorSource


def discover_sources() -> list[SensorSource]:
    """Return one unprobed instance of each adapter that exposes ``source``."""
    import ohho.bench.sensors as sensors

    found: list[SensorSource] = []
    prefix = sensors.__name__
    for info in pkgutil.iter_modules(sensors.__path__):
        module = importlib.import_module(f"{prefix}.{info.name}")
        factory = getattr(module, "source", None)
        if not callable(factory):
            continue
        found.append(factory())
    found.sort(key=lambda item: item.source_id)
    return found
