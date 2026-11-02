"""Conformance kit: per-class sim scenarios and the vendor-name grep (OHH-315)."""

from __future__ import annotations

from .runner import (
    CLASS_SCENARIOS,
    Check,
    ClassScenario,
    RobotResult,
    evaluate_mcp,
    format_result,
    main,
    open_sim_backend,
    run_all,
    run_robot,
    run_spec,
)

__all__ = [
    "CLASS_SCENARIOS",
    "Check",
    "ClassScenario",
    "RobotResult",
    "evaluate_mcp",
    "format_result",
    "main",
    "open_sim_backend",
    "run_all",
    "run_robot",
    "run_spec",
]
