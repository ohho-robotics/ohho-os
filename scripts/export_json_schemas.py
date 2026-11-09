#!/usr/bin/env python3
"""Export committed JSON schemas for the OhhO transport protocol (stdlib only).

Usage:
    python scripts/export_json_schemas.py          # write schemas/
    python scripts/export_json_schemas.py --check  # exit 1 if schemas/ is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = ROOT / "schemas"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _draft_meta(title: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://ohho-robotics.com/schemas/{title}.json",
        "title": title,
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    from ohho import __version__
    from ohho.protocol import PROTOCOL_VERSION

    velocity = {
        **_draft_meta("Velocity"),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "linear_x": {"type": "number", "default": 0},
            "linear_y": {"type": "number", "default": 0},
            "angular_z": {"type": "number", "default": 0},
        },
    }

    odometry = {
        **_draft_meta("Odometry"),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "number", "default": 0},
            "y": {"type": "number", "default": 0},
            "theta": {"type": "number", "default": 0},
            "vx": {"type": "number", "default": 0},
            "vy": {"type": "number", "default": 0},
            "omega": {"type": "number", "default": 0},
        },
    }

    joint_reading = {
        **_draft_meta("JointReading"),
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "position"],
        "properties": {
            "name": {"type": "string"},
            "position": {"type": "number"},
            "velocity": {"type": "number"},
        },
    }

    scan = {
        **_draft_meta("Scan"),
        "type": "object",
        "additionalProperties": False,
        "required": ["angle_min", "angle_increment"],
        "properties": {
            "angle_min": {"type": "number"},
            "angle_increment": {"type": "number"},
            "ranges": {"type": "array", "items": {"type": "number"}},
            "range_max": {"type": "number", "default": 4.0},
        },
    }

    telemetry = {
        **_draft_meta("Telemetry"),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "odom": {"$ref": "odometry.json"},
            "joints": {"type": "array", "items": {"$ref": "joint_reading.json"}},
            "battery": {"type": "number", "minimum": 0, "maximum": 1},
            "scan": {"$ref": "scan.json"},
            "custom": {"type": "object", "additionalProperties": True},
            "timestamp": {"type": "number"},
        },
    }

    connection_state = {
        **_draft_meta("ConnectionState"),
        "type": "string",
        "enum": [
            "idle",
            "connecting",
            "connected",
            "reconnecting",
            "error",
            "disconnected",
        ],
    }

    transport_status = {
        **_draft_meta("TransportStatus"),
        "type": "object",
        "additionalProperties": False,
        "required": ["protocol", "state"],
        "properties": {
            "protocol": {"type": "string"},
            "state": {"$ref": "connection_state.json"},
            "label": {"type": "string", "default": ""},
            "latency_ms": {"type": "number", "default": 0},
            "msg_rate": {"type": "number", "default": 0},
            "error": {"type": "string"},
            "connected_since": {"type": "number"},
        },
    }

    manifest = {
        "protocolVersion": PROTOCOL_VERSION,
        "ohhoOsVersion": __version__,
        "schemas": {
            "Velocity": "velocity.json",
            "Odometry": "odometry.json",
            "JointReading": "joint_reading.json",
            "Scan": "scan.json",
            "Telemetry": "telemetry.json",
            "TransportStatus": "transport_status.json",
            "ConnectionState": "connection_state.json",
        },
    }

    return {
        "manifest.json": manifest,
        "velocity.json": velocity,
        "odometry.json": odometry,
        "joint_reading.json": joint_reading,
        "scan.json": scan,
        "telemetry.json": telemetry,
        "transport_status.json": transport_status,
        "connection_state.json": connection_state,
    }


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_schemas() -> dict[str, str]:
    built = build_schemas()
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, doc in built.items():
        text = _canonical_json(doc)
        (SCHEMAS_DIR / name).write_text(text, encoding="utf-8")
        written[name] = text
    return written


def check_schemas() -> bool:
    built = build_schemas()
    ok = True
    for name, doc in built.items():
        path = SCHEMAS_DIR / name
        expected = _canonical_json(doc)
        if not path.is_file():
            print(f"missing {path.relative_to(ROOT)}", file=sys.stderr)
            ok = False
            continue
        if path.read_text(encoding="utf-8") != expected:
            print(f"stale {path.relative_to(ROOT)} (run export_json_schemas.py)", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify schemas/ matches the generator (no writes)",
    )
    args = parser.parse_args()
    if args.check:
        return 0 if check_schemas() else 1
    write_schemas()
    print(f"wrote {len(list(SCHEMAS_DIR.glob('*.json')))} files to {SCHEMAS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
