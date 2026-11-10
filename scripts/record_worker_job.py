#!/usr/bin/env python3
"""Write a worker job result that records the image digest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from ohho.worker_images.errors import WorkerImageError
    from ohho.worker_images.results import write_job_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        write_job_result(
            args.out,
            job_id=args.job_id,
            status=args.status,
            image=args.image,
            image_digest=args.digest,
        )
    except WorkerImageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
