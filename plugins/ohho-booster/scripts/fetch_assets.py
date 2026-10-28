"""Download booster_assets at the pinned commit.

Meshes land in the cache directory (default ``~/.ohho/assets/booster``).
They are not committed to this repository.

    python plugins/ohho-booster/scripts/fetch_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ohho_booster.assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
