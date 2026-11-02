"""Pinned booster_assets fetch. Meshes are cached, never committed.

The pin (commit, upstream URL, licence) lives in ``assets_manifest.json``.
This module uses the standard library only. It does not import a vendor SDK.
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping

_MANIFEST_NAME = "assets_manifest.json"
_ARCHIVE_ROOT_PREFIX = "booster_assets-"


def manifest_path() -> Path:
    """Path to the committed pin file inside this package."""
    return Path(__file__).with_name(_MANIFEST_NAME)


def load_manifest() -> dict[str, Any]:
    """Read the committed asset pin."""
    data = json.loads(manifest_path().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("asset manifest must be an object")
    for key in ("name", "upstream", "commit", "licence", "copyright", "paths"):
        if key not in data:
            raise ValueError(f"asset manifest missing {key}")
    commit = data["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("asset manifest commit must be a 40-character sha")
    paths = data["paths"]
    if not isinstance(paths, dict):
        raise ValueError("asset manifest paths must be an object")
    return data


def archive_url(manifest: Mapping[str, Any] | None = None) -> str:
    """GitHub archive URL for the pinned commit."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    upstream = str(pin["upstream"]).rstrip("/")
    return f"{upstream}/archive/{pin['commit']}.tar.gz"


def default_cache_dir() -> Path:
    """Where a human-run fetch stores the archive. Not inside the repo."""
    return Path.home() / ".ohho" / "assets" / "booster"


def archive_root_name(commit: str) -> str:
    """Top-level directory GitHub uses for ``archive/<commit>.tar.gz``."""
    return f"{_ARCHIVE_ROOT_PREFIX}{commit}"


def fetch_assets(
    dest: Path | None = None,
    *,
    url: str | None = None,
    timeout: float = 600,
) -> Path:
    """Download the pinned archive into ``dest`` and return the extracted root.

    ``url`` overrides the download location so tests can pass a local archive.
    The archive's top-level directory must still be the pinned commit. A
    different commit is refused.
    """
    pin = load_manifest()
    commit = str(pin["commit"])
    target = Path(dest) if dest is not None else default_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{archive_root_name(commit)}.tar.gz"
    _download(url or archive_url(pin), archive, timeout=timeout)
    extract_dir = target / commit
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    root = _extract_pinned(archive, extract_dir, commit=commit)
    _require_paths(root, pin["paths"])
    receipt = {
        "upstream": pin["upstream"],
        "commit": commit,
        "licence": pin["licence"],
        "copyright": pin["copyright"],
        "archive_url": archive_url(pin),
        "extracted": root.name,
    }
    (extract_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _download(url: str, dest: Path, *, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ohho-booster-assets"})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        dest.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)


def _member_parts(name: str) -> list[str]:
    normalised = name.replace("\\", "/")
    if normalised.startswith("/"):
        raise ValueError(f"unsafe path in asset archive: {name}")
    parts = [part for part in normalised.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe path in asset archive: {name}")
    return parts


def _extract_pinned(archive: Path, dest: Path, *, commit: str) -> Path:
    root_name = archive_root_name(commit)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            parts = _member_parts(member.name)
            if not parts or parts[0] != root_name:
                found = parts[0] if parts else ""
                raise ValueError(
                    f"asset archive root {found!r} does not match pinned commit {commit}"
                )
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"asset archive member is not a file: {member.name}")
        for member in members:
            # ``filter`` exists on 3.12+. 3.10 still extracts after the checks above.
            if sys.version_info >= (3, 12):
                tar.extract(member, dest, filter="data")
            else:
                tar.extract(member, dest)
    root = dest / root_name
    if not root.is_dir():
        raise ValueError(f"pinned archive did not contain {root_name}")
    return root


def _require_paths(root: Path, paths: Mapping[str, Any]) -> None:
    for key in ("urdf", "mjcf", "meshes"):
        rel = paths.get(key)
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"asset manifest paths.{key} must be a relative path")
        candidate = root / rel
        if not candidate.exists():
            raise ValueError(f"pinned archive is missing {rel}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m ohho_booster.assets``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download booster_assets at the pinned commit. Meshes are not committed."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="cache directory (default: ~/.ohho/assets/booster)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="archive URL (default: the pinned GitHub archive)",
    )
    args = parser.parse_args(argv)
    root = fetch_assets(args.dest, url=args.url)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
