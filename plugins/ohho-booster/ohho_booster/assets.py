"""Pinned description and locomotion-checkpoint fetch.

The pin (commit, sha256, upstream URL, licence) lives in
``assets_manifest.json``. Files are cached under the user's home directory
and are never committed. This module uses the standard library only. It does
not import a vendor SDK.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping

_MANIFEST_NAME = "assets_manifest.json"
_ARCHIVE_ROOT_PREFIX = "booster_assets-"
_HEX = frozenset("0123456789abcdef")


def manifest_path() -> Path:
    """Path to the committed pin file inside this package."""
    return Path(__file__).with_name(_MANIFEST_NAME)


def load_manifest() -> dict[str, Any]:
    """Read the committed asset pin."""
    data = json.loads(manifest_path().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("asset manifest must be an object")
    for key in (
        "name",
        "upstream",
        "commit",
        "sha256",
        "licence",
        "copyright",
        "paths",
        "policies",
    ):
        if key not in data:
            raise ValueError(f"asset manifest missing {key}")
    commit = data["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("asset manifest commit must be a 40-character sha")
    data["sha256"] = _sha256_text(data["sha256"], "asset manifest sha256")
    paths = data["paths"]
    if not isinstance(paths, dict):
        raise ValueError("asset manifest paths must be an object")
    _policy_pin(data["policies"])
    return data


def archive_url(manifest: Mapping[str, Any] | None = None) -> str:
    """GitHub archive URL for the pinned description commit."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    upstream = str(pin["upstream"]).rstrip("/")
    return f"{upstream}/archive/{pin['commit']}.tar.gz"


def policy_url(manifest: Mapping[str, Any] | None = None) -> str:
    """Raw URL for the pinned locomotion checkpoint."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    walk = pin["policies"]["walk"]
    upstream = str(walk["upstream"]).rstrip("/")
    commit = str(walk["commit"])
    rel = str(walk["path"]).lstrip("/")
    marker = "https://github.com/"
    if upstream.startswith(marker):
        repo = upstream[len(marker) :]
        return f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"
    return f"{upstream}/raw/{commit}/{rel}"


def default_cache_dir() -> Path:
    """Where a fetch stores archives. Not inside the repo."""
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
    """Download the pinned description archive and return the extracted root.

    ``url`` overrides the download location so tests can pass a local archive.
    The archive's top-level directory must still be the pinned commit, and
    the bytes must match the pinned sha256. A cached extract is reused when
    ``url`` is omitted and its receipt matches the pin.
    """
    pin = load_manifest()
    commit = str(pin["commit"])
    expected = str(pin["sha256"])
    target = Path(dest) if dest is not None else default_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    if url is None:
        cached = _cached_description(target, pin)
        if cached is not None:
            return cached
    archive = target / f"{archive_root_name(commit)}.tar.gz"
    if not archive.is_file() or _sha256_file(archive) != expected:
        _download(url or archive_url(pin), archive, timeout=timeout)
    _require_sha256(archive, expected, "description archive")
    extract_dir = target / commit
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    root = _extract_pinned(archive, extract_dir, commit=commit)
    _require_paths(root, pin["paths"])
    receipt = {
        "upstream": pin["upstream"],
        "commit": commit,
        "sha256": expected,
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


def fetch_policy(
    dest: Path | None = None,
    *,
    url: str | None = None,
    timeout: float = 120,
) -> Path:
    """Download the pinned locomotion checkpoint and return its path.

    A file already in the cache is reused when its sha256 matches the pin.
    ``url`` overrides the download location for tests.
    """
    pin = load_manifest()
    walk = pin["policies"]["walk"]
    commit = str(walk["commit"])
    expected = str(walk["sha256"])
    filename = Path(str(walk["path"])).name
    target = Path(dest) if dest is not None else default_cache_dir()
    path = target / "policies" / commit / filename
    if url is None and path.is_file() and _sha256_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(url or policy_url(pin), path, timeout=timeout)
    _require_sha256(path, expected, filename)
    return path


def ensure_assets(dest: Path | None = None) -> tuple[Path, Path]:
    """Return ``(mjcf, checkpoint)`` from the cache, fetching anything missing."""
    root = fetch_assets(dest)
    checkpoint = fetch_policy(dest)
    mjcf = root / str(load_manifest()["paths"]["mjcf"])
    return mjcf, checkpoint


def _policy_pin(policies: object) -> None:
    if not isinstance(policies, dict) or "walk" not in policies:
        raise ValueError("asset manifest policies.walk is required")
    walk = policies["walk"]
    if not isinstance(walk, dict):
        raise ValueError("asset manifest policies.walk must be an object")
    for key in ("upstream", "commit", "path", "sha256", "licence"):
        if key not in walk:
            raise ValueError(f"asset manifest policies.walk missing {key}")
    commit = walk["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("policy commit must be a 40-character sha")
    path = walk["path"]
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")):
        raise ValueError("policy path must be a relative path")
    if ".." in Path(path).parts:
        raise ValueError("policy path must stay inside the upstream tree")
    walk["sha256"] = _sha256_text(walk["sha256"], "policy sha256")


def _sha256_text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a 64-character sha256")
    lowered = value.strip().lower()
    if len(lowered) != 64 or any(char not in _HEX for char in lowered):
        raise ValueError(f"{where} must be a 64-character sha256")
    return lowered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> None:
    got = _sha256_file(path)
    if got != expected:
        path.unlink(missing_ok=True)
        raise ValueError(f"{label} sha256 {got} does not match pin {expected}")


def _cached_description(target: Path, pin: Mapping[str, Any]) -> Path | None:
    commit = str(pin["commit"])
    root = target / commit / archive_root_name(commit)
    receipt_path = target / commit / "receipt.json"
    if not root.is_dir() or not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if receipt.get("commit") != commit or receipt.get("sha256") != pin["sha256"]:
        return None
    try:
        _require_paths(root, pin["paths"])
    except ValueError:
        return None
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
        description=(
            "Download the pinned description archive and locomotion checkpoint. "
            "Neither file is committed."
        )
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
        help="description archive URL (default: the pinned GitHub archive)",
    )
    parser.add_argument(
        "--policy-url",
        default=None,
        help="checkpoint URL (default: the pinned file)",
    )
    args = parser.parse_args(argv)
    root = fetch_assets(args.dest, url=args.url)
    checkpoint = fetch_policy(args.dest, url=args.policy_url)
    print(root)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
