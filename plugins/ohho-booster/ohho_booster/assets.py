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
    _robot_entries(data)
    return data


def archive_url(manifest: Mapping[str, Any] | None = None) -> str:
    """GitHub archive URL for the pinned description commit."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    upstream = str(pin["upstream"]).rstrip("/")
    return f"{upstream}/archive/{pin['commit']}.tar.gz"


def policy_url(manifest: Mapping[str, Any] | None = None, *, name: str = "walk") -> str:
    """Raw URL for one pinned locomotion checkpoint. ``walk`` is T2."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    policies = pin["policies"]
    if name not in policies:
        raise ValueError(f"asset manifest has no policies.{name}")
    walk = policies[name]
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
    name: str = "walk",
) -> Path:
    """Download one pinned locomotion checkpoint and return its path.

    ``name`` selects ``policies.<name>``. The default ``walk`` is the T2
    checkpoint, which is what existing callers fetch. A file already in the
    cache is reused when its sha256 matches the pin. ``url`` overrides the
    download location for tests.
    """
    pin = load_manifest()
    policies = pin["policies"]
    if name not in policies:
        raise ValueError(f"asset manifest has no policies.{name}")
    walk = policies[name]
    commit = str(walk["commit"])
    expected = str(walk["sha256"])
    filename = Path(str(walk["path"])).name
    target = Path(dest) if dest is not None else default_cache_dir()
    path = target / "policies" / commit / filename
    if url is None and path.is_file() and _sha256_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(url or policy_url(pin, name=name), path, timeout=timeout)
    _require_sha256(path, expected, filename)
    return path


def description_repository(manifest: Mapping[str, Any] | None = None) -> str:
    """``owner/name`` for the pinned description archive."""
    pin = dict(manifest) if manifest is not None else load_manifest()
    upstream = str(pin["upstream"]).strip()
    marker = "https://github.com/"
    if not upstream.startswith(marker):
        raise ValueError("description upstream must be https://github.com/owner/name")
    repository = upstream[len(marker) :].strip("/")
    if repository.endswith(".git"):
        repository = repository[: -len(".git")]
    return repository


def matrix_checkout(root: Path | None = None) -> Path | None:
    """Sim-matrix checkout of this plugin when the pinned MJCF is already there.

    The workflow checks the description repository out at
    ``.sim-assets/<plugin>``. That tree is the repository root, not the
    extracted archive layout.
    """
    base = Path(root) if root is not None else Path(".sim-assets") / "booster"
    mjcf = base / str(load_manifest()["paths"]["mjcf"])
    if mjcf.is_file():
        return base
    return None


def materialise_checkout(dest: Path) -> tuple[Path, Path]:
    """Return ``(mjcf, checkpoint)`` inside ``dest``.

    A git checkout that already contains the MJCF is reused. The locomotion
    checkpoint is a second pin and is stored under ``dest`` so the matrix
    cache, keyed on the description commit, holds both. A matching checkpoint
    is not downloaded again.
    """
    pin = load_manifest()
    root = Path(dest)
    mjcf = root / str(pin["paths"]["mjcf"])
    if not mjcf.is_file():
        fetched = fetch_assets(root)
        mjcf = fetched / str(pin["paths"]["mjcf"])
    checkpoint = _fetch_policies(root)
    return mjcf, checkpoint


def ensure_assets(
    dest: Path | None = None, *, robot_id: str = "booster-t2"
) -> tuple[Path, Path]:
    """Return ``(mjcf, checkpoint)`` for ``robot_id``, fetching anything missing.

    The default robot is ``booster-t2``. A sim-matrix checkout at
    ``.sim-assets/booster`` wins over the home cache when ``dest`` is omitted,
    so CI does not download the mesh archive twice.
    """
    if dest is None:
        checkout = matrix_checkout()
        if checkout is not None:
            materialise_checkout(checkout)
            return _pair_from_root(checkout, robot_id, policy_dest=checkout)
    root = fetch_assets(dest)
    return _pair_from_root(root, robot_id, policy_dest=dest)


def _robot_files(
    pin: Mapping[str, Any], robot_id: str
) -> tuple[Mapping[str, Any], str]:
    """Relative description paths and the policy name for ``robot_id``.

    T2 stays on the top-level ``paths`` and ``policies.walk`` so older pins
    still resolve. Other ids come from ``robots``.
    """
    if robot_id == "booster-t2":
        return pin["paths"], "walk"
    robots = pin.get("robots")
    if not isinstance(robots, dict) or robot_id not in robots:
        raise ValueError(f"asset manifest has no robot {robot_id}")
    entry = robots[robot_id]
    if not isinstance(entry, dict):
        raise ValueError(f"asset manifest robots.{robot_id} must be an object")
    return entry, str(entry["policy"])


def _pair_from_root(
    root: Path, robot_id: str, *, policy_dest: Path | None
) -> tuple[Path, Path]:
    pin = load_manifest()
    paths, policy_name = _robot_files(pin, robot_id)
    rel = paths.get("mjcf")
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"asset manifest has no mjcf for {robot_id}")
    mjcf = root / rel
    if not mjcf.is_file():
        raise ValueError(f"pinned description is missing {rel}")
    # ``policy_dest`` None is the home cache. A matrix checkout passes itself.
    checkpoint = fetch_policy(policy_dest, name=policy_name)
    return mjcf, checkpoint


def _fetch_policies(root: Path) -> Path:
    """Store every pinned checkpoint under ``root``. Return the walk file."""
    pin = load_manifest()
    checkpoint: Path | None = None
    for name in pin["policies"]:
        path = fetch_policy(root, name=name)
        if name == "walk":
            checkpoint = path
    if checkpoint is None:
        raise ValueError("asset manifest policies.walk is required")
    return checkpoint


def _policy_pin(policies: object) -> None:
    if not isinstance(policies, dict) or "walk" not in policies:
        raise ValueError("asset manifest policies.walk is required")
    for name, walk in policies.items():
        if not isinstance(walk, dict):
            raise ValueError(f"asset manifest policies.{name} must be an object")
        for key in ("upstream", "commit", "path", "sha256", "licence"):
            if key not in walk:
                raise ValueError(f"asset manifest policies.{name} missing {key}")
        commit = walk["commit"]
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"policies.{name} commit must be a 40-character sha")
        path = walk["path"]
        if not isinstance(path, str) or not path or path.startswith(("/", "\\")):
            raise ValueError(f"policies.{name} path must be a relative path")
        if ".." in Path(path).parts:
            raise ValueError(f"policies.{name} path must stay inside the upstream tree")
        walk["sha256"] = _sha256_text(walk["sha256"], f"policies.{name} sha256")


def _robot_entries(data: Mapping[str, Any]) -> None:
    """Optional ``robots`` object. T2 stays on the top-level paths."""
    robots = data.get("robots")
    if robots is None:
        return
    if not isinstance(robots, dict):
        raise ValueError("asset manifest robots must be an object")
    policies = data["policies"]
    for robot_id, entry in robots.items():
        if not isinstance(entry, dict):
            raise ValueError(f"asset manifest robots.{robot_id} must be an object")
        for key in ("urdf", "mjcf", "meshes", "policy"):
            if key not in entry or not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"asset manifest robots.{robot_id}.{key} is required")
            if ".." in Path(str(entry[key])).parts:
                raise ValueError(
                    f"asset manifest robots.{robot_id}.{key} must stay relative"
                )
        if entry["policy"] not in policies:
            raise ValueError(
                f"asset manifest robots.{robot_id} policy {entry['policy']!r} is missing"
            )


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
    print(root)
    pin = load_manifest()
    for name in pin["policies"]:
        url = args.policy_url if name == "walk" else None
        print(fetch_policy(args.dest, url=url, name=name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
