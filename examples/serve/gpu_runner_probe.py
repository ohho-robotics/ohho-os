"""See whether a self-hosted GPU runner is online.

The nightly workflow runs this on a GitHub-hosted runner. It lists this
repository's self-hosted runners and writes ``available=true`` only when one
is online and labeled ``linux`` and ``gpu``. A missing token, a refused API
call, or no matching runner writes ``available=false`` and exits 0 so the
GPU job is skipped.

The example scripts themselves do not call this and do not open the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def label_names(runner: dict[str, Any]) -> set[str]:
    """Lower-cased label names from one runners API object."""
    names: set[str] = set()
    labels = runner.get("labels")
    if not isinstance(labels, list):
        return names
    for item in labels:
        if isinstance(item, str):
            names.add(item.casefold())
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"].casefold())
    return names


def runner_available(payload: dict[str, Any]) -> bool:
    """True when ``payload`` lists an online runner labeled linux and gpu."""
    runners = payload.get("runners")
    if not isinstance(runners, list):
        return False
    required = {"linux", "gpu"}
    for runner in runners:
        if not isinstance(runner, dict):
            continue
        if runner.get("status") != "online":
            continue
        if required <= label_names(runner):
            return True
    return False


def _fetch(token: str, repo: str) -> bool:
    url = f"https://api.github.com/repos/{repo}/actions/runners?per_page=100"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ohho-serve-examples",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"runner probe skipped: {exc}")
        return False
    if not isinstance(payload, dict):
        return False
    return runner_available(payload)


def write_output(available: bool) -> None:
    """Append the Actions output line when ``GITHUB_OUTPUT`` is set."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    value = "true" if available else "false"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"available={value}\n")


def main() -> int:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    repo = os.environ.get("GH_REPO") or os.environ.get("GITHUB_REPOSITORY") or ""
    available = _fetch(token, repo) if token and repo else False
    write_output(available)
    print(f"available={'true' if available else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
