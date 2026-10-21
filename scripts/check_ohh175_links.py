#!/usr/bin/env python3
"""HEAD-check ohho-os / ohho-sdk URLs touched by OHH-175 (stdlib only)."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

URLS = [
    "https://github.com/ohho-robotics/ohho-os",
    "https://github.com/ohho-robotics/ohho-sdk",
    "https://github.com/ohho-robotics/ohho-os/blob/main/LICENSE",
    "https://github.com/ohho-robotics/ohho-sdk/blob/main/LICENSE",
    "https://github.com/ohho-robotics/ohho-os/blob/main/pyproject.toml",
    "https://pypi.org/project/ohho-os/",
]


def check(url: str) -> tuple[bool, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ohh-175-link-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.status
            if code in (200, 301, 302, 303, 307, 308):
                return True, str(code)
            return False, str(code)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308, 405):
            return True, f"HTTP {exc.code}"
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 — log all failures
        return False, repr(exc)


def main() -> int:
    failed = 0
    for url in URLS:
        ok, detail = check(url)
        status = "OK" if ok else "FAIL"
        print(f"{status}\t{detail}\t{url}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
