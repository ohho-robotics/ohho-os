"""Tests for scripts/vuln_gate.py (Shield v0 critical gate)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from scripts.vuln_gate import (
    critical_from_osv_scanner,
    critical_from_pip_audit,
    is_critical_osv_vuln,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


class TestOsvSeverity(unittest.TestCase):
    def test_critical_by_cvss(self) -> None:
        vuln = {
            "id": "GHSA-test",
            "severity": [{"type": "CVSS_V3", "score": "9.8"}],
        }
        self.assertTrue(is_critical_osv_vuln(vuln))

    def test_not_critical_below_threshold(self) -> None:
        vuln = {
            "id": "GHSA-test",
            "severity": [{"type": "CVSS_V3", "score": "8.9"}],
        }
        self.assertFalse(is_critical_osv_vuln(vuln))

    def test_critical_from_osv_json(self) -> None:
        payload = {
            "results": [
                {
                    "source": {"path": "/tmp/req.txt", "type": "lockfile"},
                    "packages": [
                        {
                            "package": {
                                "name": "badpkg",
                                "version": "1.0.0",
                                "ecosystem": "PyPI",
                            },
                            "vulnerabilities": [
                                {
                                    "id": "CVE-TEST-1",
                                    "severity": [{"type": "CVSS_V3", "score": "10.0"}],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        findings = critical_from_osv_scanner(payload, source_label="osv.json")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].vuln_id, "CVE-TEST-1")


class TestPipAuditSeverity(unittest.TestCase):
    def test_ignores_without_critical_label(self) -> None:
        payload = {
            "dependencies": [
                {
                    "name": "foo",
                    "version": "1.0",
                    "vulns": [{"id": "PYSEC-1", "severity": "HIGH"}],
                }
            ]
        }
        self.assertEqual(critical_from_pip_audit(payload, source_label="pip.json"), [])

    def test_flags_explicit_critical(self) -> None:
        payload = {
            "dependencies": [
                {
                    "name": "foo",
                    "version": "1.0",
                    "vulns": [{"id": "PYSEC-CRIT", "severity": "CRITICAL"}],
                }
            ]
        }
        findings = critical_from_pip_audit(payload, source_label="pip.json")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].vuln_id, "PYSEC-CRIT")


class TestVulnGateCli(unittest.TestCase):
    def test_main_passes_on_empty_results(self) -> None:
        empty_osv = ROOT / "tests" / "fixtures" / "osv-empty.json"
        self.assertEqual(main(["--osv", str(empty_osv)]), 0)

    def test_subprocess_module_runnable(self) -> None:
        empty_osv = ROOT / "tests" / "fixtures" / "osv-empty.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "vuln_gate.py"),
                "--osv",
                str(empty_osv),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
