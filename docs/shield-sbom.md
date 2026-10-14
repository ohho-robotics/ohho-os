# Shield v0 — SBOM + dependency vulnerability scan (ohho-sdk)

OhhO **Shield v0** for this repository is an honest **SBOM + dependency vulnerability
scan**. It is **not** a security product, a SOC dashboard, or a device attestation
platform.

## What runs in CI

Workflow: [`.github/workflows/shield-sbom.yml`](../.github/workflows/shield-sbom.yml)

1. **syft** generates SPDX and CycloneDX JSON SBOMs for the checkout (including
   `pyproject.toml` optional extras).
2. A CI venv installs `ohho-os` with a representative optional-extra set so
   transitive Python packages appear in the SBOM and audits.
3. **osv-scanner** scans the CycloneDX SBOM and the repository tree.
4. **pip-audit** audits the frozen `pip freeze` from that venv.
5. **`scripts/vuln_gate.py`** fails the job only on **CRITICAL** findings (CVSS ≥
   9.0 in OSV records, or explicit `CRITICAL` severity in pip-audit JSON).

Artifacts (every run): `sbom.cdx.json`, `sbom.cyclonedx.json`, `sbom.spdx.json`, scan JSON, frozen
requirements.

**Release tags (`v*`):** the same SBOM files are attached to the GitHub Release (in
addition to workflow artifacts). Creating releases/tags is a human step; CI only
uploads when a tag is pushed.

## Local reproduction

```bash
python -m pip install -e ".[dev,yaml,serial,serve,data,sim-serve,agent]"
python -m pip freeze > /tmp/requirements-shield-frozen.txt
# Install syft + osv-scanner from https://github.com/anchore/syft and
# https://github.com/google/osv-scanner, then:
syft scan dir:. -o cyclonedx-json=/tmp/sbom.cdx.json
osv-scanner scan -f json --output-file=/tmp/osv-sbom.json -L /tmp/sbom.cdx.json
pip-audit --format json -o /tmp/pip-audit.json
python scripts/vuln_gate.py --osv /tmp/osv-sbom.json --pip-audit /tmp/pip-audit.json
```

## Documented exceptions

None at introduction. If a CRITICAL finding cannot be upgraded immediately, add an
entry here with CVE/GHSA id, affected package, justification, and planned fix date,
and add a matching `[[IgnoredVulns]]` block in `osv-scanner.toml` with the same
reason (scanner ignore + human-readable doc).

## Follow-up repos (not in this change)

Copy the same workflow pattern to **OmniBot** and **ohho-robotics.com** (npm audit
or osv-scanner on `package-lock.json` / pnpm lockfile). Keep public wording:
**SBOM + dependency vulnerability scan**, not a security product.
