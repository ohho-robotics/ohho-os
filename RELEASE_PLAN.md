# RELEASE_PLAN — OHH-71 (draft)

**Prepared:** 2026-10-14  
**Branch:** `ohh-71-release-changelogs` (ohho-sdk / ohho-os worktree)  
**Status:** planning only — **nothing here has been tagged, published, or released.**

This document is a morning checklist for **Varun** to ship:

| Artifact | Repo | Planned version | Draft changelog |
|----------|------|-----------------|-----------------|
| **ohho-os** (PyPI `ohho-os`) | [ohho-robotics/ohho-sdk](https://github.com/ohho-robotics/ohho-sdk) | **1.3.0** | [`CHANGELOG.md`](CHANGELOG.md) |
| **OmniBot** (GitHub tag + optional Release notes) | [ohho-robotics/OmniBot](https://github.com/ohho-robotics/OmniBot) | **v1.0.0** | Copy [`docs/releases/omnibot-v1.0-CHANGELOG.md`](docs/releases/omnibot-v1.0-CHANGELOG.md) → `OmniBot/CHANGELOG.md` |

Do **not** run `gh release create`, `twine upload`, `npm publish`, or HF uploads from
this prep branch. Cloud agents must not merge this PR or push release tags.

---

## Pre-flight (both repos)

- [ ] Merge the OHH-71 PR on **ohho-sdk** after review (changelog + this plan only).
- [ ] Confirm **CI green** on `main` for ohho-sdk and OmniBot (`ci.yml` on each).
- [ ] Skim changelogs for wording; add any last-minute merged PRs since the draft baseline.
- [ ] Decide whether **1.1.3** should be skipped on PyPI (recommended: yes — jump **1.1.2 → 1.3.0** as drafted).

---

## ohho-os 1.3.0 — version bump (required before tag)

Current tree on `main` still declares **1.1.3**. Before tagging:

- [ ] Set `version = "1.3.0"` in [`pyproject.toml`](pyproject.toml).
- [ ] Set `__version__ = "1.3.0"` in [`ohho/__init__.py`](ohho/__init__.py).
- [ ] Update version strings in [`README.md`](README.md) header / Releasing section if you want them to match at tag time.
- [ ] Move the `## [1.3.0] — not released yet` section in [`CHANGELOG.md`](CHANGELOG.md) to a dated **released** entry (or add release date).
- [ ] `pip install -e .` and `python -m unittest discover -s tests`
- [ ] `ruff format ohho tests` and `ruff check ohho tests`
- [ ] `ohho doctor` and `ohho selftest --sim`
- [ ] Commit version bump on `main` (separate PR or direct — your call).

---

## ohho-os 1.3.0 — tag and PyPI (manual)

Pushing tag `v1.3.0` triggers [`.github/workflows/release.yml`](.github/workflows/release.yml)
(trusted publishing to PyPI, version must match `pyproject.toml`).

```powershell
cd C:\Users\varun\ohho\ohho-sdk   # or your primary ohho-os checkout on main
git fetch origin
git checkout main
git pull origin main

# After version bump is on main:
git tag -a v1.3.0 -m "ohho-os 1.3.0"
git push origin v1.3.0
```

- [ ] Watch GitHub Actions **Release** workflow to completion.
- [ ] Verify PyPI: `pip install "ohho-os==1.3.0"` in a fresh venv; `ohho doctor`.
- [ ] Optional: create a **GitHub Release** for tag `v1.3.0` with body pasted from
  `CHANGELOG.md` — only if you want GitHub release notes (workflow does not require it).

**Do not** push a `v*` tag until the version bump is merged; a mismatched tag fails the workflow.

---

## OmniBot v1.0.0 — tag (manual)

OmniBot has **no** PyPI/npm release workflow in this prep. Typical steps:

```powershell
cd C:\Users\varun\ohho\OmniBot
git fetch origin
git checkout main
git pull origin main

# Add changelog at repo root (from ohho-sdk draft):
# copy docs\releases\omnibot-v1.0-CHANGELOG.md → CHANGELOG.md (edit header if needed)
git add CHANGELOG.md
git commit -m "Add CHANGELOG for v1.0.0"
git push origin main

git tag -a v1.0.0 -m "OmniBot v1.0.0"
git push origin v1.0.0
```

- [ ] Optional: `gh release create v1.0.0 --notes-file CHANGELOG.md` (human only — not for agents).
- [ ] Update OmniBot README badges / “ohho-os” PyPI pin to **1.3.0** when PyPI is live (currently references 1.1.2 in places).
- [ ] Post-release: note sim vs hardware validation gaps honestly in release text (see OmniBot changelog).

---

## Explicit out of scope for OHH-71

- Merging this PR without human review
- Creating GitHub Releases from automation
- `twine upload`, `npm publish`, container registry pushes, Hugging Face uploads
- Bumping ohho-os version on this branch (optional follow-up PR on `main` before tag)

---

## Draft PR

After push: open PR `ohh-71-release-changelogs` → `main` titled **OHH-71: draft changelogs and release plan (no publish)**.
