# Releasing `ohho-os` to PyPI

The Python package **`ohho-os`** (import `ohho`) is published from this repository
([ohho-robotics/ohho-os](https://github.com/ohho-robotics/ohho-os)) when a `v*`
git tag is pushed. The TypeScript npm workspace lives in
[ohho-robotics/ohho-sdk](https://github.com/ohho-robotics/ohho-sdk); it is not
published by this workflow.

## PyPI trusted publisher (manual — do before the next tag)

After the GitHub rename (`ohho-sdk` → `ohho-os` for the Python engine), update
the trusted publisher on [pypi.org](https://pypi.org) for project **`ohho-os`**:

| Field | Value |
| --- | --- |
| **Owner** | `ohho-robotics` |
| **Repository name** | `ohho-os` |
| **Workflow name** | `release.yml` |
| **Environment name** | `pypi` (GitHub Actions environment on `ohho-robotics/ohho-os`) |

Remove or disable any trusted publisher still bound to repository **`ohho-sdk`**
for this package.

### TestPyPI (optional dry run)

To verify trusted publishing without touching production PyPI:

1. On [test.pypi.org](https://test.pypi.org), add a trusted publisher for
   **`ohho-os`** with the same owner/repo/workflow, environment **`testpypi`**.
2. In GitHub → **ohho-robotics/ohho-os** → Settings → Environments, create
   **`testpypi`** if it does not exist (no secrets required for trusted publishing).
3. Run **Actions → Release → Run workflow** (`workflow_dispatch`). That job builds
   and uploads to TestPyPI only; it does not run the post-publish `pip install`
   smoke job.

## Production release

1. Bump `version` in `pyproject.toml` and commit on `main`.
2. Tag and push (human only):

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. `.github/workflows/release.yml` on tag `v*`:
   - Validates trove classifiers
   - Fails if tag version ≠ `pyproject.toml` version
   - `python -m build` + `twine check`
   - Publishes via `pypa/gh-action-pypi-publish` (environment **`pypi`**, OIDC)
   - On Ubuntu and macOS, installs `ohho-os==X.Y.Z` from PyPI (with retries) and
     runs `ohho doctor`

## Checklist after rename (OHH-171)

- [ ] PyPI trusted publisher: owner `ohho-robotics`, repo **`ohho-os`**, workflow
      `release.yml`, environment `pypi`
- [ ] TestPyPI trusted publisher + `testpypi` environment (if using dry runs)
- [ ] First tag cut from **`ohho-os`** confirms the workflow completes end to end
