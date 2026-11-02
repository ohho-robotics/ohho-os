# ohho-os protocol tag commands (do not run)

**Do not run these commands.** `protocolVersion` `1.0.0` is not in `origin/main` (`85db848268168fa27b4ebb1b5992220b3bfacfcd`). Package version on that commit is `1.1.3`.

`scripts/print-release-commands.sh` prints this file and does not create a tag or a GitHub Release.

Do not run `git tag v1.0.0`, `git push origin v1.0.0`, or any other `v*` tag for this pin. `.github/workflows/release.yml` publishes PyPI on `v*` when the tag matches `pyproject.toml`. Existing package tags: `v1.1.1`, `v1.1.2`.

The protocol commit that defines `PROTOCOL_VERSION = "1.0.0"` is `6f690d50b5991b13a35430581f84626529ab870d` (`ohh-174-protocol-version`). The `RobotDescriptor` loader is a different tip, `389c589cd00619cd12b18b97c636d6fac556337e`. Do not tag one and claim it contains the other until a commit actually contains both.

When `<COMMIT>` is the commit you mean:

```bash
git fetch origin
git checkout <COMMIT>
python -c "from pathlib import Path; t=Path('ohho/protocol.py').read_text(); assert 'PROTOCOL_VERSION = \"1.0.0\"' in t"
git tag -a protocol-v1.0.0 <COMMIT> -m "ohho-os protocolVersion 1.0.0"
git push origin protocol-v1.0.0
gh release create protocol-v1.0.0 \
  --repo ohho-robotics/ohho-os \
  --target <COMMIT> \
  --title "ohho-os protocolVersion 1.0.0" \
  --notes-file CHANGELOG.md
```

## ohho-sdk schemas (other repository)

`https://github.com/ohho-robotics/ohho-sdk` at `5a20f1c23c10c814b8a790536a80b21695b3ed82` has `ohho.telemetry.v1` `1.0.0`, `RobotDescriptor` / `ohho.descriptor.v1`, and `PINNED_PROTOCOL_VERSION` `1.0.0`. Not yet tagged. Not an npm publish.

```bash
git fetch origin
git checkout <COMMIT>
test -f packages/schemas/ohho.telemetry.v1/manifest.json
test -f packages/schemas/ohho.descriptor.v1/descriptor.json
git tag -a schemas-v1.0.0 <COMMIT> -m "ohho-sdk schemas 1.0.0"
git push origin schemas-v1.0.0
gh release create schemas-v1.0.0 \
  --repo ohho-robotics/ohho-sdk \
  --target <COMMIT> \
  --title "ohho-sdk schemas 1.0.0" \
  --notes-file CHANGELOG.md
```

Do not `npm publish` from this sheet.
