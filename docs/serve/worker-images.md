# Serve worker images

Workers that serve a policy, locally or in the cloud, run one of three pinned
images. The tag names the variant and the ohho-os version:

| Image tag | Base | torch | ORT | TensorRT | CUDA | JetPack | L4T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ohho-serve-worker:cpu-1.2.0` | `python:3.12-slim-bookworm` | 2.5.1 | 1.20.1 (`onnxruntime`) | none | none | none | none |
| `ohho-serve-worker:cuda-1.2.0` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | 2.5.1 | 1.20.1 (`onnxruntime-gpu`) | 10.0.1 | 12.4.1 | none | none |
| `ohho-serve-worker:jetson-1.2.0` | `nvcr.io/nvidia/l4t-base:r36.2.0` | 2.5.1 | 1.20.1 (`onnxruntime`) | 8.6.2 | 12.2.12 | 6.0 | 36.2.0 |

The source of truth is `ohho/worker_images/worker_matrix.json`
(`ohho.serve_worker_matrix.v1`). Dockerfiles under `docker/serve-worker/`
are generated from it. Torch 2.5.1 is the CPU wheel for the cpu image and
the cu124 wheel for the cuda image. ORT 1.20.1 is the CUDA 12.x GPU package
on the cuda image and the CPU package on the other two. TensorRT 10.0.1 is
inside the `x86-cuda` target window. TensorRT 8.6.2 is the JetPack 6.0
developer-preview stack for L4T 36.2 and is inside the `jetson-orin-nano`
window.

The jetson image uses the public L4T base already used by
[`.github/workflows/multiarch.yml`](../../.github/workflows/multiarch.yml).
`r36.2.0` is that base's tag. It is Ubuntu 22.04, `linux/arm64`, and the
JetPack 6.0 developer preview (CUDA 12.2.12, TensorRT 8.6.2). The image does
not install git. That base image does not ship git.

## Build

```bash
docker build --file docker/serve-worker/Dockerfile.cpu --tag ohho-serve-worker:cpu-1.2.0 .
docker build --file docker/serve-worker/Dockerfile.cuda --tag ohho-serve-worker:cuda-1.2.0 .
docker build --file docker/serve-worker/Dockerfile.jetson --platform linux/arm64 --tag ohho-serve-worker:jetson-1.2.0 .
```

`OHHO_WORKER_STACK` defaults to `full`, which installs the pinned torch, ORT,
and (on cuda) TensorRT wheels. CI passes `--build-arg OHHO_WORKER_STACK=platform`.
The platform build installs ohho-os only. GitHub-hosted runners have no GPU,
the unit-test job does not install torch, and the L4T image has no git, so CI
does not download those wheels. `ohho targets verify` still checks the image
tag, the CPU architecture, the CUDA image's `CUDA_VERSION`, and the jetson
image's `/etc/nv_tegra_release` against this matrix.

A full jetson wheel install uses the CPU torch index. The JetPack GPU wheel
is not in the public L4T base this image is built from. TensorRT on jetson is
the BSP pin, not a pip package.

## Verify

Inside the image:

```bash
ohho targets verify
ohho targets verify --json
```

The command exits 0 and prints `ok: ohho-serve-worker:<variant>-1.2.0`, or
exits 2 and prints `error:`. It does not set `OHHO_ARM_HARDWARE` and it does
not open a robot. With `OHHO_WORKER_STACK=full` (or `--require-stack`) the
installed torch, ONNX Runtime, and pip TensorRT versions must match the pins.
A package that is present in a platform image must still match. A missing
package is accepted only when the stack does not require it.

## Agent allowlist

`OHHO_WORKER_IMAGE` selects the image an agent or `ohho serve` would run.
When it is unset, the agent runs in-process as before. When it is set, the
tag must be one of the three pinned tags, and it must be on the console
allowlist. The default allowlist is the matrix. `OHHO_WORKER_ALLOWLIST` is a
path, or a JSON object, that narrows it:

```json
{
  "schema": "ohho.worker_allowlist.v1",
  "images": [
    {"tag": "ohho-serve-worker:cpu-1.2.0"},
    {
      "tag": "ohho-serve-worker:cuda-1.2.0",
      "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

A tag outside the matrix is refused even if the file names it. A pinned tag
the file omits is refused. When an entry has a digest, the running digest
(`OHHO_WORKER_IMAGE_DIGEST`, or `@sha256:` on the tag) must match. The agent
returns `refused:` and does not call the brain. `ohho agent` and `ohho serve`
exit 2 before they connect or listen.

## Job results

`ohho.worker_job_result.v1` records `image` and `image_digest` (`sha256:` and
64 hex characters). CI writes `worker-job-result.json` from `docker image
inspect` `Id` after verify, and prints `image_digest` in the job summary.
That id is the image digest available before a registry push. These images
are not pushed.

## SBOM, scan, and signatures

Workflow: [`.github/workflows/serve-worker-images.yml`](../../.github/workflows/serve-worker-images.yml).

On every pull request and on `main`, each variant is built, `ohho targets
verify` runs in the container, syft writes CycloneDX and SPDX SBOMs, and
grype scans the CycloneDX file. `scripts/worker_vuln_gate.py --python-only`
fails the job on a fixed CRITICAL finding in a Python package. The full scan
is uploaded. Unfixed findings, and fixed findings in OS packages from the
base image, do not fail the job. Ids listed in
`docker/serve-worker/scan-exceptions.json` are ignored.

On a `v*` tag, cosign keyless-signs the digest file (`cosign sign-blob`) and
the SBOM, signature bundle, and job result are attached to the GitHub
release. Creating the tag is a human step. The workflow does not push the
image and does not publish a package.
