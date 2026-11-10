"""Capture a versioned parity set from real inputs and an FP32 reference.

The reference is executed once. Its outputs are the golden outputs stored
beside the inputs. Random inputs are refused: capture is for a LeRobot
dataset, a sim rollout, or a camera log.
"""

from __future__ import annotations

from pathlib import Path

from ohho.parity.errors import ParityError
from ohho.parity.inputs import (
    bundle_from_frames,
    load_observation_frames,
    parse_input_source,
)
from ohho.parity.runners import prepare_reference, run_prepared
from ohho.parity.sets import ParitySet, write_parity_set


def capture_parity_set(
    reference: str,
    inputs: str,
    *,
    out: str | Path,
    set_id: str,
    version: int = 1,
    n: int = 256,
    seed: int = 0,
    record_dir: str | None = None,
    submodule: str | None = None,
) -> ParitySet:
    """Run ``reference`` on ``n`` real inputs and write the set to ``out``.

    ``out`` is a directory, or a ``.tar.gz`` / ``.tgz`` path. The FP32 module
    (or the reference ONNX file) produces the stored outputs. A later parity
    run that passes this set compares the candidate to those stored outputs.
    """
    if n < 1:
        raise ParityError("--n must be at least 1.")
    if type(seed) is not int:
        raise ParityError("--seed must be an integer.")
    if not isinstance(reference, str) or not reference.strip():
        raise ParityError("--reference is required.")
    kind, _path = parse_input_source(inputs)
    if kind == "random":
        raise ParityError(
            "parity capture needs a LeRobot dataset, a sim rollout (recorded), "
            "or a camera log."
        )
    frames = load_observation_frames(inputs, n=n, seed=seed, record_dir=record_dir)
    if frames.source not in {"dataset", "recorded", "camera"}:
        raise ParityError(
            "parity capture needs a LeRobot dataset, a sim rollout (recorded), "
            "or a camera log."
        )
    prepared = prepare_reference(reference.strip(), submodule)
    if not prepared.inputs:
        raise ParityError(f"{prepared.ref} has no input shapes.")
    bundle = bundle_from_frames(frames, prepared.inputs)
    outputs = run_prepared(prepared, bundle.batches)
    media = frames.media if frames.media else None
    return write_parity_set(
        out,
        set_id=set_id,
        version=version,
        model=prepared.ref,
        submodule=submodule,
        source=frames.source,
        source_path=frames.path,
        seed=seed,
        n_requested=n,
        indices=list(frames.indices),
        batches=bundle.batches,
        outputs=outputs,
        media_files=media,
    )
