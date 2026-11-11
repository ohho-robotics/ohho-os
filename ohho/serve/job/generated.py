"""Generated from ohho/serve/job/schema.json. Do not edit.

Schema: ohho.job/v1
Regenerate: python scripts/generate_job_types.py
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

JobSchemaVersion = Literal["ohho.job/v1"]
JobStep = Literal["optimize", "parity", "bench"]
JobPrecision = Literal["fp32", "fp16", "bf16", "int8"]
JobBackend = Literal["onnx", "torchscript", "cpu"]

SCHEMA_ID = "https://ohho-robotics.com/schemas/ohho.job/v1.json"
SCHEMA_VERSION: JobSchemaVersion = "ohho.job/v1"


@dataclass(frozen=True)
class JobModel:
    """Model version reference."""

    version: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobModel:
        """Build this object from a validated mapping."""
        version = data["version"]
        return cls(
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        out["version"] = self.version
        return out


@dataclass(frozen=True)
class JobParity:
    """Parity thresholds. Omitted fields stay unset."""

    max_abs_error: int | float | None = None
    max_rel_error: int | float | None = None
    min_cosine: int | float | None = None
    samples: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobParity:
        """Build this object from a validated mapping."""
        if "max_abs_error" in data:
            max_abs_error = data["max_abs_error"]
        else:
            max_abs_error = None
        if "max_rel_error" in data:
            max_rel_error = data["max_rel_error"]
        else:
            max_rel_error = None
        if "min_cosine" in data:
            min_cosine = data["min_cosine"]
        else:
            min_cosine = None
        if "samples" in data:
            samples = data["samples"]
        else:
            samples = None
        return cls(
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            min_cosine=min_cosine,
            samples=samples,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        if self.max_abs_error is not None:
            out["max_abs_error"] = self.max_abs_error
        if self.max_rel_error is not None:
            out["max_rel_error"] = self.max_rel_error
        if self.min_cosine is not None:
            out["min_cosine"] = self.min_cosine
        if self.samples is not None:
            out["samples"] = self.samples
        return out


@dataclass(frozen=True)
class JobBench:
    """Bench settings. Omitted fields stay unset."""

    iterations: int | None = None
    warmup: int | None = None
    batch_size: int | None = None
    timeout_s: int | float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobBench:
        """Build this object from a validated mapping."""
        if "iterations" in data:
            iterations = data["iterations"]
        else:
            iterations = None
        if "warmup" in data:
            warmup = data["warmup"]
        else:
            warmup = None
        if "batch_size" in data:
            batch_size = data["batch_size"]
        else:
            batch_size = None
        if "timeout_s" in data:
            timeout_s = data["timeout_s"]
        else:
            timeout_s = None
        return cls(
            iterations=iterations,
            warmup=warmup,
            batch_size=batch_size,
            timeout_s=timeout_s,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        if self.iterations is not None:
            out["iterations"] = self.iterations
        if self.warmup is not None:
            out["warmup"] = self.warmup
        if self.batch_size is not None:
            out["batch_size"] = self.batch_size
        if self.timeout_s is not None:
            out["timeout_s"] = self.timeout_s
        return out


@dataclass(frozen=True)
class JobBudget:
    """Budget caps. Omitted fields stay unset."""

    max_seconds: int | float | None = None
    max_usd: int | float | None = None
    max_gpu_seconds: int | float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobBudget:
        """Build this object from a validated mapping."""
        if "max_seconds" in data:
            max_seconds = data["max_seconds"]
        else:
            max_seconds = None
        if "max_usd" in data:
            max_usd = data["max_usd"]
        else:
            max_usd = None
        if "max_gpu_seconds" in data:
            max_gpu_seconds = data["max_gpu_seconds"]
        else:
            max_gpu_seconds = None
        return cls(
            max_seconds=max_seconds,
            max_usd=max_usd,
            max_gpu_seconds=max_gpu_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        if self.max_seconds is not None:
            out["max_seconds"] = self.max_seconds
        if self.max_usd is not None:
            out["max_usd"] = self.max_usd
        if self.max_gpu_seconds is not None:
            out["max_gpu_seconds"] = self.max_gpu_seconds
        return out


@dataclass(frozen=True)
class JobWandbExport:
    """Weights and Biases export."""

    enabled: bool
    project: str | None = None
    entity: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobWandbExport:
        """Build this object from a validated mapping."""
        enabled = data["enabled"]
        if "project" in data:
            project = data["project"]
        else:
            project = None
        if "entity" in data:
            entity = data["entity"]
        else:
            entity = None
        return cls(
            enabled=enabled,
            project=project,
            entity=entity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        out["enabled"] = self.enabled
        if self.project is not None:
            out["project"] = self.project
        if self.entity is not None:
            out["entity"] = self.entity
        return out


@dataclass(frozen=True)
class JobPrometheusExport:
    """Prometheus export."""

    enabled: bool
    port: int | None = None
    job_name: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobPrometheusExport:
        """Build this object from a validated mapping."""
        enabled = data["enabled"]
        if "port" in data:
            port = data["port"]
        else:
            port = None
        if "job_name" in data:
            job_name = data["job_name"]
        else:
            job_name = None
        return cls(
            enabled=enabled,
            port=port,
            job_name=job_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        out["enabled"] = self.enabled
        if self.port is not None:
            out["port"] = self.port
        if self.job_name is not None:
            out["job_name"] = self.job_name
        return out


@dataclass(frozen=True)
class JobOutputs:
    """Export targets. Omitted exports stay unset."""

    wandb: JobWandbExport | None = None
    prometheus: JobPrometheusExport | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobOutputs:
        """Build this object from a validated mapping."""
        if "wandb" in data:
            wandb = JobWandbExport.from_dict(data["wandb"])
        else:
            wandb = None
        if "prometheus" in data:
            prometheus = JobPrometheusExport.from_dict(data["prometheus"])
        else:
            prometheus = None
        return cls(
            wandb=wandb,
            prometheus=prometheus,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        if self.wandb is not None:
            out["wandb"] = self.wandb.to_dict()
        if self.prometheus is not None:
            out["prometheus"] = self.prometheus.to_dict()
        return out


@dataclass(frozen=True)
class Job:
    """Shared optimize, parity and bench request (ohho.job/v1)."""

    schema_version: JobSchemaVersion
    model: JobModel
    targets: tuple[str, ...]
    steps: tuple[JobStep, ...]
    precision: JobPrecision | None = None
    backend: JobBackend | None = None
    parity: JobParity | None = None
    bench: JobBench | None = None
    budget: JobBudget | None = None
    outputs: JobOutputs | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Job:
        """Build this object from a validated mapping."""
        schema_version = data["schema_version"]
        model = JobModel.from_dict(data["model"])
        targets = tuple(data["targets"])
        steps = tuple(data["steps"])
        if "precision" in data:
            precision = data["precision"]
        else:
            precision = None
        if "backend" in data:
            backend = data["backend"]
        else:
            backend = None
        if "parity" in data:
            parity = JobParity.from_dict(data["parity"])
        else:
            parity = None
        if "bench" in data:
            bench = JobBench.from_dict(data["bench"])
        else:
            bench = None
        if "budget" in data:
            budget = JobBudget.from_dict(data["budget"])
        else:
            budget = None
        if "outputs" in data:
            outputs = JobOutputs.from_dict(data["outputs"])
        else:
            outputs = None
        return cls(
            schema_version=schema_version,
            model=model,
            targets=targets,
            steps=steps,
            precision=precision,
            backend=backend,
            parity=parity,
            bench=bench,
            budget=budget,
            outputs=outputs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-compatible dict."""
        out: dict[str, Any] = {}
        out["schema_version"] = self.schema_version
        out["model"] = self.model.to_dict()
        out["targets"] = list(self.targets)
        out["steps"] = list(self.steps)
        if self.precision is not None:
            out["precision"] = self.precision
        if self.backend is not None:
            out["backend"] = self.backend
        if self.parity is not None:
            out["parity"] = self.parity.to_dict()
        if self.bench is not None:
            out["bench"] = self.bench.to_dict()
        if self.budget is not None:
            out["budget"] = self.budget.to_dict()
        if self.outputs is not None:
            out["outputs"] = self.outputs.to_dict()
        return out
