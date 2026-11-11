"""ServingBackend: launch an OpenAI-compatible LLM server and stop it."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .cmdline import (
    as_config,
    env_overrides,
    quote_command,
    require_model,
    semver_from_text,
    version_command,
    working_directory,
)
from .errors import ServingBackendError
from .tree import ProcessTree, http_ok


@dataclass(frozen=True)
class LaunchPlan:
    """Argv and URLs for one engine launch. No process is started."""

    commands: tuple[tuple[str, ...], ...]
    endpoint_url: str
    health_url: str
    version_argv: tuple[str, ...]
    precision: str


class ServingBackend:
    """Adapter that launches an external OpenAI-compatible server.

    Subclasses build the engine command. The engine itself is an optional
    program on ``PATH`` or a Python module. It is not a dependency of ohho-os.

    ``launch`` returns when ``health`` is true. ``teardown`` stops the
    process tree, including children the engine started. A failed launch
    does the same before raising :class:`ServingBackendError`.
    """

    name: str = ""
    default_executable: str = ""

    def __init__(self) -> None:
        self._plan: LaunchPlan | None = None
        self._tree = ProcessTree()

    def plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any] | None = None,
    ) -> LaunchPlan:
        """Build the launch command and URLs without starting a process."""
        cfg = as_config(config)
        built = self.build_plan(require_model(model), str(precision).strip(), cfg)
        if not built.commands or any(not command for command in built.commands):
            raise ServingBackendError(f"{self.name} produced an empty command")
        return built

    def build_plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        """Return the engine-specific command. Implemented by each adapter."""
        raise NotImplementedError

    def launch(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Start the engine and wait until its health URL responds.

        On timeout or an early exit the process tree is killed and
        :class:`ServingBackendError` is raised.
        """
        cfg = as_config(config)
        timeout = _timeout(cfg)
        try:
            plan = self.plan(model, precision, cfg)
            self._plan = plan
            self._tree.start(
                [list(command) for command in plan.commands],
                env=env_overrides(cfg),
                cwd=working_directory(cfg),
            )
            self._wait_until_healthy(timeout)
        except Exception:
            self.teardown()
            raise

    def health(self) -> bool:
        """True when the launched server answers its health URL."""
        if self._plan is None or not self._tree.running:
            return False
        return http_ok(self._plan.health_url)

    def endpoint_url(self) -> str:
        """OpenAI-compatible base URL, including the ``/v1`` prefix."""
        if self._plan is None:
            raise ServingBackendError(f"{self.name} has not been launched")
        return self._plan.endpoint_url

    def teardown(self) -> None:
        """Stop the engine and every process it started. Idempotent."""
        self._tree.kill()

    def version(self, config: Mapping[str, Any] | None = None) -> str:
        """Return the engine version, or raise when it is not installed."""
        cfg = as_config(config)
        argv = self.version_command(cfg)
        timeout = float(cfg.get("version_timeout_s", 15))
        if timeout <= 0:
            raise ServingBackendError("version_timeout_s must be positive")
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ServingBackendError(self._missing(argv[0])) from exc
        except subprocess.TimeoutExpired as exc:
            raise ServingBackendError(
                f"{self.name} version command timed out: {argv[0]}"
            ) from exc
        text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        found = self.parse_version_text(text)
        if found:
            return found
        tail = text.strip()[-800:]
        if completed.returncode != 0 and (
            "No module named" in text or "ModuleNotFoundError" in text
        ):
            raise ServingBackendError(self._missing(argv[0], tail))
        if completed.returncode != 0:
            raise ServingBackendError(
                f"{self.name} version command failed (exit {completed.returncode}). {tail}"
            )
        raise ServingBackendError(
            f"{self.name} version output was not recognized. {tail}"
        )

    def version_command(self, config: Mapping[str, Any]) -> list[str]:
        """Argv that prints the engine version."""
        return version_command(config, self.default_version_argv())

    def default_version_argv(self) -> list[str]:
        """Version command used when config does not override the executable."""
        return [self.default_executable, "--version"]

    def parse_version(self, text: str) -> str:
        """Parse a version string from combined stdout and stderr."""
        found = self.parse_version_text(text)
        if not found:
            detail = text.strip()[-500:]
            raise ServingBackendError(
                f"could not parse a {self.name} version from: {detail}"
            )
        return found

    def parse_version_text(self, text: str) -> str | None:
        """Return a version from ``text``, or None when none is present."""
        return semver_from_text(text)

    @property
    def running(self) -> bool:
        """True while the launched process tree is still alive."""
        return self._tree.running

    def __enter__(self) -> ServingBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.teardown()

    def _wait_until_healthy(self, timeout: float) -> None:
        assert self._plan is not None
        deadline = time.monotonic() + timeout
        while True:
            code = self._tree.poll()
            if code is not None:
                time.sleep(0.05)
                raise ServingBackendError(
                    self._failure(f"exited before it was healthy (exit {code})")
                )
            if http_ok(self._plan.health_url):
                return
            if time.monotonic() >= deadline:
                raise ServingBackendError(
                    self._failure(
                        "did not become healthy within "
                        f"{timeout:.1f}s ({self._plan.health_url})"
                    )
                )
            time.sleep(0.05)

    def _failure(self, reason: str) -> str:
        assert self._plan is not None
        shown = "\n".join(quote_command(command) for command in self._plan.commands)
        tail = self._tree.output_tail(2000).strip()
        message = f"{self.name} {reason}.\ncommand:\n{shown}"
        if tail:
            message = f"{message}\n{tail}"
        return message

    def _missing(self, executable: str, detail: str = "") -> str:
        if "No module named" in detail or "ModuleNotFoundError" in detail:
            last = detail.strip().splitlines()[-1]
            return (
                f"{self.name} is not installed. Install that engine separately. "
                f"It is not a dependency of ohho-os. {last}"
            )
        message = (
            f"{self.name} is not installed: executable {executable!r} was not found. "
            "Install that engine separately. It is not a dependency of ohho-os."
        )
        if detail:
            return f"{message} {detail}"
        return message


def _timeout(config: Mapping[str, Any]) -> float:
    raw = config.get("startup_timeout_s", 30)
    try:
        timeout = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ServingBackendError("startup_timeout_s must be a number") from exc
    if timeout <= 0:
        raise ServingBackendError("startup_timeout_s must be positive")
    return timeout


def commands_text(commands: Sequence[Sequence[str]]) -> str:
    """Join commands for logs. Not passed to a shell."""
    return "\n".join(quote_command(command) for command in commands)
