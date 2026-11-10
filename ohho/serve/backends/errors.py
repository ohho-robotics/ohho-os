"""Errors raised by LLM serving backends."""

from __future__ import annotations


class ServingBackendError(RuntimeError):
    """A serving backend failed to launch, respond, or stop.

    The process tree is stopped before this leaves :meth:`ServingBackend.launch`
    or :meth:`ServingBackend.teardown`.
    """
