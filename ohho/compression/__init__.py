"""Compression passes for ``ohho train``.

Quantisation-aware training lives in :mod:`ohho.compression.qat`. Other
compression passes should take their own modules so each one can add a
recipe block and manifest field without sharing implementation.
"""

from __future__ import annotations
