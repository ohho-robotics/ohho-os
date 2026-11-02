"""Vendor locomotion modes mapped onto the generic humanoid modes.

Plain data. ``kSoccer`` is a K1/T1 mode in the SDK header and is not a key.
Anything absent, including that name, is ``UNKNOWN``, which is not standing.
"""

from __future__ import annotations

from types import MappingProxyType

VENDOR_MODE_MAP: MappingProxyType[str, str] = MappingProxyType(
    {
        "kDamping": "DAMPING",
        "kPrepare": "PREPARE",
        "kWalking": "WALK",
        "kCustom": "CUSTOM",
        "kUnknown": "UNKNOWN",
    }
)


def generic_mode(vendor_mode: object) -> str:
    """Return the generic mode for a vendor mode name.

    Unrecognised values, including ``kSoccer`` and non-strings, are
    ``UNKNOWN``.
    """
    if not isinstance(vendor_mode, str):
        return "UNKNOWN"
    return VENDOR_MODE_MAP.get(vendor_mode, "UNKNOWN")
