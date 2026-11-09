"""OhhO wire/data protocol version shared with @ohho/schemas and ohho-sdk.

Independent from the ``ohho-os`` package semver (``ohho.__version__``): bump
``PROTOCOL_VERSION`` only when JSON schemas under ``schemas/`` change in a
breaking or contract-visible way.
"""

from __future__ import annotations

PROTOCOL_VERSION = "1.0.0"
