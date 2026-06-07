"""Save-handle feature gate (oMLX side of llamactl L4 KV slot-save).

The save-handle path lets the llamactl proxy persist an oMLX KV slot after a
cold chat by correlating a proxy-supplied save handle with the prompt token-ids
the engine consumed. The whole path is DARK by default; set
``OMLX_SAVE_HANDLE_ENABLED=1`` to turn it on. Mirrors the proxy-side
``LLAMACTL_OMLX_KV_SAVE_ENABLED`` gate so each side rolls back independently.
"""

import os

_TRUTHY = {"1", "true", "yes", "on"}


def save_handle_enabled() -> bool:
    """True when the save-handle feature is enabled via env (default off)."""
    return os.environ.get("OMLX_SAVE_HANDLE_ENABLED", "").strip().lower() in _TRUTHY
