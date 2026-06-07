"""Save-handle feature (oMLX side of llamactl L4 KV slot-save).

The save-handle path lets the llamactl proxy persist an oMLX KV slot after a
cold chat by correlating a proxy-supplied save handle with the prompt token-ids
the engine consumed. The whole path is DARK by default; set
``OMLX_SAVE_HANDLE_ENABLED=1`` to turn it on. Mirrors the proxy-side
``LLAMACTL_OMLX_KV_SAVE_ENABLED`` gate so each side rolls back independently.
"""

import os
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

_TRUTHY = {"1", "true", "yes", "on"}


def save_handle_enabled() -> bool:
    """True when the save-handle feature is enabled via env (default off)."""
    return os.environ.get("OMLX_SAVE_HANDLE_ENABLED", "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


class SaveHandleTable:
    """Bounded, TTL'd, LRU map of ``save_handle -> (prompt_token_ids, model_id)``.

    Recorded after a (non-streaming) chat that carried ``x_omlx_save_handle`` and
    read by a subsequent ``/slots/0?action=save`` keyed on the same handle.

    Lookups are NON-CONSUMING: a retried or delayed save must still resolve
    within the TTL. Token-id lists are small, so the table is bounded by entry
    count + TTL only (no byte accounting). Thread-safe (the save path runs in a
    threadpool while chats record from the event loop).
    """

    _DEFAULT_MAX_ENTRIES = 64
    _DEFAULT_TTL_SECS = 300

    def __init__(
        self,
        *,
        max_entries: Optional[int] = None,
        ttl_secs: Optional[int] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self._max_entries = (
            int(max_entries)
            if max_entries is not None
            else _env_int("OMLX_SAVE_HANDLE_MAX_ENTRIES", self._DEFAULT_MAX_ENTRIES)
        )
        self._ttl = (
            int(ttl_secs)
            if ttl_secs is not None
            else _env_int("OMLX_SAVE_HANDLE_TTL_SECS", self._DEFAULT_TTL_SECS)
        )
        # handle -> (token_ids, model_id, inserted_at)
        self._entries: "OrderedDict[str, Tuple[List[int], Optional[str], float]]" = OrderedDict()
        self._time_fn = time_fn or time.monotonic
        self._guard = threading.RLock()

    def put(self, handle: str, token_ids: List[int], model_id: Optional[str]) -> None:
        if not handle or not token_ids:
            return
        now = self._time_fn()
        with self._guard:
            self._sweep_expired(now)
            self._entries[handle] = (list(token_ids), model_id, now)
            self._entries.move_to_end(handle)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)  # evict LRU

    def get(self, handle: str, model_id: Optional[str]) -> Optional[List[int]]:
        """Return recorded ids for ``handle`` (NON-CONSUMING), or None.

        Rejects when the entry is missing, expired, or recorded under a different
        model than ``model_id`` (never serialize a cross-model cache).
        """
        if not handle:
            return None
        now = self._time_fn()
        with self._guard:
            self._sweep_expired(now)
            entry = self._entries.get(handle)
            if entry is None:
                return None
            token_ids, stored_model, _ = entry
            if model_id is not None and stored_model is not None and stored_model != model_id:
                return None
            self._entries.move_to_end(handle)  # LRU touch; entry retained
            return list(token_ids)

    def _sweep_expired(self, now: float) -> None:
        if self._ttl <= 0:
            return
        stale = [h for h, (_, _, ts) in self._entries.items() if now - ts > self._ttl]
        for h in stale:
            self._entries.pop(h, None)

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)


_table_singleton: Optional[SaveHandleTable] = None
_table_lock = threading.Lock()


def get_save_handle_table() -> SaveHandleTable:
    """Process-wide save-handle table (lazily created)."""
    global _table_singleton
    if _table_singleton is None:
        with _table_lock:
            if _table_singleton is None:
                _table_singleton = SaveHandleTable()
    return _table_singleton
