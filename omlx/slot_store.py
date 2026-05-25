# SPDX-License-Identifier: Apache-2.0
"""Slot persistence primitives for the `/slots` API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import struct
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable


class SlotState(Enum):
    IDLE = "IDLE"
    SAVING = "SAVING"
    RESTORING = "RESTORING"
    GENERATING = "GENERATING"


class InvalidFilename(ValueError):
    """Raised when a slot filename violates path-safety rules."""


class SlotBusy(RuntimeError):
    """Raised when a slot cannot transition into a requested state."""

    def __init__(self, state: SlotState) -> None:
        super().__init__(f"slot busy: {state.value}")
        self.state = state


class SlotManifestInvalid(ValueError):
    """Raised when a slot manifest is malformed or incompatible."""


class SlotGuardMismatch(RuntimeError):
    """Raised when restore guards fail against current runtime."""

    def __init__(self, field: str, expected: str, observed: str) -> None:
        super().__init__(
            f"restore guard mismatch for {field}: expected={expected} observed={observed}"
        )
        self.field = field
        self.expected = expected
        self.observed = observed


class SlotApplyEpochMismatch(RuntimeError):
    """Raised when a one-shot bind epoch token is missing or mismatched."""

    def __init__(
        self,
        model_id: str,
        request_handle: str,
        expected_epoch: str,
        observed_epoch: str | None,
    ) -> None:
        message = (
            "slot apply epoch mismatch: "
            f"model={model_id} request_handle={request_handle} "
            f"expected={expected_epoch} observed={observed_epoch}"
        )
        super().__init__(message)
        self.model_id = model_id
        self.request_handle = request_handle
        self.expected_epoch = expected_epoch
        self.observed_epoch = observed_epoch


class SlotApplyHandleNotFound(RuntimeError):
    """Raised when no one-shot bind exists for (model_id, request_handle)."""

    def __init__(self, model_id: str, request_handle: str) -> None:
        super().__init__(
            f"slot handle not found: model={model_id} request_handle={request_handle}"
        )
        self.model_id = model_id
        self.request_handle = request_handle


class SlotApplyGuardMismatch(RuntimeError):
    """Raised when one-shot admission guard checks fail."""

    def __init__(
        self,
        field: str,
        expected: str,
        observed: str,
        model_id: str,
        request_handle: str,
    ) -> None:
        super().__init__(
            f"slot apply guard mismatch for {field}: expected={expected} observed={observed}"
        )
        self.field = field
        self.expected = expected
        self.observed = observed
        self.model_id = model_id
        self.request_handle = request_handle


@dataclass
class SlotManifest:
    slot_format_version: int
    model_fingerprint: str
    model_id: str
    ctx_size: int
    n_tokens: int
    tensors: list[dict[str, Any]]
    cache_class: str
    producer: dict[str, str]
    prompt_prefix_sha256: str | None = None


@dataclass
class OneShotBind:
    model_id: str
    request_handle: str
    payload_bytes: bytes
    manifest: SlotManifest
    restore_epoch: str


class OneShotBindTable:
    _DEFAULT_MAX_ENTRIES = 1024
    _DEFAULT_MAX_TOTAL_BYTES = 1 << 30  # 1 GiB
    _DEFAULT_TTL_SECS = 600

    def __init__(
        self,
        *,
        max_entries: int | None = None,
        max_total_bytes: int | None = None,
        entry_ttl_secs: int | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._max_entries = (
            int(max_entries)
            if max_entries is not None
            else self._env_int("OMLX_ONE_SHOT_MAX_ENTRIES", self._DEFAULT_MAX_ENTRIES)
        )
        self._max_total_bytes = (
            int(max_total_bytes)
            if max_total_bytes is not None
            else self._env_int("OMLX_ONE_SHOT_MAX_BYTES", self._DEFAULT_MAX_TOTAL_BYTES)
        )
        self._entry_ttl_secs = (
            int(entry_ttl_secs)
            if entry_ttl_secs is not None
            else self._env_int("OMLX_ONE_SHOT_TTL_SECS", self._DEFAULT_TTL_SECS)
        )

        self._entries: OrderedDict[tuple[str, str], OneShotBind] = OrderedDict()
        self._expires_at: dict[tuple[str, str], float] = {}
        self._total_bytes = 0
        self._time_fn = time_fn or time.monotonic
        self._guard = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            return default
        return parsed if parsed >= 0 else default

    @staticmethod
    def _payload_size(bind: OneShotBind) -> int:
        return len(bind.payload_bytes)

    def _emit_eviction(self, bind: OneShotBind, reason: str) -> None:
        payload = {
            "handle": bind.request_handle,
            "reason": reason,
            "remaining_entries": len(self._entries),
            "remaining_bytes": self._total_bytes,
        }
        self._logger.info(
            "one_shot_bind_table_evicted %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _pop_key(self, key: tuple[str, str], *, reason: str | None = None) -> None:
        bind = self._entries.pop(key, None)
        self._expires_at.pop(key, None)
        if bind is None:
            return
        self._total_bytes = max(0, self._total_bytes - self._payload_size(bind))
        if reason is not None:
            self._emit_eviction(bind, reason)

    def _evict_oldest(self, *, reason: str) -> bool:
        if not self._entries:
            return False
        oldest_key = next(iter(self._entries))
        self._pop_key(oldest_key, reason=reason)
        return True

    def _sweep_expired_front_locked(self, now: float) -> None:
        if self._entry_ttl_secs <= 0:
            while self._evict_oldest(reason="ttl"):
                pass
            return
        while self._entries:
            oldest_key = next(iter(self._entries))
            expires_at = self._expires_at.get(oldest_key)
            if expires_at is None or expires_at > now:
                break
            self._pop_key(oldest_key, reason="ttl")

    async def bind(self, bind: OneShotBind) -> None:
        key = (bind.model_id, bind.request_handle)
        async with self._guard:
            now = self._time_fn()
            self._sweep_expired_front_locked(now)

            if key in self._entries:
                self._pop_key(key)

            self._entries[key] = bind
            self._expires_at[key] = now + float(self._entry_ttl_secs)
            self._total_bytes += self._payload_size(bind)

            while len(self._entries) > self._max_entries:
                if not self._evict_oldest(reason="lru"):
                    break

            while self._total_bytes > self._max_total_bytes:
                if not self._evict_oldest(reason="max_bytes"):
                    break

    async def put(self, bind: OneShotBind) -> None:
        await self.bind(bind)

    async def consume(
        self, model_id: str, request_handle: str, restore_epoch: str
    ) -> OneShotBind | None:
        """Atomically consume only when the entry exists and epoch matches."""
        key = (model_id, request_handle)
        async with self._guard:
            bind = self._entries.get(key)
            if bind is None:
                return None
            if bind.restore_epoch != restore_epoch:
                return None
            self._expires_at.pop(key, None)
            removed = self._entries.pop(key, None)
            if removed is not None:
                self._total_bytes = max(0, self._total_bytes - self._payload_size(removed))
            return removed

    async def consume_any(self, model_id: str, request_handle: str) -> OneShotBind | None:
        """Deprecated: consumes by key without epoch validation."""
        key = (model_id, request_handle)
        async with self._guard:
            self._expires_at.pop(key, None)
            removed = self._entries.pop(key, None)
            if removed is not None:
                self._total_bytes = max(0, self._total_bytes - self._payload_size(removed))
            return removed

    async def peek_any(self, model_id: str, request_handle: str) -> OneShotBind | None:
        """Return entry by key without removing it."""
        key = (model_id, request_handle)
        async with self._guard:
            return self._entries.get(key)

    async def drain(self) -> int:
        """Clear all entries; log each as slot_apply_drain_on_disable. Returns count."""
        async with self._guard:
            entries = list(self._entries.values())
            self._entries.clear()
            self._expires_at.clear()
            self._total_bytes = 0
        for bind in entries:
            self._logger.info(
                "[slot_apply_drain_on_disable] model_id=%s request_handle=%s restore_epoch=%s",
                bind.model_id,
                bind.request_handle,
                bind.restore_epoch,
            )
        return len(entries)


def _cheap_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a content hash with bounded cost for very large files."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    if size <= 8 * 1024 * 1024:
        with path.open("rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                digest.update(data)
    else:
        with path.open("rb") as f:
            head = f.read(chunk_size)
            digest.update(head)
            if size > chunk_size:
                f.seek(max(size - chunk_size, 0))
                tail = f.read(chunk_size)
                digest.update(tail)
        digest.update(str(size).encode("utf-8"))
    return digest.hexdigest()


def compute_model_fingerprint(model_path: Path) -> str:
    """Build a stable SHA-256 fingerprint for model artifacts under a path."""
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"model path not found: {resolved}")

    artifacts: list[dict[str, Any]] = []
    for path in sorted(p for p in resolved.rglob("*") if p.is_file()):
        rel = path.relative_to(resolved).as_posix()
        # Favor model artifacts, skip obvious non-model noise.
        if rel.startswith("."):
            continue
        if rel.endswith((".safetensors", ".json", ".txt", ".model", ".bin")):
            artifacts.append(
                {
                    "name": rel,
                    "size": path.stat().st_size,
                    "content_hash": _cheap_file_hash(path),
                }
            )

    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _parse_manifest_dict(data: dict[str, Any]) -> SlotManifest:
    required_fields = (
        "slot_format_version",
        "model_fingerprint",
        "model_id",
        "ctx_size",
        "n_tokens",
        "tensors",
        "cache_class",
        "producer",
    )
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise SlotManifestInvalid(
            f"missing required manifest fields: {', '.join(sorted(missing))}"
        )

    version = int(data["slot_format_version"])
    if version not in (1, 2):
        raise SlotManifestInvalid(f"unsupported slot_format_version: {version}")

    prompt_prefix_sha256: str | None = None
    if version >= 2:
        raw_prefix_sha = data.get("prompt_prefix_sha256")
        if not isinstance(raw_prefix_sha, str):
            raise SlotManifestInvalid(
                "missing required manifest fields: prompt_prefix_sha256"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", raw_prefix_sha):
            raise SlotManifestInvalid("invalid prompt_prefix_sha256 format")
        prompt_prefix_sha256 = raw_prefix_sha

    try:
        return SlotManifest(
            slot_format_version=int(data["slot_format_version"]),
            model_fingerprint=str(data["model_fingerprint"]),
            model_id=str(data["model_id"]),
            ctx_size=int(data["ctx_size"]),
            n_tokens=int(data["n_tokens"]),
            tensors=list(data["tensors"]),
            cache_class=str(data["cache_class"]),
            producer=dict(data["producer"]),
            prompt_prefix_sha256=prompt_prefix_sha256,
        )
    except Exception as exc:  # pragma: no cover - defensive conversion errors
        raise SlotManifestInvalid(f"invalid manifest fields: {exc}") from exc


def hash_prompt_token_prefix(prompt_token_ids: list[int]) -> str:
    """Stable SHA-256 for token-prefix identity using little-endian int32 packing."""
    try:
        prefix_bytes = b"".join(
            struct.pack("<i", int(token_id)) for token_id in prompt_token_ids
        )
    except Exception as exc:
        raise ValueError(f"invalid prompt token sequence: {exc}") from exc
    return hashlib.sha256(prefix_bytes).hexdigest()


def check_restore_guards(
    manifest: SlotManifest,
    current_fingerprint: str,
    current_ctx_size: int,
) -> None:
    """Validate v1 restore guards (fingerprint + ctx size only)."""
    if manifest.model_fingerprint != current_fingerprint:
        raise SlotGuardMismatch(
            field="model_fingerprint",
            expected=manifest.model_fingerprint,
            observed=current_fingerprint,
        )
    if int(manifest.ctx_size) != int(current_ctx_size):
        raise SlotGuardMismatch(
            field="ctx_size",
            expected=str(manifest.ctx_size),
            observed=str(current_ctx_size),
        )


class SlotStore:
    _VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

    def __init__(self, slot_save_path: Path) -> None:
        self._slot_save_path = Path(slot_save_path).expanduser().resolve()
        self._slot_save_path.mkdir(parents=True, exist_ok=True)
        self._states: dict[int, SlotState] = {}
        self._state_guard = asyncio.Lock()

    def validate_filename(self, raw: str) -> str:
        if not isinstance(raw, str):
            raise InvalidFilename("filename must be a string")
        filename = raw.strip()
        if not filename:
            raise InvalidFilename("filename cannot be empty")
        if not self._VALID_NAME_RE.fullmatch(filename):
            raise InvalidFilename("filename contains invalid characters")
        path = Path(filename)
        if path.is_absolute():
            raise InvalidFilename("absolute path is not allowed")
        if any(part == ".." for part in path.parts):
            raise InvalidFilename("path traversal is not allowed")
        if len(path.parts) != 1:
            raise InvalidFilename("filename must be a basename")
        return path.name

    def get_state(self, slot_id: int) -> SlotState:
        return self._states.get(slot_id, SlotState.IDLE)

    @asynccontextmanager
    async def acquire_for_save(self, slot_id: int) -> AsyncIterator[None]:
        async with self._state_guard:
            state = self._states.get(slot_id, SlotState.IDLE)
            if state is not SlotState.IDLE:
                raise SlotBusy(state)
            self._states[slot_id] = SlotState.SAVING
        try:
            yield
        finally:
            async with self._state_guard:
                self._states[slot_id] = SlotState.IDLE

    @asynccontextmanager
    async def acquire_for_restore(self, slot_id: int) -> AsyncIterator[None]:
        async with self._state_guard:
            state = self._states.get(slot_id, SlotState.IDLE)
            if state is not SlotState.IDLE:
                raise SlotBusy(state)
            self._states[slot_id] = SlotState.RESTORING
        try:
            yield
        finally:
            async with self._state_guard:
                self._states[slot_id] = SlotState.IDLE

    async def write_atomic(
        self,
        slot_id: int,
        filename: str,
        payload: bytes,
        manifest: SlotManifest,
    ) -> int:
        del slot_id  # slot id is part of the caller contract, not the file naming.
        safe_name = self.validate_filename(filename)
        return await asyncio.to_thread(
            self._write_atomic_sync,
            safe_name,
            payload,
            manifest,
        )

    def _write_atomic_sync(
        self,
        filename: str,
        payload: bytes,
        manifest: SlotManifest,
    ) -> int:
        payload_final = self._slot_save_path / filename
        manifest_final = self._slot_save_path / f"{filename}.manifest.json"

        token = uuid.uuid4().hex
        payload_tmp = self._slot_save_path / f"{filename}.tmp.{token}"
        manifest_tmp = self._slot_save_path / f"{filename}.manifest.json.tmp.{token}"

        manifest_bytes = json.dumps(
            asdict(manifest), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        payload_published = False
        try:
            with payload_tmp.open("wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            with manifest_tmp.open("wb") as f:
                f.write(manifest_bytes)
                f.flush()
                os.fsync(f.fileno())

            os.replace(str(payload_tmp), str(payload_final))
            payload_published = True
            os.replace(str(manifest_tmp), str(manifest_final))
            return int(manifest.n_tokens)
        except Exception:
            for tmp in (payload_tmp, manifest_tmp):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
            if payload_published:
                try:
                    if payload_final.exists():
                        payload_final.unlink()
                except OSError:
                    pass
            raise

    async def read_with_manifest(
        self, slot_id: int, filename: str
    ) -> tuple[bytes, SlotManifest]:
        del slot_id  # slot id is part of the caller contract, not file naming.
        safe_name = self.validate_filename(filename)
        return await asyncio.to_thread(self._read_with_manifest_sync, safe_name)

    def _read_with_manifest_sync(self, filename: str) -> tuple[bytes, SlotManifest]:
        payload_path = self._slot_save_path / filename
        manifest_path = self._slot_save_path / f"{filename}.manifest.json"
        if not payload_path.exists():
            raise FileNotFoundError(f"slot payload not found: {filename}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"slot manifest not found: {filename}.manifest.json")

        payload = payload_path.read_bytes()
        try:
            manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SlotManifestInvalid(f"failed to parse manifest json: {exc}") from exc
        if not isinstance(manifest_obj, dict):
            raise SlotManifestInvalid("manifest root must be a JSON object")

        manifest = _parse_manifest_dict(manifest_obj)
        return payload, manifest
