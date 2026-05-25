# SPDX-License-Identifier: Apache-2.0
"""Slot persistence primitives for the `/slots` API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator


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


class SlotStore:
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
