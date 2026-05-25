# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from omlx.slot_store import (
    InvalidFilename,
    OneShotBind,
    OneShotBindTable,
    SlotBusy,
    SlotGuardMismatch,
    SlotManifest,
    SlotManifestInvalid,
    SlotState,
    SlotStore,
    check_restore_guards,
    compute_model_fingerprint,
)


def _build_real_prompt_cache(n_layers: int = 2, seq_len: int = 2, head_dim: int = 2):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    cache_layers = []
    for i in range(n_layers):
        cache = KVCache()
        keys = mx.full((1, 1, seq_len, head_dim), i + 1, dtype=mx.float16)
        values = mx.full((1, 1, seq_len, head_dim), i + 2, dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        cache_layers.append(cache)
    return cache_layers


def _build_slot_entry(prompt_cache, cached_tokens: int = 7):
    class _Req:
        def __init__(self):
            self.prompt_cache = prompt_cache
            self.cached_tokens = cached_tokens
            self.remaining_tokens = [1, 2]

    class _Scheduler:
        def __init__(self):
            self.requests = {"req-1": _Req()}

        def snapshot_for_admin(self):
            return {"running_by_id": {}, "waiting": []}

    class _EngineCore:
        def __init__(self):
            self.engine = type("_Inner", (), {"scheduler": _Scheduler()})()

    class _Engine:
        def __init__(self):
            self._engine = _EngineCore()

    class _Entry:
        def __init__(self):
            self.engine = _Engine()

    return _Entry()


def _manifest(n_tokens: int = 3) -> SlotManifest:
    return SlotManifest(
        slot_format_version=1,
        model_fingerprint="abc123",
        model_id="test-model",
        ctx_size=4096,
        n_tokens=n_tokens,
        tensors=[{"name": "layer_0", "dtype": "f16", "shape": [1, 2, 3]}],
        cache_class="paged_ssd",
        producer={"mlx_version": "0.0.0", "omlx_cache_format_version": "v1"},
    )


@pytest.mark.asyncio
async def test_save_atomic_publish_no_partial_on_crash(tmp_path, monkeypatch):
    import omlx.slot_store as slot_store_module

    store = SlotStore(tmp_path)
    payload_name = "slot-a.kvslot"

    original_replace = slot_store_module.os.replace

    def fail_manifest_replace(src: str, dst: str) -> None:
        if str(dst).endswith(".manifest.json"):
            raise OSError("simulated replace crash")
        original_replace(src, dst)

    monkeypatch.setattr(slot_store_module.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError):
        await store.write_atomic(
            slot_id=0,
            filename=payload_name,
            payload=b"payload-bytes",
            manifest=_manifest(),
        )

    assert not (tmp_path / payload_name).exists()
    assert not (tmp_path / f"{payload_name}.manifest.json").exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.asyncio
async def test_save_manifest_paired_and_atomic(tmp_path, monkeypatch):
    store = SlotStore(tmp_path)

    filename_ok = "slot-ok.kvslot"
    n_saved = await store.write_atomic(
        slot_id=0,
        filename=filename_ok,
        payload=b"hello",
        manifest=_manifest(n_tokens=7),
    )
    assert n_saved == 7
    assert (tmp_path / filename_ok).exists()
    assert (tmp_path / f"{filename_ok}.manifest.json").exists()

    import omlx.slot_store as slot_store_module

    original_replace = slot_store_module.os.replace

    def fail_payload_replace(src: str, dst: str) -> None:
        if str(dst).endswith("slot-fail.kvslot"):
            raise OSError("simulated payload replace failure")
        original_replace(src, dst)

    monkeypatch.setattr(slot_store_module.os, "replace", fail_payload_replace)

    with pytest.raises(OSError):
        await store.write_atomic(
            slot_id=0,
            filename="slot-fail.kvslot",
            payload=b"nope",
            manifest=_manifest(n_tokens=5),
        )

    assert not (tmp_path / "slot-fail.kvslot").exists()
    assert not (tmp_path / "slot-fail.kvslot.manifest.json").exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.asyncio
async def test_save_state_returns_to_idle_after_exception(tmp_path):
    store = SlotStore(tmp_path)
    assert store.get_state(0) == SlotState.IDLE

    with pytest.raises(RuntimeError):
        async with store.acquire_for_save(0):
            raise RuntimeError("boom")

    assert store.get_state(0) == SlotState.IDLE


@pytest.mark.asyncio
async def test_save_busy_when_already_saving(tmp_path):
    store = SlotStore(tmp_path)
    started = asyncio.Event()

    async def hold_lock() -> None:
        async with store.acquire_for_save(0):
            started.set()
            await asyncio.sleep(0.05)

    task = asyncio.create_task(hold_lock())
    await started.wait()

    with pytest.raises(SlotBusy) as exc:
        async with store.acquire_for_save(0):
            pass
    assert exc.value.state == SlotState.SAVING

    await task


@pytest.mark.asyncio
async def test_restore_busy_when_already_restoring(tmp_path):
    store = SlotStore(tmp_path)
    started = asyncio.Event()

    async def hold_lock() -> None:
        async with store.acquire_for_restore(0):
            started.set()
            await asyncio.sleep(0.05)

    task = asyncio.create_task(hold_lock())
    await started.wait()

    with pytest.raises(SlotBusy) as exc:
        async with store.acquire_for_restore(0):
            pass
    assert exc.value.state == SlotState.RESTORING

    await task


@pytest.mark.asyncio
async def test_restore_busy_when_saving(tmp_path):
    store = SlotStore(tmp_path)
    store._states[0] = SlotState.SAVING  # noqa: SLF001

    with pytest.raises(SlotBusy) as exc:
        async with store.acquire_for_restore(0):
            pass
    assert exc.value.state == SlotState.SAVING


@pytest.mark.asyncio
async def test_restore_state_returns_to_idle_after_exception(tmp_path):
    store = SlotStore(tmp_path)
    assert store.get_state(0) == SlotState.IDLE

    with pytest.raises(RuntimeError):
        async with store.acquire_for_restore(0):
            raise RuntimeError("boom")

    assert store.get_state(0) == SlotState.IDLE


@pytest.mark.asyncio
async def test_read_with_manifest_returns_payload_and_manifest_dataclass(tmp_path):
    store = SlotStore(tmp_path)
    manifest = _manifest(n_tokens=11)
    payload = b"restore-me"
    await store.write_atomic(
        slot_id=0,
        filename="slot-a.kvslot",
        payload=payload,
        manifest=manifest,
    )

    read_payload, read_manifest = await store.read_with_manifest(0, "slot-a.kvslot")
    assert read_payload == payload
    assert read_manifest == manifest


@pytest.mark.asyncio
async def test_read_with_manifest_raises_on_missing_payload(tmp_path):
    store = SlotStore(tmp_path)
    manifest_path = tmp_path / "slot-a.kvslot.manifest.json"
    manifest_path.write_text(
        '{"slot_format_version":1,"model_fingerprint":"abc","model_id":"m","ctx_size":1,"n_tokens":1,"tensors":[],"cache_class":"paged_ssd","producer":{"mlx_version":"0","omlx_cache_format_version":"v1"}}',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        await store.read_with_manifest(0, "slot-a.kvslot")


@pytest.mark.asyncio
async def test_read_with_manifest_raises_on_missing_manifest_sidecar(tmp_path):
    store = SlotStore(tmp_path)
    (tmp_path / "slot-a.kvslot").write_bytes(b"payload")
    with pytest.raises(FileNotFoundError):
        await store.read_with_manifest(0, "slot-a.kvslot")


@pytest.mark.asyncio
async def test_read_with_manifest_raises_on_unknown_major_version(tmp_path):
    store = SlotStore(tmp_path)
    (tmp_path / "slot-a.kvslot").write_bytes(b"payload")
    (tmp_path / "slot-a.kvslot.manifest.json").write_text(
        '{"slot_format_version":2,"model_fingerprint":"abc","model_id":"m","ctx_size":1,"n_tokens":1,"tensors":[],"cache_class":"paged_ssd","producer":{"mlx_version":"0","omlx_cache_format_version":"v1"}}',
        encoding="utf-8",
    )
    with pytest.raises(SlotManifestInvalid):
        await store.read_with_manifest(0, "slot-a.kvslot")


def test_check_restore_guards_fingerprint_mismatch_raises():
    manifest = _manifest()
    with pytest.raises(SlotGuardMismatch) as exc:
        check_restore_guards(
            manifest=manifest,
            current_fingerprint="different",
            current_ctx_size=manifest.ctx_size,
        )
    assert exc.value.field == "model_fingerprint"


def test_check_restore_guards_ctx_size_mismatch_raises():
    manifest = _manifest()
    with pytest.raises(SlotGuardMismatch) as exc:
        check_restore_guards(
            manifest=manifest,
            current_fingerprint=manifest.model_fingerprint,
            current_ctx_size=manifest.ctx_size + 1,
        )
    assert exc.value.field == "ctx_size"


def test_check_restore_guards_no_quant_guard_in_v1():
    manifest = _manifest()
    setattr(manifest, "quant", "q4")
    check_restore_guards(
        manifest=manifest,
        current_fingerprint=manifest.model_fingerprint,
        current_ctx_size=manifest.ctx_size,
    )


def test_validate_filename_rules(tmp_path):
    store = SlotStore(tmp_path)

    assert store.validate_filename("slot.kvslot") == "slot.kvslot"

    with pytest.raises(InvalidFilename):
        store.validate_filename("")
    with pytest.raises(InvalidFilename):
        store.validate_filename("/etc/passwd")
    with pytest.raises(InvalidFilename):
        store.validate_filename("../escape.kvslot")
    with pytest.raises(InvalidFilename):
        store.validate_filename("nested/path.kvslot")


def test_model_fingerprint_is_stable(tmp_path):
    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"name":"demo"}', encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"\x00" * 64)
    (model_dir / "readme.txt").write_text("ignore me", encoding="utf-8")

    fp1 = compute_model_fingerprint(model_dir)
    fp2 = compute_model_fingerprint(Path(model_dir))
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 64


@pytest.mark.asyncio
async def test_v2a_save_writes_real_safetensors_bytes(tmp_path):
    from omlx.server import _serialize_slot_payload

    entry = _build_slot_entry(_build_real_prompt_cache(n_layers=1), cached_tokens=5)
    payload_bytes, metadata = _serialize_slot_payload(entry, model_id="test-model")

    store = SlotStore(tmp_path)
    await store.write_atomic(
        slot_id=0,
        filename="slot-v2a.kvslot",
        payload=payload_bytes,
        manifest=_manifest(n_tokens=metadata["n_tokens"]),
    )

    on_disk = (tmp_path / "slot-v2a.kvslot").read_bytes()
    assert on_disk[:1] != b"{"
    metadata_len = int.from_bytes(on_disk[:8], "little")
    assert metadata_len > 0
    assert on_disk[8:9] == b"{"


def test_v2a_save_restore_round_trip_via_mlx_lm():
    from omlx.server import _apply_slot_restore_payload, _serialize_slot_payload

    source_cache = _build_real_prompt_cache(n_layers=2, seq_len=3, head_dim=2)
    entry = _build_slot_entry(source_cache, cached_tokens=9)
    payload_bytes, _ = _serialize_slot_payload(entry, model_id="test-model")
    restored = _apply_slot_restore_payload(
        entry,
        payload_bytes,
        _manifest(n_tokens=9),
    )
    assert restored == 9


def test_v2a_apply_n_restored_comes_from_file_metadata_when_present():
    from omlx.server import _apply_slot_restore_payload, _serialize_slot_payload

    entry = _build_slot_entry(_build_real_prompt_cache(n_layers=1), cached_tokens=42)
    payload_bytes, _ = _serialize_slot_payload(entry, model_id="test-model")
    n_restored = _apply_slot_restore_payload(entry, payload_bytes, _manifest(n_tokens=99))
    assert n_restored == 42


def test_v2a_apply_n_restored_falls_back_to_manifest_when_metadata_missing():
    from mlx_lm.models.cache import save_prompt_cache
    from omlx.server import _apply_slot_restore_payload
    import tempfile
    import os

    cache_layers = _build_real_prompt_cache(n_layers=1)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_prompt_cache(tmp_path, cache_layers, metadata={})
        payload_bytes = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    entry = _build_slot_entry(cache_layers, cached_tokens=1)
    n_restored = _apply_slot_restore_payload(entry, payload_bytes, _manifest(n_tokens=99))
    assert n_restored == 99


@pytest.mark.asyncio
async def test_one_shot_bind_put_consume_returns_entry_with_matching_epoch():
    table = OneShotBindTable()
    bind = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"abc",
        manifest=_manifest(n_tokens=7),
        restore_epoch="epoch-1",
    )
    await table.put(bind)

    consumed = await table.consume("m", "h", "epoch-1")
    assert consumed == bind


@pytest.mark.asyncio
async def test_one_shot_bind_consume_returns_none_on_no_entry():
    table = OneShotBindTable()
    consumed = await table.consume("m", "missing", "epoch-1")
    assert consumed is None


@pytest.mark.asyncio
async def test_one_shot_bind_consume_returns_none_on_epoch_mismatch_and_drops_entry():
    table = OneShotBindTable()
    bind = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"abc",
        manifest=_manifest(n_tokens=3),
        restore_epoch="expected",
    )
    await table.put(bind)

    consumed = await table.consume("m", "h", "wrong")
    assert consumed is None
    assert await table.consume_any("m", "h") is None


@pytest.mark.asyncio
async def test_one_shot_bind_consume_is_idempotent_after_consume():
    table = OneShotBindTable()
    bind = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"abc",
        manifest=_manifest(n_tokens=5),
        restore_epoch="epoch",
    )
    await table.put(bind)

    first = await table.consume("m", "h", "epoch")
    second = await table.consume("m", "h", "epoch")
    assert first == bind
    assert second is None


@pytest.mark.asyncio
async def test_one_shot_bind_put_overwrites_prior_entry_for_same_key():
    table = OneShotBindTable()
    first = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"first",
        manifest=_manifest(n_tokens=1),
        restore_epoch="epoch-a",
    )
    second = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"second",
        manifest=_manifest(n_tokens=2),
        restore_epoch="epoch-b",
    )
    await table.put(first)
    await table.put(second)

    consumed = await table.consume("m", "h", "epoch-b")
    assert consumed == second
    assert await table.consume_any("m", "h") is None


@pytest.mark.asyncio
async def test_one_shot_bind_concurrent_consume_only_one_wins():
    table = OneShotBindTable()
    bind = OneShotBind(
        model_id="m",
        request_handle="h",
        payload_bytes=b"abc",
        manifest=_manifest(n_tokens=8),
        restore_epoch="epoch-1",
    )
    await table.put(bind)

    first, second = await asyncio.gather(
        table.consume("m", "h", "epoch-1"),
        table.consume("m", "h", "epoch-1"),
    )
    winners = [result for result in (first, second) if result is not None]
    assert winners == [bind]


@pytest.mark.asyncio
async def test_drain_clears_all_entries_returns_count():
    table = OneShotBindTable()
    for i in range(3):
        await table.put(
            OneShotBind(
                model_id="m",
                request_handle=f"h-{i}",
                payload_bytes=b"abc",
                manifest=_manifest(n_tokens=i + 1),
                restore_epoch=f"epoch-{i}",
            )
        )

    drained = await table.drain()

    assert drained == 3
    assert await table.consume_any("m", "h-0") is None
    assert await table.consume_any("m", "h-1") is None
    assert await table.consume_any("m", "h-2") is None


@pytest.mark.asyncio
async def test_drain_logs_each_dropped_entry(caplog):
    table = OneShotBindTable()
    for i in range(2):
        await table.put(
            OneShotBind(
                model_id="m",
                request_handle=f"h-{i}",
                payload_bytes=b"abc",
                manifest=_manifest(n_tokens=i + 1),
                restore_epoch=f"epoch-{i}",
            )
        )

    with caplog.at_level("INFO"):
        drained = await table.drain()

    assert drained == 2
    events = [record.msg for record in caplog.records if "slot_apply_drain_on_disable" in str(record.msg)]
    assert len(events) == 2
