# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from omlx.slot_store import (
    InvalidFilename,
    SlotBusy,
    SlotGuardMismatch,
    SlotManifest,
    SlotManifestInvalid,
    SlotState,
    SlotStore,
    check_restore_guards,
    compute_model_fingerprint,
)


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
