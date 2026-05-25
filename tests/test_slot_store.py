# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from omlx.slot_store import (
    InvalidFilename,
    SlotBusy,
    SlotManifest,
    SlotState,
    SlotStore,
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
