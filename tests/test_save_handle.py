"""Phase 1 unit tests for the save-handle carrier: gate + prompt_token_ids fields."""

import pytest

from omlx.save_handle import save_handle_enabled
from omlx.engine.base import GenerationOutput
from omlx.request import RequestOutput


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_SAVE_HANDLE_ENABLED", raising=False)
    assert save_handle_enabled() is False


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("", False), ("nope", False)],
)
def test_gate_env(monkeypatch, val, expected):
    monkeypatch.setenv("OMLX_SAVE_HANDLE_ENABLED", val)
    assert save_handle_enabled() is expected


def test_generation_output_defaults_empty():
    # Back-compat: existing call sites that omit the field get [].
    assert GenerationOutput(text="x").prompt_token_ids == []


def test_generation_output_carries_ids():
    assert GenerationOutput(text="x", prompt_token_ids=[1, 2, 3]).prompt_token_ids == [1, 2, 3]


def test_request_output_defaults_none():
    # None (not []) so a save can distinguish "feature off" from "empty prompt".
    assert RequestOutput(request_id="r").prompt_token_ids is None


def test_request_output_carries_ids():
    assert RequestOutput(request_id="r", prompt_token_ids=[4, 5]).prompt_token_ids == [4, 5]


# --- Phase 2: x_omlx_save_handle request field ---

from omlx.api.openai_models import ChatCompletionRequest

_MSGS = [{"role": "user", "content": "hi"}]


def test_request_save_handle_absent_is_none():
    r = ChatCompletionRequest(model="m", messages=_MSGS)
    assert r.x_omlx_save_handle is None


def test_request_save_handle_parses_and_is_disjoint_from_request_handle():
    r = ChatCompletionRequest(model="m", messages=_MSGS, x_omlx_save_handle="abc123")
    assert r.x_omlx_save_handle == "abc123"
    # Must not bleed into the restore-apply handle (which would 409 a cold chat).
    assert r.x_omlx_request_handle is None
    assert r.x_omlx_restore_epoch is None


# --- Phase 3: SaveHandleTable ---

from omlx.save_handle import SaveHandleTable


def test_table_record_then_get():
    t = SaveHandleTable(time_fn=lambda: 100.0)
    t.put("sha1", [1, 2, 3], "modelA")
    assert t.get("sha1", "modelA") == [1, 2, 3]


def test_table_missing_handle_is_none():
    assert SaveHandleTable(time_fn=lambda: 0.0).get("nope", "m") is None


def test_table_lookup_is_non_consuming():
    t = SaveHandleTable(time_fn=lambda: 0.0)
    t.put("h", [9], "m")
    assert t.get("h", "m") == [9]
    assert t.get("h", "m") == [9]  # retried/delayed save still resolves


def test_table_ttl_expiry():
    clock = {"t": 0.0}
    t = SaveHandleTable(ttl_secs=300, time_fn=lambda: clock["t"])
    t.put("h", [1], "m")
    clock["t"] = 301.0
    assert t.get("h", "m") is None


def test_table_lru_cap_evicts_oldest():
    t = SaveHandleTable(max_entries=2, ttl_secs=0, time_fn=lambda: 0.0)
    t.put("a", [1], "m")
    t.put("b", [2], "m")
    t.put("c", [3], "m")  # over cap -> evict oldest (a)
    assert t.get("a", "m") is None
    assert t.get("b", "m") == [2]
    assert t.get("c", "m") == [3]


def test_table_model_mismatch_rejected():
    t = SaveHandleTable(time_fn=lambda: 0.0)
    t.put("h", [1], "modelA")
    assert t.get("h", "modelB") is None  # never serialize a cross-model cache
    assert t.get("h", "modelA") == [1]


def test_table_put_ignores_empty():
    t = SaveHandleTable(time_fn=lambda: 0.0)
    t.put("", [1], "m")
    t.put("h", [], "m")
    assert len(t) == 0
