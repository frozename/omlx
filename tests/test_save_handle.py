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
