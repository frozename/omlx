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
