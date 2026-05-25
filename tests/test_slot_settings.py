# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
import json

import pytest

from omlx.cache.model_arch import _model_uses_chunked_kv_cache
from omlx.settings import GlobalSettings


def _write_config(model_dir, model_type: str, architecture: str) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]}),
        encoding="utf-8",
    )


def test_slot_save_path_defaults_to_none(tmp_path):
    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.slot_save_path is None


def test_cli_flag_sets_slot_save_path(tmp_path):
    slot_dir = tmp_path / "slots"
    args = Namespace(slot_save_path=str(slot_dir))

    settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

    assert settings.slot_save_path == str(slot_dir)


@pytest.mark.parametrize(
    ("model_type", "architecture", "expects_invariant_error"),
    [
        ("qwen3_5_moe", "Qwen3ForCausalLM", False),
        ("llama4", "Llama4ForCausalLM", True),
    ],
)
def test_slot_save_path_mcr_guard_depends_on_cache_architecture(
    tmp_path, model_type: str, architecture: str, expects_invariant_error: bool
):
    slot_dir = tmp_path / "slots"
    model_dir = tmp_path / "model"
    _write_config(model_dir, model_type, architecture)

    args = Namespace(
        slot_save_path=str(slot_dir),
        max_concurrent_requests=4,
        model_dir=str(model_dir),
    )

    settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)
    errors = settings.validate()

    has_invariant_error = any(
        "slot_save_path requires max_concurrent_requests=1" in error for error in errors
    )
    assert has_invariant_error is expects_invariant_error


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected"),
    [
        ("llama4", "Llama4ForCausalLM", True),
        ("qwen3", "Qwen3ForCausalLM", False),
        ("gemma3", "Gemma3ForCausalLM", False),
    ],
)
def test_model_uses_chunked_kv_cache_detection(
    tmp_path, model_type: str, architecture: str, expected: bool
):
    model_dir = tmp_path / model_type
    _write_config(model_dir, model_type, architecture)
    assert _model_uses_chunked_kv_cache(str(model_dir)) is expected


def test_model_uses_chunked_kv_cache_defaults_safe_when_missing_config(tmp_path):
    assert _model_uses_chunked_kv_cache(str(tmp_path / "missing-model"))
