"""Tests for per-model ``max_completion_batch_size`` and
``prefill_step_size`` overrides on SchedulerSettings.

Mirrors the shape of ``test_per_model_concurrency.py``. These two
overrides let operators tune batch-fusion ceiling and prefill chunk
size per model on a shared GPU — smaller models often saturate at
larger batch sizes than 8B-class peers, and prefill chunk sweet-spots
differ by model.
"""

from __future__ import annotations

import argparse

import pytest


def test_settings_default_empty_dicts():
    from omlx.settings import SchedulerSettings

    s = SchedulerSettings()
    assert s.per_model_max_completion_batch_size == {}
    assert s.per_model_prefill_step_size == {}


def test_settings_round_trip_via_to_from_dict():
    from omlx.settings import SchedulerSettings

    original = SchedulerSettings(
        max_concurrent_requests=8,
        per_model_max_completion_batch_size={"granite-3b": 64, "qwen3-8b": 16},
        per_model_prefill_step_size={"granite-3b": 512, "qwen3-8b": 2048},
    )
    round_tripped = SchedulerSettings.from_dict(original.to_dict())
    assert round_tripped.per_model_max_completion_batch_size == {
        "granite-3b": 64,
        "qwen3-8b": 16,
    }
    assert round_tripped.per_model_prefill_step_size == {
        "granite-3b": 512,
        "qwen3-8b": 2048,
    }


def test_from_dict_coerces_values_to_int():
    from omlx.settings import SchedulerSettings

    raw = {
        "max_concurrent_requests": 8,
        "per_model_max_completion_batch_size": {"granite-3b": "64"},
        "per_model_prefill_step_size": {"granite-3b": "512"},
    }
    parsed = SchedulerSettings.from_dict(raw)
    assert parsed.per_model_max_completion_batch_size == {"granite-3b": 64}
    assert parsed.per_model_prefill_step_size == {"granite-3b": 512}


def test_apply_args_parses_completion_batch_size_pairs(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    args = argparse.Namespace(
        per_model_max_completion_batch_size="granite-3b=64,qwen3-8b=16",
        per_model_prefill_step_size=None,
    )
    settings._apply_cli_overrides(args)
    assert settings.scheduler.per_model_max_completion_batch_size == {
        "granite-3b": 64,
        "qwen3-8b": 16,
    }


def test_apply_args_parses_prefill_step_size_pairs(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    args = argparse.Namespace(
        per_model_max_completion_batch_size=None,
        per_model_prefill_step_size="granite-3b=512,qwen3-8b=2048",
    )
    settings._apply_cli_overrides(args)
    assert settings.scheduler.per_model_prefill_step_size == {
        "granite-3b": 512,
        "qwen3-8b": 2048,
    }


def test_apply_args_rejects_malformed_pair(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    args = argparse.Namespace(
        per_model_max_completion_batch_size="granite-3b=64,bogus",
        per_model_prefill_step_size=None,
    )
    with pytest.raises(ValueError, match="malformed pair"):
        settings._apply_cli_overrides(args)


def test_apply_args_rejects_non_integer_value(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    args = argparse.Namespace(
        per_model_max_completion_batch_size=None,
        per_model_prefill_step_size="granite-3b=not_a_number",
    )
    with pytest.raises(ValueError, match="not an integer"):
        settings._apply_cli_overrides(args)


def test_apply_args_accepts_dict_directly(tmp_path):
    """Pre-parsed dict (e.g. from JSON config) should pass through coerced."""
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    args = argparse.Namespace(
        per_model_max_completion_batch_size={"granite-3b": 64, "qwen3-8b": "16"},
        per_model_prefill_step_size={"granite-3b": "512"},
    )
    settings._apply_cli_overrides(args)
    assert settings.scheduler.per_model_max_completion_batch_size == {
        "granite-3b": 64,
        "qwen3-8b": 16,
    }
    assert settings.scheduler.per_model_prefill_step_size == {"granite-3b": 512}


def test_to_scheduler_config_propagates_per_model_dicts(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    settings.scheduler.per_model_max_completion_batch_size = {
        "granite-3b": 64,
        "qwen3-8b": 16,
    }
    settings.scheduler.per_model_prefill_step_size = {
        "granite-3b": 512,
        "qwen3-8b": 2048,
    }
    cfg = settings.to_scheduler_config()
    assert cfg.per_model_max_completion_batch_size == {
        "granite-3b": 64,
        "qwen3-8b": 16,
    }
    assert cfg.per_model_prefill_step_size == {
        "granite-3b": 512,
        "qwen3-8b": 2048,
    }


def test_scheduler_config_default_empty_dicts():
    from omlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.per_model_max_completion_batch_size == {}
    assert cfg.per_model_prefill_step_size == {}
