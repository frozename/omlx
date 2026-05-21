"""Tests for the per-model concurrency override on SchedulerSettings.

The override lets operators run heterogeneous concurrency budgets across
co-resident models on a shared GPU (e.g. mcr=4 on a 3B + mcr=1 on an 8B).
"""

from __future__ import annotations

import argparse

import pytest


def test_settings_default_empty_dict():
    from omlx.settings import SchedulerSettings

    s = SchedulerSettings()
    assert s.per_model_max_concurrent == {}


def test_settings_round_trip_via_to_from_dict():
    from omlx.settings import SchedulerSettings

    original = SchedulerSettings(
        max_concurrent_requests=8,
        per_model_max_concurrent={"granite-3b": 8, "qwen3-8b": 4},
    )
    round_tripped = SchedulerSettings.from_dict(original.to_dict())
    assert round_tripped.per_model_max_concurrent == {
        "granite-3b": 8,
        "qwen3-8b": 4,
    }
    assert round_tripped.max_concurrent_requests == 8


def test_from_dict_coerces_values_to_int():
    from omlx.settings import SchedulerSettings

    raw = {
        "max_concurrent_requests": 8,
        "per_model_max_concurrent": {"granite-3b": "8", "qwen3-8b": "4"},
    }
    parsed = SchedulerSettings.from_dict(raw)
    assert parsed.per_model_max_concurrent == {"granite-3b": 8, "qwen3-8b": 4}
    assert all(isinstance(v, int) for v in parsed.per_model_max_concurrent.values())


def test_apply_args_parses_cli_pairs():
    from omlx.settings import Settings

    settings = Settings()
    args = argparse.Namespace(
        per_model_max_concurrent="granite-3b=8,qwen3-8b=4",
    )
    settings.apply_args(args)
    assert settings.scheduler.per_model_max_concurrent == {
        "granite-3b": 8,
        "qwen3-8b": 4,
    }


def test_apply_args_ignores_empty_and_whitespace_pairs():
    from omlx.settings import Settings

    settings = Settings()
    args = argparse.Namespace(
        per_model_max_concurrent="granite-3b=8, ,qwen3-8b=4,",
    )
    settings.apply_args(args)
    assert settings.scheduler.per_model_max_concurrent == {
        "granite-3b": 8,
        "qwen3-8b": 4,
    }


def test_apply_args_rejects_malformed_pair():
    from omlx.settings import Settings

    settings = Settings()
    args = argparse.Namespace(per_model_max_concurrent="granite-3b=8,bogus")
    with pytest.raises(ValueError, match="malformed pair"):
        settings.apply_args(args)


def test_apply_args_rejects_non_integer_value():
    from omlx.settings import Settings

    settings = Settings()
    args = argparse.Namespace(per_model_max_concurrent="granite-3b=not_a_number")
    with pytest.raises(ValueError, match="not an integer"):
        settings.apply_args(args)


def test_apply_args_accepts_dict_directly():
    """When the value comes pre-parsed (e.g. from a JSON config that already
    contains a dict), apply_args should pass it through coerced to int.
    """
    from omlx.settings import Settings

    settings = Settings()
    args = argparse.Namespace(
        per_model_max_concurrent={"granite-3b": 8, "qwen3-8b": "4"},
    )
    settings.apply_args(args)
    assert settings.scheduler.per_model_max_concurrent == {
        "granite-3b": 8,
        "qwen3-8b": 4,
    }


def test_to_scheduler_config_propagates_per_model_dict():
    from omlx.settings import Settings

    settings = Settings()
    settings.scheduler.per_model_max_concurrent = {"granite-3b": 8, "qwen3-8b": 4}
    cfg = settings.to_scheduler_config()
    assert cfg.per_model_max_concurrent == {"granite-3b": 8, "qwen3-8b": 4}
