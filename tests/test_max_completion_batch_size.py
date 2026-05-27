# SPDX-License-Identifier: Apache-2.0
"""Tests for the --max-completion-batch-size scheduler knob."""

from argparse import Namespace
from pathlib import Path

from omlx.settings import GlobalSettings, SchedulerSettings


def test_default_is_none() -> None:
    assert SchedulerSettings().max_completion_batch_size is None


def test_to_dict_round_trip_preserves_value() -> None:
    original = SchedulerSettings(
        max_concurrent_requests=4,
        max_completion_batch_size=2,
    )

    restored = SchedulerSettings.from_dict(original.to_dict())

    assert restored.max_completion_batch_size == 2
    assert restored.max_concurrent_requests == 4


def test_from_dict_migrates_missing_key_to_none() -> None:
    settings = SchedulerSettings.from_dict({"max_concurrent_requests": 4})

    assert settings.max_completion_batch_size is None
    assert settings.max_concurrent_requests == 4


def test_to_scheduler_config_falls_back_to_admission(tmp_path: Path) -> None:
    settings = GlobalSettings(base_path=tmp_path)
    settings.scheduler.max_concurrent_requests = 4
    settings.scheduler.max_completion_batch_size = None

    config = settings.to_scheduler_config()

    assert config.completion_batch_size == 4


def test_to_scheduler_config_uses_override_when_set(tmp_path: Path) -> None:
    settings = GlobalSettings(base_path=tmp_path)
    settings.scheduler.max_concurrent_requests = 4
    settings.scheduler.max_completion_batch_size = 1

    config = settings.to_scheduler_config()

    assert config.completion_batch_size == 1
    # Admission limit is independent and stays at max_concurrent_requests.
    assert config.max_num_seqs == 4


def test_apply_cli_overrides_sets_max_completion_batch_size(tmp_path: Path) -> None:
    settings = GlobalSettings(base_path=tmp_path)
    settings.scheduler.max_concurrent_requests = 8

    settings._apply_cli_overrides(
        Namespace(max_completion_batch_size=1, max_concurrent_requests=None)
    )

    assert settings.scheduler.max_completion_batch_size == 1
    # Admission limit untouched.
    assert settings.scheduler.max_concurrent_requests == 8


def test_apply_cli_overrides_none_leaves_field_unset(tmp_path: Path) -> None:
    settings = GlobalSettings(base_path=tmp_path)

    settings._apply_cli_overrides(
        Namespace(max_completion_batch_size=None, max_concurrent_requests=None)
    )

    assert settings.scheduler.max_completion_batch_size is None
