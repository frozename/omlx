# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from omlx.settings import GlobalSettings


def test_slot_save_path_defaults_to_none(tmp_path):
    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.slot_save_path is None


def test_cli_flag_sets_slot_save_path(tmp_path):
    slot_dir = tmp_path / "slots"
    args = Namespace(slot_save_path=str(slot_dir))

    settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

    assert settings.slot_save_path == str(slot_dir)


def test_slot_save_path_requires_single_concurrency(tmp_path):
    slot_dir = tmp_path / "slots"
    args = Namespace(slot_save_path=str(slot_dir), max_concurrent_requests=2)

    settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)
    errors = settings.validate()

    assert any("slot_save_path requires max_concurrent_requests=1" in error for error in errors)
