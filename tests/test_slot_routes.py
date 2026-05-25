# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from omlx.settings import GlobalSettings


def _configure_slot_state(slot_save_path: str | None, max_concurrent_requests: int = 1, api_key: str | None = None):
    from omlx.server import _server_state

    settings = GlobalSettings()
    settings.slot_save_path = slot_save_path
    settings.scheduler.max_concurrent_requests = max_concurrent_requests

    _server_state.global_settings = settings
    _server_state.api_key = api_key


def test_slot_route_returns_404_when_disabled(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        _configure_slot_state(slot_save_path=None)
        client = TestClient(app)
        response = client.post("/slots/0?action=save", json={"filename": "slot.kv"})
        assert response.status_code == 404
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key


def test_slot_route_returns_501_stub_when_enabled_and_valid(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        _configure_slot_state(slot_save_path=str(slot_dir))

        client = TestClient(app)
        response = client.post("/slots/0?action=save", json={"filename": "slot.kv"})
        assert response.status_code == 501
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key


def test_slot_route_rejects_invalid_action(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        _configure_slot_state(slot_save_path=str(slot_dir))

        client = TestClient(app)
        response = client.post("/slots/0?action=invalid", json={"filename": "slot.kv"})
        assert response.status_code == 400
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key


def test_slot_route_rejects_invalid_filename(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        _configure_slot_state(slot_save_path=str(slot_dir))

        client = TestClient(app)

        response = client.post("/slots/0?action=save", json={"filename": "../slot.kv"})
        assert response.status_code == 400

        response = client.post("/slots/0?action=save", json={"filename": "/tmp/slot.kv"})
        assert response.status_code == 400
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key


def test_slot_route_rejects_non_zero_slot_id(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        _configure_slot_state(slot_save_path=str(slot_dir))

        client = TestClient(app)
        response = client.post("/slots/1?action=save", json={"filename": "slot.kv"})
        assert response.status_code == 404
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key


def test_slot_route_honors_auth_gate(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    original_api_key = _server_state.api_key
    try:
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        _configure_slot_state(slot_save_path=str(slot_dir), api_key="secret")

        client = TestClient(app)

        unauthorized = client.post("/slots/0?action=save", json={"filename": "slot.kv"})
        assert unauthorized.status_code == 401

        authorized = client.post(
            "/slots/0?action=save",
            json={"filename": "slot.kv"},
            headers={"Authorization": "Bearer secret"},
        )
        assert authorized.status_code == 501
    finally:
        _server_state.global_settings = original_settings
        _server_state.api_key = original_api_key
