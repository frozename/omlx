# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from omlx.settings import GlobalSettings


def test_slot_capabilities_shape_and_version(tmp_path):
    from omlx.server import app, _server_state

    original_settings = _server_state.global_settings
    try:
        settings = GlobalSettings()
        settings.slot_save_path = None
        settings.scheduler.max_concurrent_requests = 8
        _server_state.global_settings = settings

        client = TestClient(app)
        response = client.get("/v1/slots/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert data["slot_api_version"] == "0.1.0"
        assert data["actions"] == ["save", "restore"]
        assert data["slot_count"] == 1
        assert data["max_concurrent_requests"] == 8
        assert data["slot_save_path_configured"] is False
    finally:
        _server_state.global_settings = original_settings
