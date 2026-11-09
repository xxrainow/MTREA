"""Serving API checks without remote connections or model execution."""

import pytest
from fastapi.testclient import TestClient

from lab.server.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(
        data_base=tmp_path / "data",
        remote_config=tmp_path / "no-remote.yaml",
    )

    with TestClient(app) as test_client:
        yield test_client


def test_sessions_are_empty_without_server_jobs(client):
    response = client.get("/api/serve/sessions")

    assert response.status_code == 200
    assert response.json() == []


def test_real_serving_on_local_machine_is_refused(client):
    response = client.post(
        "/api/serve/start",
        json={
            "target": "local",
            "mode": "real",
        },
    )

    assert response.status_code == 400
    assert "fake hold only" in response.json()["detail"]


def test_tunnel_for_missing_session_is_not_found(client):
    response = client.post(
        "/api/serve/sessions/no-such-job/tunnel"
    )

    assert response.status_code == 404
