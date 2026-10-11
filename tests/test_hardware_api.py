"""Hardware routes with MTREA_HARDWARE=fake. No lerobot, no robot."""

import time

import pytest
from fastapi.testclient import TestClient

from lab.core import paths
from lab.server.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("HF_HOME", "HF_LEROBOT_HOME", "HF_LEROBOT_CALIBRATION", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(create_app(data_base=tmp_path / "data")) as c:
        yield c


def wait_state(client, state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/calibration/status").json()
        if body["state"] == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"calibration never reached {state}")


def test_config_and_ports(client):
    config = client.get("/api/hardware/config").json()
    assert config["backend"] == "fake"
    assert config["arms"]["follower"]["id"] and config["arms"]["leader"]["id"]
    assert config["arms"]["follower"]["calibration_present"] is False
    assert config["arms"]["follower"]["config_file"].replace("\\", "/").endswith("fake/robot_configs/so101_follower.yaml")
    ports = client.get("/api/hardware/ports").json()
    assert "/dev/tty.fake-follower" in ports["ports"]


def test_detect_and_save_port(client):
    start = client.post("/api/hardware/ports/detect/start", json={"arm": "follower"})
    assert start.status_code == 200 and "/dev/tty.fake-follower" in start.json()["ports_before"]
    finish = client.post("/api/hardware/ports/detect/finish", json={"arm": "follower"})
    assert finish.status_code == 200 and finish.json()["port"] == "/dev/tty.fake-follower"
    saved = client.post("/api/hardware/ports/save", json={"arm": "follower", "port": finish.json()["port"]})
    assert saved.status_code == 200
    config = client.get("/api/hardware/config").json()
    assert config["arms"]["follower"]["port"] == "/dev/tty.fake-follower"
    assert config["arms"]["follower"]["port_present"] is True


def test_detect_finish_without_start_and_bad_arm(client):
    assert client.post("/api/hardware/ports/detect/finish", json={"arm": "leader"}).status_code == 409
    assert client.post("/api/hardware/ports/detect/start", json={"arm": "middle"}).status_code == 422
    assert client.post("/api/hardware/ports/save", json={"arm": "leader", "port": ""}).status_code == 422


def test_calibration_flow_and_conflicts(client):
    assert client.post("/api/calibration/next").status_code == 409
    assert client.post("/api/calibration/cancel").status_code == 409

    started = client.post("/api/calibration/start", json={"arm": "leader"})
    assert started.status_code == 200 and started.json()["backend"] == "fake"
    again = client.post("/api/calibration/start", json={"arm": "follower"})
    assert again.status_code == 409 and "already running" in again.json()["detail"]
    teleop = client.post("/api/teleop/start")
    assert teleop.status_code == 409 and "calibration of the leader arm" in teleop.json()["detail"]
    detect = client.post("/api/hardware/ports/detect/start", json={"arm": "leader"})
    assert detect.status_code == 409

    homing = wait_state(client, "homing")
    assert homing["can_next"] and "middle of its range" in homing["instruction"]
    assert client.post("/api/calibration/next").status_code == 200
    wait_state(client, "recording_ranges")
    time.sleep(0.4)
    assert client.post("/api/calibration/next").status_code == 200
    done = wait_state(client, "done")
    assert done["saved_path"].endswith("so-arm1.fake.json")
    assert client.post("/api/calibration/next").status_code == 409
    assert client.get("/api/hardware/config").json()["arms"]["leader"]["calibration_present"] is True


def test_calibration_cancel(client):
    client.post("/api/calibration/start", json={"arm": "follower"})
    wait_state(client, "homing")
    cancelled = client.post("/api/calibration/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled"
    assert client.get("/api/hardware/config").json()["lock"] is None


def test_teleop_and_joint_stream(client):
    assert client.post("/api/teleop/stop").status_code == 409
    started = client.post("/api/teleop/start")
    assert started.status_code == 200 and started.json()["running"] and started.json()["backend"] == "fake"
    assert client.post("/api/teleop/start").status_code == 409
    assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 409

    with client.websocket_connect("/ws/joints") as ws:
        frames = [ws.receive_json() for _ in range(3)]
    assert all(f["type"] == "teleop" for f in frames)
    assert set(frames[0]["leader"]) == set(frames[0]["follower"]) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}

    status = client.get("/api/teleop/status").json()
    assert status["running"] and status["fps"] > 0
    stopped = client.post("/api/teleop/stop")
    assert stopped.status_code == 200 and not stopped.json()["running"]


def test_joint_stream_during_calibration(client):
    client.post("/api/calibration/start", json={"arm": "leader"})
    with client.websocket_connect("/ws/joints") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "calibration" and frame["arm"] == "leader" and frame["joints"]
    client.post("/api/calibration/cancel")


def test_shutdown_stops_the_session(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    data = tmp_path / "data"
    with TestClient(create_app(data_base=data)) as c:
        assert c.post("/api/teleop/start").status_code == 200
        assert paths.hardware_lock(data).exists()
    assert not paths.hardware_lock(data).exists()
