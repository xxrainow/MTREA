"""Actual hold server + installed LeRobot client, replacing hardware only."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robot.config import CAMERAS, JOINT_ORDER
from robot.rollout_commands import submit_command

REPO = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def wait_for(predicate, *, processes=(), timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        for process in processes:
            assert process.poll() is None, f"Process exited unexpectedly: {process.returncode}"
        time.sleep(0.025)
    raise AssertionError("Timed out waiting for hold integration state; inspect test logs")


@pytest.fixture
def hold_system(tmp_path):
    pytest.importorskip("grpc")
    pytest.importorskip("lerobot.async_inference.configs")
    checkpoint = tmp_path / "synthetic-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "pi05", "chunk_size": 50,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [len(JOINT_ORDER)]},
            **{f"observation.images.{camera.name}": {
                "type": "VISUAL", "shape": [3, camera.height, camera.width],
            } for camera in CAMERAS},
        },
        "action_feature_names": [f"{joint}.pos" for joint in JOINT_ORDER],
    }))
    processes = []
    streams = []

    def start(command, name):
        stream = (tmp_path / f"{name}.log").open("w")
        streams.append(stream)
        process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1",
                                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        processes.append(process)
        return process

    session = tmp_path / "server-data" / "rollouts" / "serve" / "integration"
    try:
        server = start([
            sys.executable, "-B", "-m", "research.serve_policy", "--fake=hold", "--port=0",
            f"--checkpoint={checkpoint}", f"--data-dir={tmp_path / 'server-data'}",
            "--session=integration",
        ], "server")
        wait_for(lambda: read(session / "status.json").get("status") == "ready", processes=[server])
        port = read(session / "status.json")["port"]

        def client(mode):
            trial = tmp_path / "trial"
            client_checkpoint = str(checkpoint) + "-wrong" if mode == "checkpoint_mismatch" else str(checkpoint)
            process = start([
                sys.executable, "-B", str(REPO / "tests" / "rollout_fake_worker.py"),
                str(trial), f"127.0.0.1:{port}", client_checkpoint, mode,
            ], "client")
            return process, trial

        yield server, session, client, tmp_path
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                # A failed assertion must never leave the synthetic owner running.
                process.kill()
            process.wait(timeout=10)
        for stream in streams:
            stream.close()


@pytest.mark.parametrize("mode", ["stop", "time_limit", "checkpoint_mismatch", "shape_mismatch", "server_disconnect"])
def test_hold_protocol_and_owner_lifecycle(hold_system, mode):
    server, session, start_client, root = hold_system
    client, trial = start_client(mode)
    status_path = trial / "status.json"
    bad_config = mode in {"checkpoint_mismatch", "shape_mismatch"}
    if bad_config:
        wait_for(lambda: read(status_path).get("state") == "failed", processes=[client, server])
    else:
        wait_for(lambda: read(status_path).get("actions_executed", 0) >= 3,
                 processes=[client, server])
        if mode == "stop":
            wait_for(lambda: read(trial / "frames" / "frames.json"), processes=[client, server])
            command = submit_command(trial, "stop")
            wait_for(lambda: read(trial / "command_results" / f"{command}.json").get("state") == "completed",
                     processes=[client, server])
        elif mode == "server_disconnect":
            server.terminate()
            server.wait(timeout=15)
            wait_for(lambda: read(status_path).get("state") == "aborted", processes=[client])

    wait_for(lambda: read(status_path).get("workers_stopped"), processes=[client])
    stopped = read(status_path)
    assert stopped["process_alive"] and not stopped["arm_released"]
    assert stopped["hold_command_sent"] and (root / "test-hardware.lock").exists()
    count = stopped["actions_executed"]
    time.sleep(0.15)
    assert read(status_path)["actions_executed"] == count
    if bad_config:
        assert count == 0 and stopped["reason"] == "connection"
        expected = "pretrained_name_or_path" if mode == "checkpoint_mismatch" else "shape"
        assert expected in stopped["error"]
    else:
        assert stopped["chunks_received"] > 0 and count >= 3
        assert stopped["last_latency_ms"] is not None
        assert stopped["state"] == ("aborted" if mode == "server_disconnect" else "stopped")
        assert stopped["reason"] == {"stop": "user_stop", "time_limit": "time_limit",
                                       "server_disconnect": "connection"}[mode]

    release = submit_command(trial, "release")
    client.wait(timeout=10)
    assert client.returncode == (1 if bad_config or mode == "server_disconnect" else 0)
    assert read(trial / "command_results" / f"{release}.json")["state"] == "completed"
    assert not (root / "test-hardware.lock").exists()
    final = read(status_path)
    assert final["arm_released"] and not final["process_alive"]
    robot = read(trial / "fake_robot.json")
    assert robot["torque_disabled"] and not robot["bus_connected"] and not robot["cameras_connected"]
    assert robot["hold_targets"] == [robot["initial"]]
    for action in robot["actions"]:
        assert action == pytest.approx({f"{joint}.pos": value for joint, value in robot["initial"].items()})
    assert len(robot["actions"]) == count
    if not bad_config:
        assert robot["observations"] > 0
        assert read(session / "status.json")["counts"]["chunks"] > 0
    metadata = read(trial / "session.json")
    assert metadata["lerobot_version"] == "0.6.1"
    assert metadata["git_commit"] and metadata["robot_id"] == "synthetic"
