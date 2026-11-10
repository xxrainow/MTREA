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


def test_lab_api_drives_hold_owner_across_backend_restart(hold_system):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from lab.core import paths, serving
    from lab.core.executors.local import LocalExecutor
    from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
    from lab.server.main import create_app

    server, session, _, root = hold_system
    data = root / "lab-data"
    port = read(session / "status.json")["port"]
    checkpoint = str(root / "synthetic-checkpoint")
    children = []

    class FakeHardwareExecutor(LocalExecutor):
        def start(self, spec, log_path):
            if spec.kind == JobKind.ROLLOUT:
                runtime = {"data_dir": str(data), "task": spec.metadata["instruction"],
                           "time_limit_s": spec.metadata["time_limit_s"]}
                argv = [sys.executable, "-B", str(REPO / "tests" / "rollout_fake_worker.py"),
                        spec.run_dir, spec.metadata["server_address"],
                        spec.metadata["checkpoint"], "stop", json.dumps(runtime)]
                pid = super().start(replace(spec, cmd=serving.command(argv),
                                           metadata={**spec.metadata, "argv": argv}), log_path)
                children.append(self._procs[pid])
                return pid
            return super().start(spec, log_path)

    def app():
        return create_app(data_base=data, remote_config=root / "no-remote.yaml")

    def state(client, job_id):
        response = client.get(f"/api/rollouts/{job_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def command_done(client, job_id, command_id):
        response = client.get(f"/api/rollouts/{job_id}/commands/{command_id}")
        assert response.status_code == 200, response.text
        return response.json().get("state") == "completed"

    body = {"server_job_id": "real-hold", "instruction": "Synthetic API hold",
            "max_relative_target": 5.0, "time_limit_s": 30.0}
    try:
        with TestClient(app()) as client:
            client.app.state.runner.executors[Executor.LOCAL] = FakeHardwareExecutor(REPO)
            client.app.state.runner.store.save(Job(
                id="real-hold", pid=server.pid, status=JobStatus.RUNNING,
                spec=JobSpec(kind=JobKind.SERVE_POLICY, cmd="already running hold server",
                             run_dir=str(session), metadata={"checkpoint": checkpoint,
                                                             "server_port": port}),
            ))
            assert client.post("/api/rollouts/check", json=body).status_code == 200
            response = client.post("/api/rollouts/start", json=body)
            assert response.status_code == 201, response.text
            job_id = response.json()["job"]["id"]
            wait_for(lambda: (state(client, job_id).get("status") or {}).get("actions_executed", 0) >= 3,
                     processes=[server, children[0]])
            assert paths.hardware_lock(data).exists()
            from robot.config import CAMERAS
            camera_name = CAMERAS[0].name
            def live_frame():
                return client.get(f"/api/rollouts/{job_id}/frames/{camera_name}").status_code == 200
            wait_for(live_frame, processes=[server, children[0]])
            assert client.post("/api/rollouts/start", json=body).status_code == 409
            response = client.post(f"/api/rollouts/{job_id}/stop")
            assert response.status_code == 202
            wait_for(lambda: command_done(client, job_id, response.json()["command_id"]),
                     processes=[server, children[0]])
            assert state(client, job_id)["state"] == "stopped"
            assert state(client, job_id)["owner_alive"]
            assert client.get(f"/api/rollouts/{job_id}/frames/{camera_name}").status_code == 404

        # Restart drops executor Popen handles; ledger + PID/start-time still recover ownership.
        with TestClient(app()) as restarted:
            restored = state(restarted, job_id)
            assert restored["owner_alive"] and restored["status"]["workers_stopped"]
            assert restarted.get("/api/rollouts").json()[0]["job"]["id"] == job_id
            assert restarted.post("/api/cameras/start").status_code == 409
            assert restarted.post(f"/api/jobs/{job_id}/stop").status_code == 200
            assert children[0].poll() is None and paths.hardware_lock(data).exists()
            response = restarted.post(f"/api/rollouts/{job_id}/release")
            assert response.status_code == 202
            wait_for(lambda: command_done(restarted, job_id, response.json()["command_id"]),
                     processes=[server])
            children[0].wait(timeout=10)
            assert children[0].returncode == 0 and not paths.hardware_lock(data).exists()
            final = state(restarted, job_id)
            assert final["status"]["arm_released"] and not final["owner_alive"]
            assert restarted.get(f"/api/rollouts/{job_id}/logs").status_code == 200
    finally:
        for child in children:
            if child.poll() is None:
                # The helper forbids hardware access; forcibly clean up a failed test.
                import psutil
                process = psutil.Process(child.pid)
                for descendant in process.children(recursive=True):
                    descendant.kill()
                child.kill()
            child.wait(timeout=10)
