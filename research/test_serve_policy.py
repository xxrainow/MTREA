"""Tests for the policy server, its validation and its check client. No GPU, no robot.

Validation needs nothing beyond the repo. Anything that talks gRPC is skipped
when lerobot or grpc is missing, and anything that reads a dataset when pyarrow
or av is missing (lerobot[async] alone has neither).
"""

import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research import serve_common
from research.serve_common import (CheckpointError, ChunkErrors, ServedPolicy, checkpoint_provenance,
                                   fake_served_policy, feature_mismatches, gpu_busy_reason, instruction_mismatches,
                                   is_loopback, percentile, same_checkpoint, validate_checkpoint_dir)

REPO_ROOT = Path(__file__).resolve().parent.parent
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
HEIGHT, WIDTH = 48, 64
FRAMES_PER_EPISODE = 45
INPUT_FEATURES = {
    "observation.state": {"type": "STATE", "shape": [6]},
    "observation.images.wrist": {"type": "VISUAL", "shape": [3, HEIGHT, WIDTH]},
    "observation.images.scene": {"type": "VISUAL", "shape": [3, HEIGHT, WIDTH]},
}


def checkpoint_config(**changes) -> dict:
    return {"type": "pi05", "chunk_size": 50, "input_features": INPUT_FEATURES,
            "action_feature_names": JOINTS, **changes}


def write_checkpoint(directory: Path, config=None, skip=()) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": json.dumps(config or checkpoint_config()),
        "model.safetensors": "weights",
        "policy_preprocessor.json": json.dumps({"steps": [{"state_file": "pre_norm.safetensors"}]}),
        "policy_postprocessor.json": json.dumps({"steps": [{"state_file": "post_norm.safetensors"}]}),
        "pre_norm.safetensors": "stats",
        "post_norm.safetensors": "stats",
    }
    for name, content in files.items():
        if name not in skip:
            (directory / name).write_text(content, encoding="utf-8")
    return directory


def served(checkpoint="/runs/x/pretrained_model") -> ServedPolicy:
    return ServedPolicy(checkpoint=checkpoint, resolved=os.path.realpath(checkpoint), policy_type="pi05",
                        chunk_size=50, input_features=INPUT_FEATURES, state_names=JOINTS)


def client_features(wrist=(HEIGHT, WIDTH, 3), scene=(HEIGHT, WIDTH, 3), joints=None) -> dict:
    """As hw_to_dataset_features builds them for an so101_follower with these cameras."""
    joints = JOINTS if joints is None else joints
    image = {"dtype": "image", "names": ["height", "width", "channels"]}
    return {"observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
            "observation.images.wrist": {**image, "shape": wrist},
            "observation.images.scene": {**image, "shape": scene}}


# --- checkpoint validation ---

def test_complete_checkpoint_is_accepted(tmp_path):
    assert validate_checkpoint_dir(write_checkpoint(tmp_path / "ckpt"))["type"] == "pi05"


@pytest.mark.parametrize("missing, named", [
    ("model.safetensors", "weights"),
    ("policy_preprocessor.json", "policy_preprocessor.json"),
    ("policy_postprocessor.json", "policy_postprocessor.json"),
    ("pre_norm.safetensors", "pre_norm.safetensors"),
    ("config.json", "config.json"),
])
def test_incomplete_checkpoint_is_refused(tmp_path, missing, named):
    with pytest.raises(CheckpointError, match=named):
        validate_checkpoint_dir(write_checkpoint(tmp_path / "ckpt", skip=(missing,)))


def test_empty_weights_are_refused(tmp_path):
    ckpt = write_checkpoint(tmp_path / "ckpt")
    (ckpt / "model.safetensors").write_bytes(b"")
    with pytest.raises(CheckpointError, match="no weights"):
        validate_checkpoint_dir(ckpt)


def test_wrong_policy_type_is_refused(tmp_path):
    ckpt = write_checkpoint(tmp_path / "ckpt", config=checkpoint_config(type="sac"))
    with pytest.raises(CheckpointError, match="'sac'"):
        validate_checkpoint_dir(ckpt)


def test_fake_run_checkpoint_is_refused(tmp_path):
    # research/fake_lerobot_train.py writes "fake weights" into model.safetensors.
    ckpt = write_checkpoint(tmp_path / "ckpt", config=checkpoint_config(fake=True))
    with pytest.raises(CheckpointError, match="fake run"):
        validate_checkpoint_dir(ckpt)


def _run_checkpoint(tmp_path, step=2000, last=2000, run_config=None) -> Path:
    run = tmp_path / "runs" / "h0_x_s1"
    checkpoints = run / "lerobot" / "checkpoints"
    ckpt = write_checkpoint(checkpoints / f"{step:06d}" / "pretrained_model")
    if last is not None:
        (checkpoints / f"{last:06d}").mkdir(exist_ok=True)
        try:
            (checkpoints / "last").symlink_to(f"{last:06d}")
        except OSError:
            pytest.skip("symlinks need elevated rights here")
    (run / "config.json").write_text(json.dumps(run_config or {
        "args": {"method": "h0", "seed": 1, "dataset": "local/x"}, "git_commit": "abc", "fake": False}),
        encoding="utf-8")
    return ckpt


def test_provenance_from_the_run_dir(tmp_path):
    info = checkpoint_provenance(_run_checkpoint(tmp_path))
    assert (info["step"], info["run_id"], info["method"], info["seed"], info["train_git_commit"], info["complete"]) \
        == (2000, "h0_x_s1", "h0", 1, "abc", True)


def test_provenance_outside_a_run_dir_is_null(tmp_path):
    info = checkpoint_provenance(write_checkpoint(tmp_path / "ckpt"))
    assert info["step"] is None and info["run_id"] is None


def test_checkpoint_past_last_is_refused(tmp_path):
    # docs/TRAINING_JOBS.md: only complete checkpoints may be served.
    with pytest.raises(CheckpointError, match="not complete"):
        validate_checkpoint_dir(_run_checkpoint(tmp_path, step=4000, last=2000))


def test_checkpoint_of_a_fake_run_dir_is_refused(tmp_path):
    ckpt = _run_checkpoint(tmp_path, run_config={"args": {}, "fake": True})
    with pytest.raises(CheckpointError, match="fake run"):
        validate_checkpoint_dir(ckpt)


# --- host and GPU ---

@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost"])
def test_loopback_hosts_pass(host):
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "::", "", "no-such-host.invalid"])
def test_other_hosts_are_refused(host):
    assert not is_loopback(host)


GIB = 1 << 30


def test_idle_slice_is_not_busy():
    assert gpu_busy_reason(39 * GIB, 39 * GIB + GIB // 4) is None


@pytest.mark.parametrize("held_gib", [37, 13])  # H0 and expert-only training, docs/COMPUTE.md
def test_training_on_the_slice_is_busy(held_gib):
    assert gpu_busy_reason((39 - held_gib) * GIB, 39 * GIB).startswith("GPU busy")


def test_small_device_is_refused():
    assert "GPU busy" in gpu_busy_reason(8 * GIB, 8 * GIB)


# --- client validation ---

def test_matching_features_pass():
    assert feature_mismatches(client_features(), served()) == []


def test_renamed_cameras_are_named():
    client = client_features()
    client["observation.images.front"] = client.pop("observation.images.scene")
    problems = " | ".join(feature_mismatches(client, served()))
    assert "observation.images.scene is missing" in problems
    assert "observation.images.front is not in the checkpoint" in problems


def test_swapped_camera_names_with_different_shapes_are_named():
    policy = ServedPolicy("c", "c", "pi05", 50, {
        "observation.state": {"shape": [6]},
        "observation.images.wrist": {"shape": [3, 480, 640]},
        "observation.images.scene": {"shape": [3, 720, 1280]},
    }, JOINTS)
    # The checkpoint's wrist resolution arrives under the name scene, and the other way round.
    problems = feature_mismatches(client_features(wrist=(720, 1280, 3), scene=(480, 640, 3)), policy)
    assert len(problems) == 2
    assert "observation.images.wrist has shape [720, 1280, 3]" in problems[0] + problems[1]
    assert "observation.images.scene has shape [480, 640, 3]" in problems[0] + problems[1]


def test_swap_by_rename_map_is_refused():
    swap = {"observation.images.wrist": "observation.images.scene",
            "observation.images.scene": "observation.images.wrist"}
    problems = instruction_mismatches(policy_type="pi05", pretrained_name_or_path=served().checkpoint,
                                      actions_per_chunk=50, lerobot_features=client_features(), rename_map=swap,
                                      served=served())
    assert len(problems) == 1 and "rename_map" in problems[0]


def test_wrong_image_shape_is_named():
    problems = feature_mismatches(client_features(wrist=(HEIGHT * 2, WIDTH, 3)), served())
    assert len(problems) == 1 and "observation.images.wrist" in problems[0]


def test_channels_first_from_a_client_is_not_mistaken_for_a_match():
    assert feature_mismatches(client_features(scene=(3, HEIGHT, WIDTH)), served())


def test_wrong_state_dim_is_named():
    problems = feature_mismatches(client_features(joints=JOINTS[:5]), served())
    assert any("[5]" in p and "[6]" in p for p in problems)


def test_reordered_joints_are_named():
    problems = feature_mismatches(client_features(joints=[JOINTS[1], JOINTS[0], *JOINTS[2:]]), served())
    assert len(problems) == 1 and "in that order" in problems[0]


def test_every_mismatch_is_named_together():
    client = client_features(wrist=(1, 2, 3), joints=JOINTS[:5])
    del client["observation.images.scene"]
    problems = instruction_mismatches(policy_type="act", pretrained_name_or_path="/other/ckpt", actions_per_chunk=99,
                                      lerobot_features=client, rename_map={}, served=served())
    text = " | ".join(problems)
    for expected in ("/other/ckpt", "'act'", "actions_per_chunk 99", "observation.images.scene is missing",
                     "observation.images.wrist has shape", "observation.state has shape"):
        assert expected in text


@pytest.mark.parametrize("client, ok", [
    ("/runs/x/pretrained_model", True),
    ("/runs/x/../x/pretrained_model", True),
    ("/runs/y/pretrained_model", False),
    ("lerobot/pi05_base", False),
])
def test_client_path_must_name_the_loaded_checkpoint(client, ok):
    assert same_checkpoint(client, served()) is ok


def test_hub_id_is_compared_as_a_string():
    assert same_checkpoint("lerobot/pi05_base", served("lerobot/pi05_base"))


def test_last_and_its_step_dir_are_the_same_checkpoint(tmp_path):
    (tmp_path / "010000").mkdir()
    try:
        (tmp_path / "last").symlink_to("010000")
    except OSError:
        pytest.skip("symlinks need elevated rights here")
    assert same_checkpoint(str(tmp_path / "010000"), served(str(tmp_path / "last")))


def test_fake_server_without_any_features_is_refused(tmp_path):
    with pytest.raises(CheckpointError, match="--dataset-root"):
        fake_served_policy(str(tmp_path / "not-here"), None)


def test_fake_server_reads_a_fake_runs_config_then_the_dataset(tmp_path):
    # A fake training run's config.json has no input_features.
    ckpt = write_checkpoint(tmp_path / "ckpt", config={"type": "pi05", "fake": True})
    with pytest.raises(CheckpointError, match="--dataset-root"):
        fake_served_policy(str(ckpt), None)


# --- numbers ---

def test_percentile_is_a_value_that_happened():
    values = [float(i) for i in range(1, 101)]
    assert (percentile(values, 50), percentile(values, 95), percentile([], 50)) == (50.0, 95.0, None)


def test_chunk_errors_pool_by_horizon():
    errors = ChunkErrors(["a", "b"])
    errors.add([[1.0, 1.0]] * 50, [[0.0, 0.0]] + [[1.0, 3.0]] * 49)
    summary = errors.summary()
    assert summary["first_1"]["mae"] == {"a": 1.0, "b": 1.0}
    assert summary["first_10"]["n"] == 10
    assert summary["first_50"]["mae"]["a"] == pytest.approx(0.02)
    assert summary["first_50"]["rmse"]["b"] == pytest.approx(1.98, abs=0.01)


def test_chunk_errors_stop_at_the_episode_end():
    errors = ChunkErrors(["a"])
    errors.add([[1.0]] * 50, [[1.0]] * 3)
    assert errors.summary()["first_50"]["n"] == 3


# --- dataset ---

def _write_dataset(root: Path) -> Path:
    """A 2-episode v3.0 dataset with real parquet and MP4 files, small enough to run in real time."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    n = FRAMES_PER_EPISODE
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {
        "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
    }
    for camera in ("wrist", "scene"):
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [HEIGHT, WIDTH, 3],
                                                    "names": ["height", "width", "channels"]}
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30, "features": features,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }), encoding="utf-8")
    pq.write_table(pa.table({"task_index": [0, 1], "task": ["task zero", "task one"]}), root / "meta" / "tasks.parquet")

    frames = 2 * n
    episode = np.repeat([0, 1], n)
    frame = np.tile(np.arange(n), 2)
    # Joints move about 2 per frame, so the similarity filter does not hold observations back.
    state = np.stack([2.0 * np.arange(frames) + j for j in range(6)], axis=1).astype(np.float32)
    action = state + 0.5
    pq.write_table(pa.table({
        "index": np.arange(frames), "episode_index": episode, "frame_index": frame, "task_index": episode,
        "action": action.tolist(), "observation.state": state.tolist(),
    }), root / "data" / "chunk-000" / "file-000.parquet")

    row = {"episode_index": [0, 1]}
    for camera in ("wrist", "scene"):
        key = f"observation.images.{camera}"
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            for i in range(frames):
                image = np.full((HEIGHT, WIDTH, 3), i * 2 % 255, dtype=np.uint8)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
            container.mux(stream.encode())
        row[f"videos/{key}/chunk_index"] = [0, 0]
        row[f"videos/{key}/file_index"] = [0, 0]
        row[f"videos/{key}/from_timestamp"] = [0.0, n / 30]
    pq.write_table(pa.table(row), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def test_dataset_reads_episodes_tasks_and_frames(tmp_path):
    dataset = serve_common.DemoDataset(_write_dataset(tmp_path / "ds"))
    assert dataset.cameras == ["scene", "wrist"]
    assert dataset.episodes == [0, 1]
    assert dataset.episode_range(1) == (FRAMES_PER_EPISODE, 2 * FRAMES_PER_EPISODE)
    assert dataset.task_of(FRAMES_PER_EPISODE) == "task one"
    assert dataset.eligible_frames(FRAMES_PER_EPISODE)[0] == range(0, 1)
    assert dataset.eligible_frames(FRAMES_PER_EPISODE + 1) == {}
    assert len(dataset.episode_images(1)["wrist"]) == FRAMES_PER_EPISODE
    # Decoded by timestamp from a file shared between episodes: frame 3 of episode 1 has
    # brightness 2 * (FRAMES_PER_EPISODE + 3).
    pixel = dataset.frame_images(1, 3)["scene"].mean()
    assert abs(pixel - 2 * (FRAMES_PER_EPISODE + 3)) < 6


def test_samples_stay_inside_episodes_and_repeat_with_a_seed(tmp_path):
    pytest.importorskip("numpy")
    from research.infer_offline import sample_frames

    dataset = serve_common.DemoDataset(_write_dataset(tmp_path / "ds"))
    frames = sample_frames(dataset, 10, seed=3, horizon=40)
    assert frames == sample_frames(dataset, 10, seed=3, horizon=40)
    assert len(frames) == 10 and {e for e, _ in frames} == {0, 1}
    assert all(f + 40 <= FRAMES_PER_EPISODE for _, f in frames)


# --- the server process ---

class Server:
    """A fake server subprocess, with its stdout lines."""

    def __init__(self, tmp_path: Path, *flags: str, session="s1"):
        self.data_dir = tmp_path / "data"
        self.session = self.data_dir / "rollouts" / "serve" / session
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "research.serve_policy", "--port=0", f"--data-dir={self.data_dir}",
             f"--session={session}", *flags],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.lines: list[str] = []
        self.port = self._wait_ready()

    def _wait_ready(self) -> int:
        for line in self.proc.stdout:
            self.lines.append(line.strip())
            if line.startswith("STATUS ready port="):
                return int(line.split("=")[1])
            if line.startswith("STATUS failed"):
                break
        raise AssertionError(f"server did not start: {self.lines}\n{self.proc.stderr.read()}")

    def stop(self) -> int:
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=30)
        self.lines += out.splitlines()
        return self.proc.returncode

    def json(self, name: str) -> dict:
        return json.loads((self.session / name).read_text(encoding="utf-8"))

    def chunks(self) -> list[dict]:
        text = (self.session / "inference.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]


@pytest.fixture
def needs_grpc(tmp_path):
    pytest.importorskip("grpc")
    with serve_common.lerobot_log_dir(tmp_path / "import_logs"):
        pytest.importorskip("lerobot.async_inference.policy_server")


@pytest.fixture
def checkpoint(tmp_path) -> str:
    return str(write_checkpoint(tmp_path / "ckpt"))


def _stub(port: int):
    import grpc

    from lerobot.transport import services_pb2_grpc

    return services_pb2_grpc.AsyncInferenceStub(grpc.insecure_channel(f"127.0.0.1:{port}"))


def _instructions(checkpoint: str, features=None, policy_type="pi05", actions=50, rename_map=None):
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.transport import services_pb2

    specs = RemotePolicyConfig(policy_type, checkpoint, features or client_features(), actions, "cpu",
                               rename_map or {})
    return services_pb2.PolicySetup(data=pickle.dumps(specs))


def _refusal(stub, setup) -> str:
    import grpc

    from lerobot.transport import services_pb2

    stub.Ready(services_pb2.Empty())
    with pytest.raises(grpc.RpcError) as raised:
        stub.SendPolicyInstructions(setup)
    assert raised.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    return raised.value.details()


def _send_observation(stub, timestep: int, must_go=True, state=0.0):
    import numpy as np

    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.transport import services_pb2
    from lerobot.transport.utils import send_bytes_in_chunks

    raw = {**{j: state + i for i, j in enumerate(JOINTS)}, "task": "hold still",
           "wrist": np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), "scene": np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)}
    obs = TimedObservation(timestamp=time.time(), timestep=timestep, observation=raw, must_go=must_go)
    stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation))


def _accepted_session(tmp_path, checkpoint):
    from lerobot.transport import services_pb2

    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    stub = _stub(server.port)
    stub.Ready(services_pb2.Empty())
    stub.SendPolicyInstructions(_instructions(checkpoint))
    return server, stub


def test_non_loopback_host_exits_2_before_writing(tmp_path, checkpoint):
    proc = subprocess.run(
        [sys.executable, "-m", "research.serve_policy", f"--checkpoint={checkpoint}", "--host=0.0.0.0",
         f"--data-dir={tmp_path / 'data'}", "--fake=hold"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "not a loopback address" in proc.stderr
    assert proc.stdout.splitlines()[-1] == "STATUS failed exit=2"
    assert not (tmp_path / "data").exists()


def test_refused_checkpoint_exits_2(tmp_path):
    ckpt = write_checkpoint(tmp_path / "ckpt", skip=("model.safetensors",))
    proc = subprocess.run(
        [sys.executable, "-m", "research.serve_policy", f"--checkpoint={ckpt}", "--port=0",
         f"--data-dir={tmp_path / 'data'}"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "no weights" in proc.stderr


def test_existing_session_is_never_overwritten(tmp_path, checkpoint, needs_grpc):
    session = tmp_path / "data" / "rollouts" / "serve" / "taken"
    session.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-m", "research.serve_policy", f"--checkpoint={checkpoint}", "--fake=hold", "--port=0",
         "--session=taken", f"--data-dir={tmp_path / 'data'}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "never overwritten" in proc.stderr


def test_server_refuses_every_kind_of_mismatch_over_grpc(tmp_path, checkpoint, needs_grpc):
    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    try:
        stub = _stub(server.port)
        wrong_path = _refusal(stub, _instructions("/some/other/checkpoint"))
        assert "/some/other/checkpoint" in wrong_path and checkpoint in wrong_path

        renamed = client_features()
        renamed["observation.images.front"] = renamed.pop("observation.images.wrist")
        details = _refusal(stub, _instructions(checkpoint, renamed, policy_type="act", actions=51))
        for expected in ("observation.images.wrist is missing", "observation.images.front is not in the checkpoint",
                         "'act'", "actions_per_chunk 51"):
            assert expected in details

        assert "observation.state" in _refusal(stub, _instructions(checkpoint, client_features(joints=JOINTS[:5])))
        assert "rename_map" in _refusal(stub, _instructions(checkpoint, rename_map={"a": "b"}))
        assert server.json("status.json")["counts"]["refusals"] == 4
        assert "rename_map" in server.json("status.json")["last_refusal"]
    finally:
        server.stop()


def test_observations_before_acceptance_are_refused(tmp_path, checkpoint, needs_grpc):
    import grpc

    from lerobot.transport import services_pb2

    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    try:
        stub = _stub(server.port)
        _refusal(stub, _instructions("/some/other/checkpoint"))
        with pytest.raises(grpc.RpcError) as raised:
            _send_observation(stub, 0)
        assert raised.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(grpc.RpcError) as raised:
            stub.GetActions(services_pb2.Empty())
        assert raised.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.stop()


def test_hold_returns_the_observed_joints(tmp_path, checkpoint, needs_grpc):
    from lerobot.transport import services_pb2

    server, stub = _accepted_session(tmp_path, checkpoint)
    try:
        _send_observation(stub, 7, state=12.5)
        reply = stub.GetActions(services_pb2.Empty())
        chunk = pickle.loads(reply.data)
        assert len(chunk) == 50
        assert [a.get_timestep() for a in chunk] == list(range(7, 57))
        expected = [12.5 + i for i in range(6)]
        assert all(a.get_action().tolist() == expected for a in chunk)
    finally:
        server.stop()


def test_empty_poll_is_an_empty_actions_message(tmp_path, checkpoint, needs_grpc):
    # robot_client's receive loop skips it; upstream's Empty cannot be serialized as Actions.
    from lerobot.transport import services_pb2

    server, stub = _accepted_session(tmp_path, checkpoint)
    try:
        assert len(stub.GetActions(services_pb2.Empty()).data) == 0
    finally:
        server.stop()


def test_session_files_record_the_session(tmp_path, checkpoint, needs_grpc):
    from lerobot.transport import services_pb2

    server, stub = _accepted_session(tmp_path, checkpoint)
    try:
        status = server.json("status.json")
        assert (status["status"], status["client_connected"], status["fake"]) == ("serving", True, True)
        _send_observation(stub, 3, state=1.0)
        stub.GetActions(services_pb2.Empty())
        status = server.json("status.json")
        assert status["counts"]["chunks"] == 1 and status["counts"]["observations"] == 1
        assert set(status["last_latency_ms"]) == {"prepare_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
                                                  "total_ms"}
    finally:
        server.stop()
    session = server.json("session.json")
    assert session["checkpoint"] == checkpoint and session["fake"] is True and session["fake_mode"] == "hold"
    assert session["git_commit"] and "git_dirty" in session and session["lerobot_version"]
    assert session["host"] == "127.0.0.1" and session["fps"] == 30
    assert session["start_time"]

    [record] = server.chunks()
    assert (record["fake"], record["timestep"], record["chunk_length"]) == (True, 3, 50)
    assert record["timestamp"] <= record["receive_time"]
    assert record["task"] == "hold still"
    assert len(record["chunk"]) == 50 and record["chunk"][0] == [1.0 + i for i in range(6)]
    for key in ("prepare_ms", "preprocess_ms", "inference_ms", "postprocess_ms"):
        assert key in record

    if os.name != "nt":  # terminate() on Windows is a hard kill, so the shutdown path never runs
        assert server.json("status.json")["status"] == "stopped"
        assert session["end_time"] and session["exit_code"] == 143
    assert server.lines[0].startswith("STATUS loading checkpoint=")
    assert "CLIENT connected" in server.lines
    # lerobot's own log goes inside the session, not into the working directory.
    assert list((server.session / "logs").glob("policy_server_*.log"))


def test_stdout_has_only_protocol_lines(tmp_path, checkpoint, needs_grpc):
    server, _ = _accepted_session(tmp_path, checkpoint)
    server.stop()
    assert all(line.startswith(("STATUS ", "CLIENT ")) for line in server.lines), server.lines


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery; lab's runner stops jobs with SIGTERM")
def test_sigterm_stops_with_143(tmp_path, checkpoint, needs_grpc):
    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    server.proc.send_signal(signal.SIGTERM)
    out, _ = server.proc.communicate(timeout=30)
    assert server.proc.returncode == 143
    assert out.splitlines()[-1] == "STATUS stopped exit=143"
    assert server.json("status.json")["status"] == "stopped"
    assert server.json("session.json")["end_time"]


@pytest.mark.skipif(os.name == "nt", reason="SO_REUSEPORT is the Linux behaviour under test")
def test_second_server_on_the_same_port_is_refused(tmp_path, checkpoint, needs_grpc):
    first = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "research.serve_policy", f"--checkpoint={checkpoint}", "--fake=hold",
             f"--port={first.port}", "--session=second", f"--data-dir={tmp_path / 'data'}"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 2 and "cannot bind" in proc.stderr
    finally:
        first.stop()


def test_fake_server_validates_against_dataset_meta(tmp_path, needs_grpc):
    # The robot PC has no checkpoint, only the string that names it on the GPU host.
    pytest.importorskip("pyarrow")
    root = _write_dataset(tmp_path / "ds")
    name = "/data/mtrea/data/runs/h0/lerobot/checkpoints/last/pretrained_model"
    server = Server(tmp_path, f"--checkpoint={name}", "--fake=hold", f"--dataset-root={root}")
    try:
        stub = _stub(server.port)
        renamed = client_features()
        renamed["observation.images.front"] = renamed.pop("observation.images.wrist")
        assert "observation.images.front" in _refusal(stub, _instructions(name, renamed))
    finally:
        server.stop()


def _run_check(checkpoint: str, port: int, root: Path, episodes: int = 2, *flags: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "research.serve_client_check", f"--server=127.0.0.1:{port}",
         f"--checkpoint={checkpoint}", f"--dataset-root={root}", f"--episodes={episodes}", *flags],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    report = next((json.loads(line[len("REPORT "):]) for line in proc.stdout.splitlines()
                   if line.startswith("REPORT ")), None)
    assert report is not None or proc.returncode != 0, proc.stderr
    return proc.returncode, report or {"stdout": proc.stdout}


def test_check_client_end_to_end_against_hold(tmp_path, checkpoint, needs_grpc):
    root = _write_dataset(tmp_path / "ds")
    dataset = serve_common.DemoDataset(root)
    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    try:
        code, report = _run_check(checkpoint, server.port, root, 1)
    finally:
        server.stop()
    assert code == 0
    assert report["chunks"] >= 1 and report["round_trip_ms"]["n"] == report["chunks"]
    # 2 cameras of raw HWC uint8 plus small change for the rest of the pickle.
    assert report["observation_bytes"] > 2 * HEIGHT * WIDTH * 3
    expected_peak = report["observation_bytes"] * 8 * 30 / 1e6
    assert report["mbit_per_s"]["while_sending_every_tick"] == pytest.approx(expected_peak, abs=0.1)
    assert report["mbit_per_s"]["average"] < expected_peak
    assert "CLIENT connected" in server.lines

    records = server.chunks()
    assert records and all(r["fake"] for r in records)
    for record in records:
        # The state of the frame the client reported at that timestep, repeated.
        state = dataset.state_at(0, min(record["timestep"], FRAMES_PER_EPISODE - 1)).tolist()
        assert all(row == pytest.approx(state, abs=1e-3) for row in record["chunk"])


def test_check_client_end_to_end_against_replay(tmp_path, checkpoint, needs_grpc):
    root = _write_dataset(tmp_path / "ds")
    dataset = serve_common.DemoDataset(root)
    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=replay", f"--dataset-root={root}")
    try:
        code, report = _run_check(checkpoint, server.port, root, 2)
    finally:
        server.stop()
    assert code == 0
    # Recorded actions of each client's own episode, in order: nothing differs from the dataset.
    for errors in report["vs_dataset"].values():
        assert errors["mean_mae"] == pytest.approx(0.0, abs=1e-3)
    assert [e["episode"] for e in report["episodes"]] == [0, 1]
    records = server.chunks()
    assert all(r["chunk_length"] == 50 for r in records)
    first = records[0]
    recorded = dataset.actions_from(0, first["timestep"], 50).tolist()
    for row, action in zip(first["chunk"], recorded):
        assert row == pytest.approx(action, abs=1e-3)
    # Past the episode's end it holds the last action rather than jump to the next episode.
    assert first["chunk"][-1] == pytest.approx(recorded[-1], abs=1e-3)


def test_check_client_reports_a_refusal(tmp_path, checkpoint, needs_grpc):
    root = _write_dataset(tmp_path / "ds")
    server = Server(tmp_path, f"--checkpoint={checkpoint}", "--fake=hold")
    try:
        code, report = _run_check("/some/other/checkpoint", server.port, root, 1)
    finally:
        server.stop()
    assert code == 1
    assert "REFUSED" in report["stdout"] and "/some/other/checkpoint" in report["stdout"]


def test_replay_needs_its_dataset(tmp_path, checkpoint):
    proc = subprocess.run(
        [sys.executable, "-m", "research.serve_policy", f"--checkpoint={checkpoint}", "--fake=replay",
         f"--data-dir={tmp_path / 'data'}"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "--dataset-root" in proc.stderr


def test_load_policy_refuses_a_silently_random_model(monkeypatch, tmp_path):
    """PI05Policy.from_pretrained prints a warning and returns random weights on any load failure."""
    with serve_common.lerobot_log_dir(tmp_path / "logs"):
        pytest.importorskip("lerobot.async_inference.policy_server")
        from research import serve_backend

    class Config:
        type = "pi05"
        device = "cpu"

    class Policy:
        config = Config()

        @classmethod
        def from_pretrained(cls, path, config):
            print("Warning: Could not load state dict: nope")
            return cls()

    import lerobot.configs.policies as policies

    monkeypatch.setattr(policies.PreTrainedConfig, "from_pretrained", lambda path: Config())
    monkeypatch.setattr(serve_backend, "get_policy_class", lambda name: Policy)
    with pytest.raises(serve_backend.WeightsNotLoaded, match="randomly initialized"):
        serve_backend.load_policy(tmp_path, "cpu")
