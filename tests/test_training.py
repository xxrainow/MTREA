"""lab.core.training: resolving a Train page request, checking it, and reading run files.

No network. The GPU host is LocalTransport over a temp directory, or a
scripted transport where the remote shell itself matters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lab.core import dataset_sync, paths, training
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.remote import LocalTransport, Remote, RemoteConfig

QUANTILES = {f: {"mean": [0.0], "q01": [-1.0], "q99": [1.0]} for f in ("observation.state", "action")}


@pytest.fixture(scope="module")
def spec() -> dict:
    return training.describe()


def write_dataset(root: Path, fake: bool = False) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1, "fps": 30, **({"fake": True} if fake else {})}))
    (root / "meta" / "stats.json").write_text(json.dumps(QUANTILES))
    (root / "data").mkdir()
    (root / "data" / "episode_000000.parquet").write_bytes(b"x" * 1000)
    return root


@pytest.fixture
def base(tmp_path) -> Path:
    b = tmp_path / "data"
    paths.ensure_dirs(b)
    write_dataset(paths.raw_dir(b) / "pick_v1")
    write_dataset(b / "fake" / "collection" / "raw" / "demo", fake=True)
    return b


@pytest.fixture
def remote(tmp_path) -> Remote:
    config = RemoteConfig(host="gpu", user="lab", repo_dir=str(tmp_path / "repo"),
                          data_dir=(tmp_path / "gpu").as_posix(), python=Path(sys.executable).as_posix())
    return Remote(config, LocalTransport())


def request(**overrides) -> training.TrainRequest:
    values = {"target": "ssh", "dataset_backend": "real", "dataset_name": "pick_v1", "run_name": "h0_pick_v1_s0",
              "fake": False, "params": {"method": "h0", "seed": 0}}
    return training.TrainRequest(**{**values, **overrides})


def check_ids(checks, status):
    return [c["id"] for c in checks if c["status"] == status]


# --- resolve ---

def test_ssh_unsynced_dataset_is_a_failing_check(spec, base, remote):
    res = training.resolve(request(), spec, remote, base)
    assert check_ids(res.checks, "fail") == ["dataset_sync"]
    (sync,) = [c for c in res.checks if c["id"] == "dataset_sync"]
    assert sync["message"].startswith("Sync this dataset to the GPU host first") and sync["link"] == "datasets"


def test_ssh_synced_dataset_uses_its_remote_path(spec, base, remote):
    assert dataset_sync.sync("pick_v1", remote, base) == 0
    res = training.resolve(request(params={"method": "h0", "seed": 3, "steps": 200}), spec, remote, base)
    assert check_ids(res.checks, "fail") == []
    assert res.run_dir == f"{remote.config.data_dir}/runs/h0_pick_v1_s0"
    assert res.host == "lab@gpu"
    assert res.args == ["--method=h0", "--dataset=local/pick_v1",
                        f"--dataset-root={remote.remote_dataset_root('pick_v1')}", "--seed=3",
                        f"--output={res.run_dir}", "--steps=200"]


def test_ssh_refuses_fake_datasets_and_needs_a_configured_host(spec, base, remote):
    res = training.resolve(request(dataset_backend="fake", dataset_name="demo"), spec, remote, base)
    assert check_ids(res.checks, "fail") == ["dataset_sync"]
    res = training.resolve(request(), spec, None, base, not_configured="remote.yaml not found")
    assert check_ids(res.checks, "fail") == ["remote"] and "remote.yaml not found" in res.checks[0]["message"]


def test_local_real_is_refused(spec, base):
    res = training.resolve(request(target="local"), spec, None, base)
    assert check_ids(res.checks, "fail") == ["target"]
    assert "no GPU" in res.checks[0]["message"]


def test_local_fake_runs_under_data_fake_runs(spec, base):
    res = training.resolve(request(target="local", fake=True, dataset_backend="fake", dataset_name="demo",
                                   params={"method": "h0", "seed": 0, "fake_seconds_per_step": 0.05}),
                           spec, None, base)
    assert res.checks == []
    assert Path(res.run_dir) == paths.fake_runs_dir(base) / "h0_pick_v1_s0"
    assert f"--dataset-root={base / 'fake' / 'collection' / 'raw' / 'demo'}" in res.args
    assert res.args[-2:] == ["--fake-seconds-per-step=0.05", "--fake"]


def test_only_user_set_values_and_settable_params_pass(spec, base):
    res = training.resolve(request(target="local", fake=True), spec, None, base)
    assert [a.split("=")[0] for a in res.args] == ["--method", "--dataset", "--dataset-root", "--seed", "--output", "--fake"]
    for params in ({"output": "/elsewhere"}, {"dataset_root": "/x"}, {"base": "other/model"}, {"nope": 1},
                   {"fake_fail": "oom"}):
        with pytest.raises(training.TrainError):
            training.resolve(request(params={"method": "h0", "seed": 0, **params}), spec, None, base)
    for params in ({"seed": "0"}, {"seed": True}, {"steps": 0}, {"lr": float("inf")}, {"method": "h9"},
                   {"keep_checkpoints": [1, "2"]}, {"wandb": "yes"}):
        with pytest.raises(training.TrainError):
            training.resolve(request(params={"method": "h0", "seed": 0, **params}), spec, None, base)
    with pytest.raises(training.TrainError):
        training.resolve(request(run_name="../escape"), spec, None, base)
    missing = training.resolve(request(target="local", fake=True, params={"method": "h0"}), spec, None, base)
    assert check_ids(missing.checks, "fail") == ["params"]


# --- check ---

class ScriptedHost(LocalTransport):
    """The GPU host's filesystem is a temp dir; its shell answers are scripted."""

    def __init__(self, check_output: dict, head: str):
        super().__init__()
        self.check_output, self.head = check_output, head
        self.commands: list[str] = []

    def run(self, command, *, input=None, timeout=60):
        self.commands.append(command)
        if "rev-parse HEAD" in command:
            return subprocess.CompletedProcess(command, 0, (self.head + "\n").encode(), b"")
        if "scripts/train.sh" in command:
            return subprocess.CompletedProcess(command, 0, json.dumps(self.check_output).encode(), b"")
        raise AssertionError(command)


def test_ssh_check_runs_train_check_on_the_host_and_warns_on_commit(spec, base, remote, tmp_path):
    assert dataset_sync.sync("pick_v1", remote, base) == 0
    explicit = "scripts/train.sh --method=h0 --dataset=local/pick_v1 --seed=0 --steps=10000"
    host = ScriptedHost({"ok": True, "checks": [{"id": "output", "status": "ok", "message": "free"}],
                         "estimated_seconds": 39200, "command": explicit}, head="0" * 40)
    scripted = Remote(remote.config, host)
    result = training.check(request(), spec, scripted, base)
    assert result["ok"] is True  # a commit mismatch warns, it does not block
    assert result["command"] == f"bash {explicit}"
    assert result["estimated_seconds"] == 39200 and result["host"] == "lab@gpu"
    commit = next(c for c in result["checks"] if c["id"] == "remote_commit")
    assert commit["status"] == "warn" and "git pull on the GPU host" in commit["message"]
    (train_check,) = [c for c in host.commands if "scripts/train.sh" in c]
    assert train_check.startswith(remote.remote_prefix() + " bash scripts/train.sh ")
    assert train_check.endswith(" --check")


def test_local_check_runs_train_check_without_bash(spec, base, tmp_path):
    result = training.check(request(target="local", fake=True), spec, None, base)
    assert result["ok"] is True, result["checks"]
    assert {c["id"] for c in result["checks"]} >= {"output", "warmup", "stats", "lerobot"}
    quote = subprocess.list2cmdline if os.name == "nt" else None
    assert result["command"].startswith((quote([sys.executable]) if quote else sys.executable) + " -m research.train ")
    assert "--seed=0" in result["command"] and "--steps=10000" in result["command"]  # every option explicit


def test_failed_lab_check_skips_the_target(spec, base, remote):
    result = training.check(request(), spec, remote, base)
    assert result["ok"] is False and result["command"] is None
    assert check_ids(result["checks"], "fail") == ["dataset_sync"]


# --- run files ---

def train_job(run_dir: str, status=JobStatus.RUNNING, executor=Executor.LOCAL) -> Job:
    return Job(id="j1", spec=JobSpec(kind=JobKind.TRAIN, cmd="x", executor=executor, run_dir=run_dir), status=status)


def test_metrics_page_by_offset_and_skip_a_partial_line(tmp_path):
    run = training.LocalRunDir(str(tmp_path))
    assert training.read_metrics(run, 0) == {"metrics": [], "offset": 0}
    lines = [json.dumps({"step": s, "loss": 1.0 / s, "lr": 1e-5}) + "\n" for s in (10, 20)]
    (tmp_path / "metrics.jsonl").write_bytes(("".join(lines) + '{"step": 30, "lo').encode())
    first = training.read_metrics(run, 0)
    assert [m["step"] for m in first["metrics"]] == [10, 20]
    assert first["offset"] == len("".join(lines).encode())
    with (tmp_path / "metrics.jsonl").open("ab") as f:
        f.write(b'ss": 0.03, "lr": 1e-5}\r\n')  # train.py on Windows writes CRLF
    second = training.read_metrics(run, first["offset"])
    assert [m["step"] for m in second["metrics"]] == [30]
    assert training.read_metrics(run, second["offset"])["metrics"] == []


def test_run_files_are_confined_to_the_run_dir(tmp_path):
    run = training.LocalRunDir(str(tmp_path))
    with pytest.raises(ValueError):
        run.read_text("../../etc/passwd")
    assert training.read_json(run, "result.json") is None


def test_run_state_corrects_a_stale_running_status():
    running = {"status": "running"}
    assert training.run_state(train_job("/r"), running) == "running"
    assert training.run_state(train_job("/r", JobStatus.UNKNOWN), running) == "ended_unknown"
    assert training.run_state(train_job("/r", JobStatus.FAILED), running) == "ended_unknown"
    # Windows stops without a signal train.py can catch; the runner knows it stopped the job.
    assert training.run_state(train_job("/r", JobStatus.STOPPED), running) == "stopped"
    assert training.run_state(train_job("/r", JobStatus.DONE), {"status": "done"}) == "done"
    assert training.run_state(train_job("/r"), None) == "starting"
    assert training.run_state(train_job("/r", JobStatus.FAILED), None) == "failed"  # refused before launch


class Counting(LocalTransport):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def read_texts(self, paths):
        self.calls += 1
        return super().read_texts(paths)


def test_list_runs_reads_all_ssh_runs_in_one_remote_call(tmp_path, remote):
    transport = Counting()
    remote = Remote(remote.config, transport)
    jobs = []
    for i in range(3):
        run_dir = tmp_path / "gpu" / "runs" / f"r{i}"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({"status": "done", "step": 10}))
        (run_dir / "checkpoints.json").write_text(json.dumps({"checkpoints": [{"step": 10}] * (i + 1)}))
        job = train_job(run_dir.as_posix(), JobStatus.DONE, Executor.SSH)
        job.id = f"j{i}"
        jobs.append(job)
    jobs.append(Job(id="other", spec=JobSpec(kind=JobKind.SYNC, cmd="x")))
    runs = training.list_runs(jobs, remote)
    assert transport.calls == 1
    assert [(r["job"]["id"], r["state"], r["checkpoint_count"]) for r in runs] == [
        ("j0", "done", 1), ("j1", "done", 2), ("j2", "done", 3)]
    unconfigured = training.list_runs(jobs, None)
    assert {r["unreachable"] for r in unconfigured} == {"GPU host not configured"}
