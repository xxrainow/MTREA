"""SshExecutor, the runner's handling of an unreachable host, and the job ledger fields it adds.

No network. A scripted transport checks the remote commands on every OS.
On POSIX, LocalTransport runs the same shell locally: setsid, nohup, the exit
file and group kill are exercised for real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.executors import local as local_executor
from lab.core.executors.ssh import SshExecutor
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.remote import LocalTransport, Remote, RemoteConfig, RemoteError, Transport
from lab.core.runner import Runner
from lab.core.store import JobStore

posix_only = pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell with setsid on this machine")


def make_config(tmp_path: Path) -> RemoteConfig:
    return RemoteConfig(host="gpu", user="lab", repo_dir=str(tmp_path / "repo"), data_dir=(tmp_path / "gpu").as_posix(),
                        python=Path(sys.executable).as_posix())


class Scripted(Transport):
    """Answers run() from a list of (returncode, stdout) and records the commands."""

    def __init__(self, answers=None, fail=False):
        self.answers = list(answers or [])
        self.commands: list[str] = []
        self.fail = fail

    def run(self, command, *, input=None, timeout=60):
        self.commands.append(command)
        if self.fail:
            raise RemoteError("no answer from lab@gpu within 10 s")
        rc, out = self.answers.pop(0) if self.answers else (0, "")
        return subprocess.CompletedProcess(command, rc, out.encode(), b"")

    # The rest of Transport is not used by the executor.
    ping = read_text = read_from = read_texts = write_text = makedirs = rename = None
    probe = list_manifests = manifest = transfer = terminate = None


def make_runner(tmp_path: Path, transport: Transport) -> Runner:
    paths.ensure_dirs(tmp_path / "data")
    remote = Remote(make_config(tmp_path), transport)
    return Runner(JobStore(paths.jobs_ledger(tmp_path / "data")),
                  {Executor.LOCAL: LocalExecutor(tmp_path), Executor.SSH: SshExecutor(remote)},
                  data_base=tmp_path / "data", repo_root=tmp_path)


# --- command shape (every OS) ---

def test_start_detaches_with_setsid_and_writes_an_exit_file(tmp_path):
    transport = Scripted([(0, "4242\n")])
    executor = SshExecutor(Remote(make_config(tmp_path), transport))
    log_path = tmp_path / "logs" / "20261031-120000-train-abc123.log"
    pid = executor.start(JobSpec(kind=JobKind.TRAIN, cmd="bash scripts/train.sh --seed=0"), log_path)
    assert pid == 4242
    assert log_path.exists()  # local copy, so the Jobs page has a file to show
    (command,) = transport.commands
    logs = (tmp_path / "gpu").as_posix() + "/logs"
    assert command.startswith(f"mkdir -p {logs} || exit 1; setsid nohup bash -c ")
    # `a && b & echo $!` would report a subshell's PID instead of the job's.
    assert "&& setsid" not in command
    assert command.endswith(f"> {logs}/20261031-120000-train-abc123.log 2>&1 < /dev/null & echo $!")
    # The job command runs after remote_prefix(), in repo_dir with the venv on PATH.
    assert "export PATH=" in command and "bash scripts/train.sh --seed=0; echo $?" in command
    assert "20261031-120000-train-abc123.exit.tmp" in command


def test_job_ids_are_validated(tmp_path):
    executor = SshExecutor(Remote(make_config(tmp_path), Scripted([(0, "1\n")])))
    with pytest.raises(ValueError):
        executor.start(JobSpec(kind=JobKind.TRAIN, cmd="true"), tmp_path / "x; rm -rf ~.log")
    with pytest.raises(ValueError):
        executor.is_running(1, job_id=None)


def test_exit_file_and_lost_process(tmp_path):
    executor = SshExecutor(Remote(make_config(tmp_path), Scripted([
        (0, "running\n"), (0, "exited\n3\n"), (0, "exited\n3\n"), (0, "gone\n"), (0, "gone\n")])))
    assert executor.is_running(10, job_id="j1")
    assert not executor.is_running(10, job_id="j1")
    assert executor.exit_code(10, job_id="j1") == 3
    assert not executor.is_running(10, job_id="j1")
    assert executor.exit_code(10, job_id="j1") is None  # container restarted: no exit file


def test_stop_signals_the_process_group(tmp_path):
    transport = Scripted([(0, "")])
    SshExecutor(Remote(make_config(tmp_path), transport)).stop(321, grace_s=2, job_id="j1")
    (command,) = transport.commands
    assert command.startswith("kill -TERM -- -321 ")
    assert "kill -0 -- -321" in command and "kill -KILL -- -321" in command


def test_unreachable_leaves_status_unchanged(tmp_path):
    transport = Scripted([(0, "777\n")])
    runner = make_runner(tmp_path, transport)
    job = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="bash scripts/train.sh", executor=Executor.SSH))
    lines_before = paths.jobs_ledger(tmp_path / "data").read_text().splitlines()
    transport.fail = True
    (seen,) = [j for j in runner.refresh() if j.id == job.id]
    assert seen.status == JobStatus.RUNNING
    assert "no answer" in seen.unreachable
    assert paths.jobs_ledger(tmp_path / "data").read_text().splitlines() == lines_before
    seen = runner.get(job.id)  # what /api/train/runs/<id>/status reports
    assert seen.status == JobStatus.RUNNING and "no answer" in seen.unreachable
    with pytest.raises(RemoteError):
        runner.stop(job.id)
    assert runner.store.get(job.id).status == JobStatus.RUNNING


def test_lost_process_becomes_unknown(tmp_path):
    transport = Scripted([(0, "778\n"), (0, "gone\n"), (0, "gone\n")])
    runner = make_runner(tmp_path, transport)
    job = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="bash scripts/train.sh", executor=Executor.SSH))
    assert runner.get(job.id).status == JobStatus.UNKNOWN


def test_read_log_keeps_a_local_copy_and_falls_back_to_it(tmp_path):
    transport = Scripted([(0, "777\n"), (0, "STATUS started\nPROGRESS step=10/60\n")])
    runner = make_runner(tmp_path, transport)
    job = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="bash scripts/train.sh", executor=Executor.SSH))
    assert runner.read_log(job.id, tail_lines=50) == "STATUS started\nPROGRESS step=10/60"
    assert transport.commands[-1].endswith(f"then tail -n 50 {(tmp_path / 'gpu').as_posix()}/logs/{job.id}.log; fi")
    transport.fail = True
    assert runner.read_log(job.id, tail_lines=1) == "PROGRESS step=10/60"


# --- the job ledger ---

def test_old_ledger_lines_without_run_dir_or_host_load(tmp_path):
    ledger = paths.jobs_ledger(tmp_path)
    ledger.parent.mkdir(parents=True)
    old = {"id": "old-1", "status": "done", "pid": 1, "exit_code": 0, "submitted_at": "2026-10-01T10:00:00",
           "started_at": None, "finished_at": None, "log_path": None, "git_commit": "abc",
           "spec": {"kind": "train", "cmd": "scripts/train.sh", "executor": "local", "label": "x", "cwd": None}}
    ledger.write_text(json.dumps(old) + "\n", encoding="utf-8")
    job = JobStore(ledger).get("old-1")
    assert job.spec.run_dir is None and job.spec.host is None and job.unreachable is None
    job.spec = JobSpec(kind=JobKind.TRAIN, cmd="x", run_dir="/data/runs/r", host="lab@gpu")
    job.unreachable = "transient"
    again = Job.from_dict(json.loads(json.dumps(job.to_dict())))
    assert (again.spec.run_dir, again.spec.host, again.unreachable) == ("/data/runs/r", "lab@gpu", None)


# --- LocalExecutor's Windows branch (logic only; real runs are covered by test_runner on Windows) ---

def test_windows_start_uses_a_new_process_group(tmp_path, monkeypatch):
    seen = {}

    class FakePopen:
        pid = 99

        def __init__(self, cmd, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(local_executor.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(local_executor.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(local_executor.os, "name", "nt")
    LocalExecutor(tmp_path).start(JobSpec(kind=JobKind.CONVERT, cmd="x"), tmp_path / "j.log")
    assert seen["creationflags"] == 0x200 and "start_new_session" not in seen


def test_windows_stop_asks_then_forces_the_tree(tmp_path, monkeypatch):
    calls = []
    alive = iter([True, True, True])
    monkeypatch.setattr(local_executor.subprocess, "run", lambda argv, **kw: calls.append(argv))
    executor = LocalExecutor(tmp_path)
    monkeypatch.setattr(executor, "is_running", lambda pid, job_id=None: next(alive, True))
    executor._stop_windows(55, grace_s=0.2)
    assert calls == [["taskkill", "/T", "/PID", "55"], ["taskkill", "/T", "/F", "/PID", "55"]]


# --- the real shell (POSIX) ---

def _wait(runner: Runner, job_id: str, timeout_s: float = 10.0) -> Job:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if not job.is_active():
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def posix_runner(tmp_path):
    (tmp_path / "repo").mkdir()
    return make_runner(tmp_path, LocalTransport())


@posix_only
def test_real_start_runs_to_done_and_failed(posix_runner, tmp_path):
    done = posix_runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="pwd; echo hello", executor=Executor.SSH))
    job = _wait(posix_runner, done.id)
    assert (job.status, job.exit_code) == (JobStatus.DONE, 0)
    log = posix_runner.read_log(done.id)
    assert "hello" in log and str(tmp_path / "repo") in log  # remote_prefix() ran first
    failed = posix_runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo boom; exit 3", executor=Executor.SSH))
    job = _wait(posix_runner, failed.id)
    assert (job.status, job.exit_code) == (JobStatus.FAILED, 3)


@posix_only
def test_real_stop_kills_the_whole_group(posix_runner):
    job = posix_runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="sleep 60 & sleep 60; wait", executor=Executor.SSH))
    time.sleep(0.5)
    assert posix_runner.get(job.id).status == JobStatus.RUNNING
    assert os.getpgid(job.pid) == job.pid  # setsid: the PID is the group id
    stopped = posix_runner.stop(job.id)
    assert stopped.status == JobStatus.STOPPED
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.killpg(job.pid, 0)


@posix_only
def test_real_lost_process_without_exit_file_is_unknown(posix_runner):
    import signal
    job = posix_runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="sleep 60", executor=Executor.SSH))
    time.sleep(0.3)
    os.killpg(job.pid, signal.SIGKILL)  # what a container restart leaves behind
    assert _wait(posix_runner, job.id).status == JobStatus.UNKNOWN
