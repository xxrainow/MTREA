"""Runner + LocalExecutor + JobStore against throwaway shell commands.

No robot, no GPU, no research scripts: `sleep` and `echo` stand in for
train.sh. What is checked is the contract the UI will rely on — submit
returns a running job, refresh notices completion, logs are captured,
stop kills the process, exclusive kinds do not overlap, and the ledger
survives a "server restart" (a fresh Runner over the same file).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor, JobKind, JobSpec, JobStatus
from lab.core.runner import JobConflict, Runner
from lab.core.store import JobStore


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    paths.ensure_dirs(tmp_path)
    store = JobStore(paths.jobs_ledger(tmp_path))
    return Runner(
        store=store,
        executors={Executor.LOCAL: LocalExecutor(repo_root=tmp_path)},
        data_base=tmp_path,
        repo_root=tmp_path,
    )


def _wait(runner: Runner, job_id: str, timeout_s: float = 5.0) -> JobStatus:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        assert job is not None
        if not job.is_active():
            return job.status
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_submit_runs_and_captures_log(runner: Runner) -> None:
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo hello; echo world >&2"))
    assert job.status == JobStatus.RUNNING
    assert job.pid is not None
    assert _wait(runner, job.id) == JobStatus.DONE
    done = runner.get(job.id)
    assert done is not None and done.exit_code == 0
    log = runner.read_log(job.id)
    assert "hello" in log and "world" in log  # stderr merged into the same file


def test_failure_is_recorded(runner: Runner) -> None:
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo boom; exit 3"))
    assert _wait(runner, job.id) == JobStatus.FAILED
    assert runner.get(job.id).exit_code == 3


def test_stop_kills_running_job(runner: Runner) -> None:
    job = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="sleep 30"))
    time.sleep(0.2)
    assert runner.get(job.id).status == JobStatus.RUNNING
    stopped = runner.stop(job.id)
    assert stopped.status == JobStatus.STOPPED
    time.sleep(0.2)
    assert not runner.executors[Executor.LOCAL].is_running(job.pid)


def test_exclusive_kinds_do_not_overlap(runner: Runner) -> None:
    first = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="sleep 30", label="first"))
    with pytest.raises(JobConflict):
        runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="sleep 30", label="second"))
    with pytest.raises(JobConflict):
        runner.submit(JobSpec(kind=JobKind.NORM_STATS, cmd="sleep 30"))
    # a different, non-conflicting kind is fine
    other = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo ok"))
    runner.stop(first.id)
    _wait(runner, other.id)
    # once the train job is gone, a new one may start
    again = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="echo ok"))
    _wait(runner, again.id)


def test_ledger_survives_restart(runner: Runner, tmp_path: Path) -> None:
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo persisted"))
    _wait(runner, job.id)

    # A "new server process": fresh store + runner over the same ledger file.
    store2 = JobStore(paths.jobs_ledger(tmp_path))
    runner2 = Runner(store2, {Executor.LOCAL: LocalExecutor(tmp_path)}, data_base=tmp_path, repo_root=tmp_path)
    seen = runner2.get(job.id)
    assert seen is not None
    assert seen.status == JobStatus.DONE
    assert seen.spec.cmd == "echo persisted"
    assert "persisted" in runner2.read_log(job.id)


def test_ledger_is_append_only(runner: Runner, tmp_path: Path) -> None:
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo x"))
    _wait(runner, job.id)
    lines = paths.jobs_ledger(tmp_path).read_text().strip().splitlines()
    assert len(lines) >= 2  # RUNNING line, then DONE line — history kept
