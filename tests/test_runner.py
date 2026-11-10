"""Runner + LocalExecutor + JobStore against throwaway shell commands.

No robot, no GPU, no research scripts: short python processes stand in for
train.sh, quoted for the platform shell (cmd.exe on Windows, sh elsewhere). What is checked is the contract the UI will rely on — submit
returns a running job, refresh notices completion, logs are captured,
stop kills the process, exclusive kinds do not overlap, and the ledger
survives a "server restart" (a fresh Runner over the same file).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
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


def py(code: str) -> str:
    argv = [sys.executable, "-c", code]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


SLEEP = py("import time; time.sleep(30)")


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
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd=py("import sys; print('hello'); print('world', file=sys.stderr)")))
    assert job.status == JobStatus.RUNNING
    assert job.pid is not None
    assert _wait(runner, job.id) == JobStatus.DONE
    done = runner.get(job.id)
    assert done is not None and done.exit_code == 0
    log = runner.read_log(job.id)
    assert "hello" in log and "world" in log  # stderr merged into the same file


def test_failure_is_recorded(runner: Runner) -> None:
    job = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd=py("print('boom'); raise SystemExit(3)")))
    assert _wait(runner, job.id) == JobStatus.FAILED
    assert runner.get(job.id).exit_code == 3


def test_stop_kills_running_job(runner: Runner) -> None:
    job = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd=SLEEP))
    time.sleep(0.2)
    assert runner.get(job.id).status == JobStatus.RUNNING
    stopped = runner.stop(job.id)
    assert stopped.status == JobStatus.STOPPED
    time.sleep(0.2)
    assert not runner.executors[Executor.LOCAL].is_running(job.pid)


def test_exclusive_kinds_do_not_overlap(runner: Runner) -> None:
    first = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd=SLEEP, label="first"))
    with pytest.raises(JobConflict):
        runner.submit(JobSpec(kind=JobKind.TRAIN, cmd=SLEEP, label="second"))
    with pytest.raises(JobConflict):
        runner.submit(JobSpec(kind=JobKind.NORM_STATS, cmd=SLEEP))
    # a different, non-conflicting kind is fine
    other = runner.submit(JobSpec(kind=JobKind.CONVERT, cmd="echo ok"))
    runner.stop(first.id)
    _wait(runner, other.id)
    # once the train job is gone, a new one may start
    again = runner.submit(JobSpec(kind=JobKind.TRAIN, cmd="echo ok"))
    _wait(runner, again.id)

def test_rollout_jobs_do_not_overlap(runner: Runner) -> None:
    first = runner.submit(
        JobSpec(kind=JobKind.ROLLOUT, cmd=SLEEP)
    )

    try:
        with pytest.raises(JobConflict):
            runner.submit(
                JobSpec(kind=JobKind.ROLLOUT, cmd=py("pass"))
            )
    finally:
        # These dummy commands have no robot command channel; test cleanup only.
        runner.executors[Executor.LOCAL].stop(first.pid)
        runner.refresh()

    again = runner.submit(
        JobSpec(kind=JobKind.ROLLOUT, cmd=py("pass"))
    )
    assert _wait(runner, again.id) == JobStatus.DONE


def test_rollout_executes_argv_without_shell_interpreting_instruction(runner: Runner):
    instruction = 'Hold "block"; $(false) & | < >'
    job = runner.submit(JobSpec(
        kind=JobKind.ROLLOUT, cmd="must never execute this display command",
        metadata={"argv": [sys.executable, "-c", "import sys; print(sys.argv[1])", instruction]},
    ))
    assert _wait(runner, job.id) == JobStatus.DONE
    assert runner.read_log(job.id).strip() == instruction

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


@pytest.mark.parametrize(
    "first_kind, second_kind",
    [
        (JobKind.TRAIN, JobKind.SERVE_POLICY),
        (JobKind.SERVE_POLICY, JobKind.TRAIN),
        (JobKind.NORM_STATS, JobKind.SERVE_POLICY),
        (JobKind.SERVE_POLICY, JobKind.NORM_STATS),
    ],
)
def test_serving_conflicts_with_other_gpu_jobs(
    runner, first_kind, second_kind,
):
    first = runner.submit(
        JobSpec(kind=first_kind, cmd=SLEEP)
    )

    try:
        with pytest.raises(JobConflict):
            runner.submit(
                JobSpec(kind=second_kind, cmd=py("pass"))
            )
    finally:
        runner.stop(first.id)
        
def test_stopping_policy_server_also_stops_its_tunnel(runner):
    server = runner.submit(
        JobSpec(kind=JobKind.SERVE_POLICY, cmd=SLEEP)
    )
    tunnel = None

    try:
        tunnel = runner.submit(
            JobSpec(
                kind=JobKind.POLICY_TUNNEL,
                cmd=SLEEP,
                metadata={"server_job_id": server.id},
            )
        )

        runner.stop(server.id)

        saved_tunnel = runner.store.get(tunnel.id)

        assert saved_tunnel is not None
        assert saved_tunnel.status == JobStatus.STOPPED
        assert not runner.executors[Executor.LOCAL].is_running(tunnel.pid)

    finally:
        if tunnel is not None:
            runner.stop(tunnel.id)
        runner.stop(server.id)

def test_refresh_stops_tunnel_after_policy_server_exits(runner):
    server = runner.submit(
        JobSpec(kind=JobKind.SERVE_POLICY, cmd=SLEEP)
    )
    tunnel = None

    try:
        tunnel = runner.submit(
            JobSpec(
                kind=JobKind.POLICY_TUNNEL,
                cmd=SLEEP,
                metadata={"server_job_id": server.id},
            )
        )

        executor = runner.executors[Executor.LOCAL]
        executor.stop(server.pid)

        before = runner.store.get(tunnel.id)
        assert before is not None
        assert before.status == JobStatus.RUNNING

        runner.refresh()

        saved_server = runner.store.get(server.id)
        saved_tunnel = runner.store.get(tunnel.id)

        assert saved_server is not None
        assert not saved_server.is_active()
        assert saved_tunnel is not None
        assert saved_tunnel.status == JobStatus.STOPPED
        assert not executor.is_running(tunnel.pid)

    finally:
        if tunnel is not None:
            runner.stop(tunnel.id)
        runner.stop(server.id)
