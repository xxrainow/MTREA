"""Submit, watch and stop long-running jobs (train, norm_stats, serve_policy, ...).

The runner never imports research/. It only builds a command line, hands it to
an executor, and records what happened. A person can paste the same command
into a terminal and get the same run — that is the reproducibility contract.
"""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from lab.core import paths
from lab.core.executors.base import BaseExecutor, read_local_log
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.remote import RemoteError
from lab.core.store import JobStore


class JobConflict(RuntimeError):
    """A job of this kind is already active (one GPU, one policy server)."""


# Kinds that must not overlap with themselves. Train and norm_stats also
# compete for the single GPU, so they exclude each other. One dataset sync at
# a time keeps the GPU host's .incoming/ free of concurrent writers.
_EXCLUSIVE: dict[JobKind, set[JobKind]] = {
    JobKind.TRAIN: {JobKind.TRAIN, JobKind.NORM_STATS},
    JobKind.NORM_STATS: {JobKind.TRAIN, JobKind.NORM_STATS},
    JobKind.SERVE_POLICY: {JobKind.SERVE_POLICY},
    JobKind.SYNC: {JobKind.SYNC},
}


class Runner:
    def __init__(
        self,
        store: JobStore,
        executors: dict[Executor, BaseExecutor],
        data_base: Path | None = None,
        repo_root: Path | None = None,
    ):
        self.store = store
        self.executors = executors
        self.data_base = data_base
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    # --- submit ---

    def submit(self, spec: JobSpec) -> Job:
        self.refresh()
        blocked_by = _EXCLUSIVE.get(spec.kind, set())
        for other in self.active():
            if other.spec.kind in blocked_by:
                raise JobConflict(f"{other.spec.kind.value} job {other.id} is still {other.status.value}")

        job = Job(id=self._new_id(spec), spec=spec, git_commit=self._git_commit())
        job.log_path = str(paths.job_log(job.id, self.data_base))
        executor = self.executors[spec.executor]
        job.pid = executor.start(spec, Path(job.log_path))
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.store.save(job)
        return job

    # --- watch ---

    def refresh(self) -> list[Job]:
        """Poll every active job; persist any state change. Returns all jobs."""
        jobs = self.store.list()
        for job in jobs:
            # Do not change the ledger if the termination result is unknown
            if job.status not in (JobStatus.RUNNING, JobStatus.UNKNOWN):
                continue
            if job.pid is None:               
                continue
            
            # If it is running and has a PID, check whether it is actually running
            executor = self.executors.get(job.spec.executor)
            if executor is None:
                continue
            try:
                if executor.is_running(job.pid, job_id=job.id):
                    continue
                job.exit_code = executor.exit_code(job.pid, job_id=job.id)
            except RemoteError as e:
                # The GPU host did not answer: no evidence either way, so the
                # status stays as recorded. Shown to the UI, never persisted.
                job.unreachable = str(e)
                continue
            
            if job.exit_code is None:
                # Confirmed is_running = False; but don't know the exit_code
                if job.status == JobStatus.UNKNOWN:
                    continue

                job.status = JobStatus.UNKNOWN
            elif job.exit_code == 0:
                job.status = JobStatus.DONE
            else:
                job.status = JobStatus.FAILED

            job.finished_at = datetime.now()
            self.store.save(job)
        return jobs

    def active(self) -> list[Job]:
        return [j for j in self.store.list() if j.is_active()]

    def get(self, job_id: str) -> Job | None:
        # The refreshed object, not a reload: it carries `unreachable`.
        return next((job for job in self.refresh() if job.id == job_id), None)

    # --- stop ---

    def stop(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == JobStatus.RUNNING and job.pid is not None:
            self.executors[job.spec.executor].stop(job.pid, job_id=job.id) # stop process by executor
            job.status = JobStatus.STOPPED
            job.finished_at = datetime.now()
            self.store.save(job) # Record the changed job status in the jobs.jsonl file
        return job

    # --- helpers ---

    def read_log(self, job_id: str, tail_lines: int | None = None) -> str:
        job = self.store.get(job_id)
        if job is None or not job.log_path:
            return ""
        executor = self.executors.get(job.spec.executor)
        if executor is None:
            # No executor (e.g. ssh without remote.yaml): whatever copy is here.
            return read_local_log(Path(job.log_path), tail_lines)
        return executor.read_log(job_id, Path(job.log_path), tail_lines)

    @staticmethod
    def _new_id(spec: JobSpec) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{spec.kind.value}-{uuid.uuid4().hex[:6]}"

    def _git_commit(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
