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
from lab.core.executors.base import BaseExecutor
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.store import JobStore


class JobConflict(RuntimeError):
    """A job of this kind is already active (one GPU, one policy server)."""


# Kinds that must not overlap with themselves. Train and norm_stats also
# compete for the single GPU, so they exclude each other.
_EXCLUSIVE: dict[JobKind, set[JobKind]] = {
    JobKind.TRAIN: {JobKind.TRAIN, JobKind.NORM_STATS},
    JobKind.NORM_STATS: {JobKind.TRAIN, JobKind.NORM_STATS},
    JobKind.SERVE_POLICY: {JobKind.SERVE_POLICY},
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
            if job.status != JobStatus.RUNNING or job.pid is None:
                continue
            executor = self.executors.get(job.spec.executor)
            if executor is None or executor.is_running(job.pid):
                continue
            job.exit_code = executor.exit_code(job.pid)
            job.status = JobStatus.DONE if job.exit_code == 0 else JobStatus.FAILED
            job.finished_at = datetime.now()
            self.store.save(job)
        return jobs

    def active(self) -> list[Job]:
        return [j for j in self.store.list() if j.is_active()]

    def get(self, job_id: str) -> Job | None:
        self.refresh()
        return self.store.get(job_id)

    # --- stop ---

    def stop(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == JobStatus.RUNNING and job.pid is not None:
            self.executors[job.spec.executor].stop(job.pid)
            job.status = JobStatus.STOPPED
            job.finished_at = datetime.now()
            self.store.save(job)
        return job

    # --- helpers ---

    def read_log(self, job_id: str, tail_lines: int | None = None) -> str:
        job = self.store.get(job_id)
        if job is None or not job.log_path or not Path(job.log_path).exists():
            return ""
        text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")
        if tail_lines is None:
            return text
        return "\n".join(text.splitlines()[-tail_lines:])

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
