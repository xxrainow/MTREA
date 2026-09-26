"""Plain data shapes shared by core, server and (as JSON) web.

Dataclasses only — no behaviour, no I/O, no framework types. server/ turns
these into JSON; core/ passes them around. Keeping them here means a field
added for the UI is the same field the runner writes to data/jobs/.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


# --- Jobs (anything long-running: train, norm_stats, serve_policy, ...) ---


class JobKind(str, Enum):
    TRAIN = "train"
    NORM_STATS = "norm_stats"
    SERVE_POLICY = "serve_policy"
    CONVERT = "convert"
    SYNC = "sync"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


class Executor(str, Enum):
    LOCAL = "local"
    SSH = "ssh"


@dataclass(frozen=True)
class JobSpec:
    """What to run. Immutable: the same spec re-submitted must reproduce the run."""

    kind: JobKind
    # The exact shell command line. Reproducibility rule: a person must be able
    # to paste this into a terminal and get the same run (AGENTS.md).
    cmd: str
    executor: Executor = Executor.LOCAL
    # Free-form label for the UI list, e.g. "h0_seed0".
    label: str = ""
    # Working directory on the executing machine. None = repo root there.
    cwd: str | None = None


@dataclass
class Job:
    """One submitted JobSpec and everything the runner learned about it."""

    id: str
    spec: JobSpec
    status: JobStatus = JobStatus.QUEUED
    # PID on the executing machine (remote PID for ssh jobs).
    pid: int | None = None
    exit_code: int | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Path of the captured stdout/stderr, under data/logs/.
    log_path: str | None = None
    # Git commit the job was submitted from (AGENTS.md: every run is traceable).
    git_commit: str | None = None

    def is_active(self) -> bool:
        return self.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("submitted_at", "started_at", "finished_at"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        d["spec"]["kind"] = self.spec.kind.value
        d["spec"]["executor"] = self.spec.executor.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        spec = JobSpec(
            kind=JobKind(d["spec"]["kind"]),
            cmd=d["spec"]["cmd"],
            executor=Executor(d["spec"].get("executor", "local")),
            label=d["spec"].get("label", ""),
            cwd=d["spec"].get("cwd"),
        )

        def _dt(v: str | None) -> datetime | None:
            return datetime.fromisoformat(v) if v else None

        return cls(
            id=d["id"],
            spec=spec,
            status=JobStatus(d["status"]),
            pid=d.get("pid"),
            exit_code=d.get("exit_code"),
            submitted_at=_dt(d.get("submitted_at")) or datetime.now(),
            started_at=_dt(d.get("started_at")),
            finished_at=_dt(d.get("finished_at")),
            log_path=d.get("log_path"),
            git_commit=d.get("git_commit"),
        )


# --- Status lights (top bar, every page) ---


@dataclass
class RobotStatus:
    connected: bool = False
    follower_port: str | None = None
    leader_port: str | None = None
    cameras_ok: bool = False
    detail: str = ""


@dataclass
class GpuHostStatus:
    reachable: bool = False
    host: str | None = None
    # Whether a training job currently holds the (single) GPU.
    busy: bool = False
    detail: str = ""


@dataclass
class PolicyServerStatus:
    # Probe result on the robot-side tunnel end (127.0.0.1:8765), not the GPU
    # host: if the tunnel is down the arm cannot be driven, whatever the
    # server is doing.
    ready: bool = False
    loading: bool = False
    checkpoint: str | None = None
    detail: str = ""


@dataclass
class SystemStatus:
    robot: RobotStatus = field(default_factory=RobotStatus)
    gpu: GpuHostStatus = field(default_factory=GpuHostStatus)
    policy_server: PolicyServerStatus = field(default_factory=PolicyServerStatus)


# --- Artifacts the UI lists ---


@dataclass
class DatasetInfo:
    name: str
    path: str
    num_episodes: int = 0
    tasks: list[str] = field(default_factory=list)
    has_norm_stats: bool = False
    validated: bool | None = None  # None = validate.py not run yet


@dataclass
class CheckpointInfo:
    run_id: str
    step: int
    # Path on the GPU host; checkpoints stay remote, only the listing comes over.
    remote_path: str
    recipe: str | None = None


# --- Rollouts (the raw material of eval; append-only, never rewritten) ---


@dataclass
class RolloutRecord:
    id: str
    checkpoint: str  # run_id/step, or "pretrained" for the pre-adaptation baseline
    task: str  # task name from research/collect/tasks/*.yaml
    instruction: str
    success: bool | None  # None until a human marks it
    started_at: datetime
    duration_s: float = 0.0
    video_path: str | None = None
    trajectory_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        return d
