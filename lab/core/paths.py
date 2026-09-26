"""The data/ layout, as code.

This is the only place that spells out where things live under data/. lab
calls these functions; research scripts receive the resulting paths as
arguments (they never import lab). Same layout on both machines.

    data/
      raw/         LeRobot recordings straight from lerobot-record
      datasets/    converted, validated training sets
      norm_stats/  per-dataset normalisation statistics
      runs/        one dir per training run: config, metrics.jsonl, checkpoints
      rollouts/    policy executions on the real robot (append-only)
      jobs/        runner ledger (jobs.jsonl)
      logs/        captured stdout/stderr, one file per job
"""

from __future__ import annotations

from pathlib import Path

from robot.config import data_root

SUBDIRS = (
    "raw",
    "datasets",
    "norm_stats",
    "runs",
    "rollouts",
    "jobs",
    "logs",
)


def root(base: Path | None = None) -> Path:
    """data/ root. `base` overrides the configured root (tests, one-off scripts)."""
    return base if base is not None else data_root()


def raw_dir(base: Path | None = None) -> Path:
    return root(base) / "raw"


def datasets_dir(base: Path | None = None) -> Path:
    return root(base) / "datasets"


def norm_stats_dir(base: Path | None = None) -> Path:
    return root(base) / "norm_stats"


def runs_dir(base: Path | None = None) -> Path:
    return root(base) / "runs"


def rollouts_dir(base: Path | None = None) -> Path:
    return root(base) / "rollouts"


def jobs_dir(base: Path | None = None) -> Path:
    return root(base) / "jobs"


def logs_dir(base: Path | None = None) -> Path:
    return root(base) / "logs"


def jobs_ledger(base: Path | None = None) -> Path:
    return jobs_dir(base) / "jobs.jsonl"


def job_log(job_id: str, base: Path | None = None) -> Path:
    return logs_dir(base) / f"{job_id}.log"


def run_dir(run_id: str, base: Path | None = None) -> Path:
    return runs_dir(base) / run_id


def dataset_dir(name: str, base: Path | None = None) -> Path:
    return datasets_dir(base) / name


# examples
def ensure_dirs(base: Path | None = None) -> Path:
    """Create the whole layout. Called once at server start; idempotent."""
    r = root(base)
    for sub in SUBDIRS:
        (r / sub).mkdir(parents=True, exist_ok=True)
    return r
