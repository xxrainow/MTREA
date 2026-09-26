"""lab/core boundaries and the data/ layout.

Companion to test_boundary.py. Three rules from AGENTS.md:
  - lab/core never imports FastAPI (core is usable from a shell/tests alone)
  - lab/core/runner.py never imports research (it runs scripts as subprocesses)
  - ensure_dirs() produces exactly the documented data/ layout
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "lab" / "core"


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_does_not_import_fastapi() -> None:
    offenders = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in sorted(CORE_DIR.rglob("*.py"))
        if {"fastapi", "starlette", "uvicorn"} & _imported_top_levels(p)
    ]
    assert not offenders, "lab/core must not import the web framework:\n  " + "\n  ".join(offenders)


def test_runner_does_not_import_research() -> None:
    runner = CORE_DIR / "runner.py"
    if not runner.exists():
        pytest.skip("runner.py not written yet")
    assert "research" not in _imported_top_levels(runner), (
        "lab/core/runner.py must run research scripts as subprocesses, not import them"
    )


def test_ensure_dirs_creates_layout(tmp_path: Path) -> None:
    from lab.core import paths

    root = paths.ensure_dirs(tmp_path)
    assert root == tmp_path
    created = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert created == sorted(paths.SUBDIRS)
    # idempotent
    paths.ensure_dirs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == created


def test_paths_are_under_root(tmp_path: Path) -> None:
    from lab.core import paths

    assert paths.jobs_ledger(tmp_path) == tmp_path / "jobs" / "jobs.jsonl"
    assert paths.job_log("j1", tmp_path) == tmp_path / "logs" / "j1.log"
    assert paths.run_dir("h0_seed0", tmp_path) == tmp_path / "runs" / "h0_seed0"
    assert paths.dataset_dir("pick_place", tmp_path) == tmp_path / "datasets" / "pick_place"


def test_job_round_trips_through_dict() -> None:
    from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus

    spec = JobSpec(
        kind=JobKind.TRAIN,
        cmd="bash research/scripts/train.sh --recipe recipes/h0.yaml --out data/runs/h0_seed0",
        executor=Executor.SSH,
        label="h0_seed0",
    )
    job = Job(id="j1", spec=spec, status=JobStatus.RUNNING, pid=4321,
              started_at=datetime(2026, 10, 3, 14, 2), log_path="data/logs/j1.log")
    back = Job.from_dict(job.to_dict())
    assert back == job
    assert back.is_active()
