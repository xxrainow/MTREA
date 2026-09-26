"""Job ledger: data/jobs/jobs.jsonl.

Append-only. Every state change appends the whole Job as one JSON line; the
latest line per id wins. Rewriting nothing means a crash mid-write loses at
most one line, and the file doubles as a history of what was run when.
"""

from __future__ import annotations

import json
from pathlib import Path

from lab.core.models import Job


class JobStore:
    def __init__(self, ledger: Path):
        self.ledger = ledger
        self.ledger.parent.mkdir(parents=True, exist_ok=True)

    def save(self, job: Job) -> None:
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> dict[str, Job]:
        """Latest state of every job ever recorded, keyed by id."""
        jobs: dict[str, Job] = {}
        if not self.ledger.exists():
            return jobs
        with self.ledger.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = Job.from_dict(json.loads(line))
                jobs[job.id] = job
        return jobs

    def get(self, job_id: str) -> Job | None:
        return self.load_all().get(job_id)

    def list(self, newest_first: bool = True) -> list[Job]:
        jobs = list(self.load_all().values())
        jobs.sort(key=lambda j: j.submitted_at, reverse=newest_first)
        return jobs
