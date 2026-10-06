import shlex
import sys
import time
from pathlib import Path

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor, JobKind, JobSpec
from lab.core.runner import Runner
from lab.core.store import JobStore


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_base = paths.ensure_dirs()

    runner = Runner(
        store=JobStore(paths.jobs_ledger(data_base)),
        executors={
            Executor.LOCAL: LocalExecutor(repo_root=repo_root),
        },
        data_base=data_base,
        repo_root=repo_root,
    )

    code = """
import time

for step in range(60):
    print(f"step={step + 1}/60", flush=True)
    time.sleep(1)
"""

    command = shlex.join([
        sys.executable,
        "-u",
        "-c",
        code,
    ])

    job = runner.submit(
        JobSpec(
            kind=JobKind.TRAIN,
            cmd=command,
            label="API test: 60-second counter",
        )
    )

    print(f"Job ID: {job.id}")
    print(f"PID: {job.pid}")
    print(f"Log: {job.log_path}")
    
    import shlex
import sys
import time
from pathlib import Path

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor, JobKind, JobSpec
from lab.core.runner import Runner
from lab.core.store import JobStore


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_base = paths.ensure_dirs()

    runner = Runner(
        store=JobStore(paths.jobs_ledger(data_base)),
        executors={
            Executor.LOCAL: LocalExecutor(repo_root=repo_root),
        },
        data_base=data_base,
        repo_root=repo_root,
    )

    code = """
import time

for step in range(60):
    print(f"step={step + 1}/60", flush=True)
    time.sleep(1)
"""

    command = shlex.join([
        sys.executable,
        "-u",
        "-c",
        code,
    ])

    job = runner.submit(
        JobSpec(
            kind=JobKind.TRAIN,
            cmd=command,
            label="API test: 30-second counter",
        )
    )

    print(f"Job ID: {job.id}")
    print(f"PID: {job.pid}")
    print(f"Log: {job.log_path}")

    executor = runner.executors[Executor.LOCAL]

    try:
        print("Request 'stop' in Swagger.")

        while executor.is_running(job.pid):
            time.sleep(0.1)

        exit_code = executor.exit_code(job.pid)
        print(f"Process exit code: {exit_code}")

    except KeyboardInterrupt:
        executor.stop(job.pid)
        print("Stopped on the test terminal.")

    

if __name__ == "__main__":
    main()
