from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from lab.core.executors.base import BaseExecutor
from lab.core.models import JobSpec


class LocalExecutor(BaseExecutor):
    """Run the command on this machine in its own session (survives our exit)."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        # Popen handles for jobs started in this process. After a server
        # restart these are gone; we fall back to PID probing and lose the
        # exit code — acceptable, the log file still says what happened.
        self._procs: dict[int, subprocess.Popen] = {}

    def start(self, spec: JobSpec, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cwd = Path(spec.cwd) if spec.cwd else self.repo_root
        log = log_path.open("ab")
        proc = subprocess.Popen(
            spec.cmd,
            shell=True,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group: our SIGINT doesn't reach it
        )
        log.close()  # child holds its own fd
        self._procs[proc.pid] = proc
        return proc.pid

    def is_running(self, pid: int) -> bool:
        proc = self._procs.get(pid)
        if proc is not None:
            return proc.poll() is None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def exit_code(self, pid: int) -> int | None:
        proc = self._procs.get(pid)
        return proc.returncode if proc is not None else None

    def stop(self, pid: int, grace_s: float = 5.0) -> None:
        # Signal the whole group so `bash -c "python train.py"` takes python
        # down with it, not just the bash wrapper.
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not self.is_running(pid):
                return
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc = self._procs.get(pid)
        if proc is not None:
            proc.wait(timeout=2)
