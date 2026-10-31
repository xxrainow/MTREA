"""Run a job on the GPU host, detached from the ssh session that started it.

    setsid nohup bash -c '<remote_prefix> <cmd>; echo $? > <job>.exit' > <job>.log

The remote shell writes the log and the exit file itself, so the job survives
the lab server, the ssh connection and a laptop going to sleep. setsid makes
the job's PID its process-group id, so stop() signals python and lerobot
together. Every command is built through lab.core.remote and runs through its
transport; tests swap in LocalTransport, which runs the same shell locally.

Unreachable is not failed: a RemoteError propagates to the runner, which
leaves the job's status as it was.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from pathlib import Path

from lab.core.executors.base import BaseExecutor, read_local_log
from lab.core.models import JobSpec
from lab.core.remote import Remote, RemoteError

JOB_ID = re.compile(r"[A-Za-z0-9_.-]+")
STOP_GRACE_S = 20.0  # train.py forwards SIGTERM to lerobot, then writes its files and prunes


class SshExecutor(BaseExecutor):
    def __init__(self, remote: Remote):
        self.remote = remote

    def _files(self, job_id: str | None) -> tuple[str, str]:
        if job_id is None or not JOB_ID.fullmatch(job_id):
            raise ValueError(f"invalid job id {job_id!r}")
        logs = self.remote.remote_logs_dir()
        return posixpath.join(logs, f"{job_id}.log"), posixpath.join(logs, f"{job_id}.exit")

    def _run(self, command: str, what: str, timeout: float = 60) -> str:
        cp = self.remote.transport.run(command, timeout=timeout)
        if cp.returncode != 0:
            err = cp.stderr.decode("utf-8", "replace").strip().splitlines()
            tail = err[-1] if err else "no message"
            if cp.returncode == 255:
                raise RemoteError(f"cannot reach {self.remote.label}: {tail}")
            raise RemoteError(f"{what} failed on {self.remote.label} (exit {cp.returncode}): {tail}")
        return cp.stdout.decode("utf-8", "replace")

    def start(self, spec: JobSpec, log_path: Path) -> int:
        log, exit_file = self._files(log_path.stem)
        q = shlex.quote
        # The subshell keeps a command that calls `exit` from skipping the
        # exit file. Written then renamed, so a reader never sees it empty.
        inner = (f"( {self.remote.remote_prefix()} {spec.cmd} ); "
                 f"echo $? > {q(exit_file)}.tmp && mv -f {q(exit_file)}.tmp {q(exit_file)}")
        # `;`, not `&&`, before the background job: `a && b &` would background
        # a subshell and $! would be its PID instead of the job's.
        command = (f"mkdir -p {q(self.remote.remote_logs_dir())} || exit 1; "
                   f"setsid nohup bash -c {q(inner)} > {q(log)} 2>&1 < /dev/null & echo $!")
        out = self._run(command, "start job", timeout=30)
        try:
            pid = int(out.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise RemoteError(f"start job: no PID from {self.remote.label}: {out.strip()!r}") from None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        return pid

    def _state(self, pid: int, job_id: str | None) -> tuple[str, int | None]:
        """("running" | "exited" | "gone", exit code if exited)."""
        _, exit_file = self._files(job_id)
        q = shlex.quote(exit_file)
        # Not `kill -0`: a container's PID 1 may never reap orphans, and a
        # zombie still answers kill -0.
        out = self._run(f"if [ -f {q} ]; then echo exited; cat {q}; "
                        f"elif ps -o stat= -p {int(pid)} | grep -qv '^Z'; then echo running; else echo gone; fi",
                        "job status").split()
        if out and out[0] == "exited":
            try:
                return "exited", int(out[1])
            except (IndexError, ValueError):
                return "exited", None
        return (out[0] if out else "gone"), None

    def is_running(self, pid: int, job_id: str | None = None) -> bool:
        return self._state(pid, job_id)[0] == "running"

    def exit_code(self, pid: int, job_id: str | None = None) -> int | None:
        # Gone without an exit file (container restart, SIGKILL): None, and the
        # runner records UNKNOWN.
        return self._state(pid, job_id)[1]

    def stop(self, pid: int, grace_s: float = STOP_GRACE_S, job_id: str | None = None) -> None:
        group = f"-{int(pid)}"
        ticks = max(1, int(grace_s / 0.5))
        # Zombies do not count as alive. `ps -g` selects by session; setsid
        # made the session id equal to the group id.
        alive = f"ps -o stat= -g {int(pid)} | grep -qv '^Z'"
        # `kill -s SIG --`: dash (sh on Debian/Ubuntu) rejects `kill -TERM -- -pgid`.
        self._run(f"kill -s TERM -- {group} 2>/dev/null || exit 0; "
                  f"for i in $(seq {ticks}); do {alive} || exit 0; sleep 0.5; done; "
                  f"kill -s KILL -- {group} 2>/dev/null; exit 0",
                  "stop job", timeout=grace_s + 30)

    def read_log(self, job_id: str, log_path: Path, tail_lines: int | None = None) -> str:
        """The remote log's tail, also kept at log_path; the local copy if unreachable."""
        log, _ = self._files(job_id)
        q = shlex.quote(log)
        read = f"cat {q}" if tail_lines is None else f"tail -n {int(tail_lines)} {q}"
        try:
            text = self._run(f"if [ -f {q} ]; then {read}; fi", "read log")
        except RemoteError:
            return read_local_log(log_path, tail_lines)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
        return text.rstrip("\n")
