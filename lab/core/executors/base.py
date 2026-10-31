from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lab.core.models import JobSpec


class BaseExecutor(ABC):
    """Start a shell command detached from the lab server, then watch it by PID.

    Detached matters: the lab server restarting must not kill a training run.
    stdout+stderr go to `log_path` so the UI can tail the same file whether the
    server was up the whole time or not.

    `job_id` names per-job files on the executing machine (the ssh executor's
    remote log and exit file). The runner always passes it; executors that
    need only the PID ignore it.
    """

    @abstractmethod
    def start(self, spec: JobSpec, log_path: Path) -> int:
        """Launch and return the PID on the executing machine. log_path.stem is the job id."""

    @abstractmethod
    def is_running(self, pid: int, job_id: str | None = None) -> bool: ...

    @abstractmethod
    def exit_code(self, pid: int, job_id: str | None = None) -> int | None:
        """Exit code once finished, if this executor can still know it."""

    @abstractmethod
    def stop(self, pid: int, grace_s: float = 5.0, job_id: str | None = None) -> None:
        """SIGTERM, then SIGKILL after `grace_s` if still alive."""

    def read_log(self, job_id: str, log_path: Path, tail_lines: int | None = None) -> str:
        """The job's captured output; the last `tail_lines` lines if given."""
        return read_local_log(log_path, tail_lines)


def read_local_log(log_path: Path, tail_lines: int | None = None) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if tail_lines is None:
        return text
    return "\n".join(text.splitlines()[-tail_lines:])
