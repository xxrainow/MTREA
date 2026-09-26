from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lab.core.models import JobSpec


class BaseExecutor(ABC):
    """Start a shell command detached from the lab server, then watch it by PID.

    Detached matters: the lab server restarting must not kill a training run.
    stdout+stderr go to `log_path` so the UI can tail the same file whether the
    server was up the whole time or not.
    """

    @abstractmethod
    def start(self, spec: JobSpec, log_path: Path) -> int:
        """Launch and return the PID on the executing machine."""

    @abstractmethod
    def is_running(self, pid: int) -> bool: ...

    @abstractmethod
    def exit_code(self, pid: int) -> int | None:
        """Exit code once finished, if this executor can still know it."""

    @abstractmethod
    def stop(self, pid: int, grace_s: float = 5.0) -> None:
        """SIGTERM, then SIGKILL after `grace_s` if still alive."""
