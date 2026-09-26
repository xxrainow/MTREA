"""Where a job's command actually runs: this machine (local) or the GPU host (ssh).

The runner picks one by JobSpec.executor and treats both the same way:
start → poll → stop. Nothing about the command changes between them.
"""

from lab.core.executors.base import BaseExecutor
from lab.core.executors.local import LocalExecutor

__all__ = ["BaseExecutor", "LocalExecutor"]
