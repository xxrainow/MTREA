import math
from threading import Lock

from lab.core.models import (
    RolloutConfig,
    RolloutPhase,
    RolloutStatus,
)
from lab.core import serving
from lab.core.models import Executor, JobKind
from lab.core.remote import Remote
from lab.core.runner import Runner


def validate_rollout_config(config: RolloutConfig) -> None:
    if not isinstance(config.server_job_id, str) or not config.server_job_id.strip():
        raise ValueError("Select a policy server session")

    if not isinstance(config.instruction, str) or not config.instruction.strip():
        raise ValueError("Enter an instruction")

    for name, value in (
        ("max_relative_target", config.max_relative_target),
        ("time_limit_s", config.time_limit_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")





class RolloutService:
    def __init__(self) -> None:
        self._mutex = Lock()
        self._status = RolloutStatus()

    def status(self) -> RolloutStatus:
        with self._mutex:
            return self._status

    def is_active(self) -> bool:
        with self._mutex:
            return self._status.phase in (
                RolloutPhase.CONNECTING,
                RolloutPhase.RUNNING,
                RolloutPhase.STOPPING,
            )
    def check(
        self,
        config: RolloutConfig,
        runner: Runner,
        remote: Remote | None,
    ) -> dict:
        validate_rollout_config(config)

        if self.is_active():
            raise ValueError("A rollout is already active")

        jobs = runner.refresh()
        job = next(
            (item for item in jobs if item.id == config.server_job_id),
            None,
        )
        if job is None or job.spec.kind != JobKind.SERVE_POLICY:
            raise ValueError("Select a policy server session")

        session = serving.session_status(job, jobs, remote)

        if session["unreachable"]:
            raise ValueError(session["unreachable"])
        if session["read_error"]:
            raise ValueError(session["read_error"])
        if session["state"] not in ("ready", "serving"):
            raise ValueError("The policy server is not ready")

        if job.spec.executor == Executor.SSH:
            tunnel = session["tunnel"]
            if tunnel is None or tunnel["status"] != "running":
                raise ValueError("Start the SSH tunnel for this policy server")

        return session
