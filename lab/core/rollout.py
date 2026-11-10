"""Rollout jobs and their file contract; device access stays in the worker."""

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from lab.core import paths, serving
from lab.core.hardware import HardwareBusy
from lab.core.hardware.lock import HardwareLock
from lab.core.models import Executor, Job, JobKind, JobSpec, RolloutConfig
from lab.core.remote import Remote
from lab.core.runner import JobConflict, Runner
from robot.config import POLICY_SERVER_HOST, POLICY_SERVER_PORT
from robot.rollout_commands import owner_alive, submit_command
from robot.rollout_preview import read_trial_frame


def validate_rollout_config(config: RolloutConfig) -> None:
    if not isinstance(config.server_job_id, str) or not config.server_job_id.strip():
        raise ValueError("Select a policy server session")
    if not isinstance(config.instruction, str) or not config.instruction.strip():
        raise ValueError("Enter an instruction")
    for name, value in (("max_relative_target", config.max_relative_target),
                        ("time_limit_s", config.time_limit_s)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a finite positive number")


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


class RolloutService:
    def __init__(self, data_base: Path | None = None):
        self.data_base = paths.root(data_base).resolve()
        self.root = paths.rollouts_dir(self.data_base) / "robot"

    def _root(self, job: Job) -> Path:
        if job.spec.kind != JobKind.ROLLOUT or not job.spec.run_dir:
            raise ValueError("A rollout job with a trial directory is required")
        trial = Path(job.spec.run_dir).resolve()
        if trial.parent != self.root.resolve():
            raise ValueError("Rollout directory is outside this Lab's storage")
        return trial

    def get_job(self, job_id: str, runner: Runner) -> Job:
        job = runner.store.get(job_id)
        if job is None or job.spec.kind != JobKind.ROLLOUT:
            raise KeyError(job_id)
        self._root(job)
        return job

    def snapshot(self, job: Job) -> dict:
        root = self._root(job)
        status = session = None
        error = None
        try:
            status = _read(root / "status.json")
            session = _read(root / "session.json")
        except (OSError, ValueError) as exc:
            error = str(exc)
        alive = owner_alive(status) if status else False
        state = status.get("state") if status else "connecting"
        if not alive and (state in {"connecting", "running", "stopping"}) and not job.is_active():
            state = "ended_unknown"
        elif status and status.get("process_alive") and not alive:
            state = "ended_unknown"
        return {"job": job.to_dict(), "state": state, "status": status,
                "session": session, "owner_alive": alive, "read_error": error}

    def sessions(self, runner: Runner) -> list[dict]:
        return [self.snapshot(job) for job in runner.refresh(kind=JobKind.ROLLOUT)
                if job.spec.kind == JobKind.ROLLOUT]

    def is_active(self, runner: Runner) -> bool:
        for job in runner.refresh(kind=JobKind.ROLLOUT):
            if job.spec.kind == JobKind.ROLLOUT:
                if job.is_active() or self.snapshot(job)["owner_alive"]:
                    return True
        return False

    def check(self, config: RolloutConfig, runner: Runner, remote: Remote | None) -> dict:
        validate_rollout_config(config)
        if self.is_active(runner):
            raise JobConflict("A rollout still owns the devices; Stop and Release it first")
        holder = HardwareLock(paths.hardware_lock(self.data_base)).holder()
        if holder:
            raise HardwareBusy(f"The arms are in use by {holder.get('owner', 'another session')}")
        jobs = runner.refresh()
        job = next((item for item in jobs if item.id == config.server_job_id), None)
        if job is None or job.spec.kind != JobKind.SERVE_POLICY:
            raise ValueError("Select a policy server session")
        session = serving.session_status(job, jobs, remote)
        if session["unreachable"] or session["read_error"]:
            raise ValueError(session["unreachable"] or session["read_error"])
        if session["state"] not in ("ready", "serving"):
            raise ValueError("The policy server is not ready")
        if job.spec.executor == Executor.SSH:
            tunnel = session["tunnel"]
            if tunnel is None or tunnel["status"] != "running":
                raise ValueError("Start the SSH tunnel for this policy server")
            port = POLICY_SERVER_PORT
        else:
            port = (session["status"] or {}).get("port", job.spec.metadata.get("server_port"))
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("The policy server has no valid local port")
        checkpoint = job.spec.metadata.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("The policy session has no checkpoint string")
        return {"session": session, "checkpoint": checkpoint,
                "server_address": f"{POLICY_SERVER_HOST}:{port}",
                "configuration": asdict(config)}

    def start(self, config: RolloutConfig, runner: Runner, remote: Remote | None,
              *, prepare_devices=None) -> Job:
        checked = self.check(config, runner, remote)
        if prepare_devices is not None:
            prepare_devices()
        trial = self.root / uuid4().hex
        argv = [sys.executable, "-m", "robot.rollout_worker",
                f"--checkpoint={checked['checkpoint']}", f"--task={config.instruction}",
                f"--server={checked['server_address']}", f"--trial-dir={trial}",
                f"--data-dir={self.data_base}", f"--max-relative-target={config.max_relative_target}",
                f"--time-limit-s={config.time_limit_s}"]
        return runner.submit(JobSpec(
            kind=JobKind.ROLLOUT, executor=Executor.LOCAL, cmd=serving.command(argv),
            cwd=str(runner.repo_root), run_dir=str(trial), label=config.instruction,
            metadata={**asdict(config), "checkpoint": checked["checkpoint"],
                      "server_address": checked["server_address"], "argv": argv},
        ))

    def command(self, job_id: str, action: str, runner: Runner) -> dict:
        job = self.get_job(job_id, runner)
        root = self._root(job)
        snapshot = self.snapshot(job)
        if snapshot["read_error"]:
            raise OSError(snapshot["read_error"])
        status = snapshot["status"]
        if not status:
            raise JobConflict("The rollout command channel is not ready yet")
        if not snapshot["owner_alive"]:
            raise JobConflict("The rollout owner is no longer accepting commands")
        if action == "release" and status.get("state") not in {"stopped", "aborted", "failed"}:
            raise JobConflict("Wait for Stop and policy cleanup before Release")
        try:
            command_id = submit_command(root, action)
        except RuntimeError as exc:
            raise JobConflict(str(exc)) from exc
        return {"job_id": job.id, "command_id": command_id, "state": "queued"}

    def command_result(self, job_id: str, command_id: str, runner: Runner) -> dict:
        if len(command_id) != 32 or any(char not in "0123456789abcdef" for char in command_id):
            raise KeyError(command_id)
        root = self._root(self.get_job(job_id, runner))
        if _read(root / "commands" / f"{command_id}.json") is None:
            raise KeyError(command_id)
        return (_read(root / "command_results" / f"{command_id}.json")
                or {"id": command_id, "state": "queued", "error": None})

    def frame(self, job_id: str, name: str, runner: Runner) -> bytes:
        job = self.get_job(job_id, runner)
        snapshot = self.snapshot(job)
        status = snapshot["status"] or {}
        return read_trial_frame(
            self._root(job), name,
            active=bool(snapshot["owner_alive"] and snapshot["state"] == "running"
                        and status.get("preview_active") and not snapshot["read_error"]),
            error=status.get("preview_error"),
        )
