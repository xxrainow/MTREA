import posixpath
import uuid
import sys
import os
import shlex
import subprocess

from lab.core.datasets import dataset_root
from robot.config import FPS, POLICY_SERVER_PORT

from pathlib import Path, PurePosixPath

from lab.core import training
from lab.core.models import Executor, Job, JobKind, JobSpec
from lab.core.remote import Remote, RemoteError
from lab.core.runner import Runner


def command(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)

def checkpoint_options(jobs: list[Job], remote: Remote | None) -> dict:
    options = []
    errors = []

    for job in jobs:
        if job.spec.kind != JobKind.TRAIN:
            continue
        if job.spec.executor != Executor.SSH or not job.spec.run_dir:
            continue
        if job.is_active():
            continue

        try:
            reader = training.run_dir_of(job, remote)
            status = training.read_json(reader, "status.json") or {}
            config = training.read_json(reader, "config.json") or {}

            if status.get("fake") or config.get("fake") or "--fake" in job.spec.cmd:
                continue

            saved = training.read_json(reader, "checkpoints.json") or {}

            for entry in saved.get("checkpoints", []):
                if not entry.get("last") or not entry.get("complete"):
                    continue

                relative = PurePosixPath(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    continue

                options.append({
                    "job_id": job.id,
                    "step": entry["step"],
                    "label": job.spec.label,
                    "checkpoint": posixpath.join(
                        job.spec.run_dir,
                        str(relative),
                        "pretrained_model",
                    ),
                })

        except (RemoteError, OSError) as exc:
            errors.append(str(exc))

    return {
        "checkpoints": options,
        "errors": list(dict.fromkeys(errors)),
    }

def start(
    *,
    target: str,
    mode: str,
    source_job_id: str | None,
    dataset_name: str | None,
    runner: Runner,
    remote: Remote | None,
    data_base: Path,
) -> Job:
    if target not in ("ssh", "local"):
        raise ValueError("target must be ssh or local")

    if mode not in ("real", "hold"):
        raise ValueError("mode must be real or hold")

    if target == "local" and mode != "hold":
        raise ValueError(
            "This machine supports fake hold only; "
            "real serving needs the GPU host"
        )

    session = f"serve_{uuid.uuid4().hex}"

    if target == "ssh":
        if remote is None:
            raise ValueError("GPU host is not configured")

        remote.config.policy_tunnel_argv()

        choices = checkpoint_options(runner.store.list(), remote)
        chosen = next(
            (
                checkpoint
                for checkpoint in choices["checkpoints"]
                if checkpoint["job_id"] == source_job_id
            ),
            None,
        )

        if chosen is None:
            detail = "; ".join(choices["errors"])
            message = "Select a complete last checkpoint from a real GPU training run"
            if detail:
                message += f": {detail}"
            raise ValueError(message)

        checkpoint = chosen["checkpoint"]
        base = remote.config.data_dir
        port = remote.config.policy_port
        session_dir = posixpath.join(base, "rollouts", "serve", session)
        argv = ["bash", "scripts/serve_policy.sh"]

    else:
        if not dataset_name:
            raise ValueError("Select a recorded dataset for fake hold")

        dataset = dataset_root("real", dataset_name, data_base)

        if not (dataset / "meta" / "info.json").is_file():
            raise ValueError("The recorded dataset has no meta/info.json")

        base = str(data_base / "fake")
        checkpoint = str(Path(base) / "serving-feature-only")
        port = POLICY_SERVER_PORT
        session_dir = str(
            Path(base) / "rollouts" / "serve" / session
        )

        argv = [
            sys.executable,
            "-m",
            "research.serve_policy",
            f"--dataset-root={dataset}",
        ]
        
    argv += [
        f"--checkpoint={checkpoint}",
        "--host=127.0.0.1",
        f"--port={port}",
        f"--fps={FPS}",
        f"--session={session}",
        f"--data-dir={base}",
    ]

    if mode == "hold":
        argv.append("--fake=hold")
        
    return runner.submit(
        JobSpec(
            kind=JobKind.SERVE_POLICY,
            cmd=(
                shlex.join(argv)
                if target == "ssh"
                else command(argv)
            ),
            executor=(
                Executor.SSH
                if target == "ssh"
                else Executor.LOCAL
            ),
            label=session,
            run_dir=session_dir,
            host=remote.label if target == "ssh" else None,
            metadata={
                "checkpoint": checkpoint,
                "mode": mode,
                "source_job_id": source_job_id,
                "server_port": port,
            },
        )
    )
