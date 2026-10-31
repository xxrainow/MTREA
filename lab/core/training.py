"""Training jobs for lab's Train page: resolve a request, check it, start it, read its run files.

research/train.py is driven only through its CLI and its run directory
(docs/TRAINING_JOBS.md); lab never imports research/. Targets:

    ssh    the GPU host. Real training, or --fake to exercise the remote path.
    local  this machine, --fake only: there is no local GPU for pi05.

The form sends only what the user set. train.sh fills its own defaults, and
--check returns the fully explicit command that is then submitted, so the job
ledger holds a command line that reproduces the run.

Run files are the source of truth for progress: status.json, metrics.jsonl,
checkpoints.json, result.json, config.json, read locally for local jobs and
through the remote transport for ssh jobs.
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from lab.core import paths
from lab.core.dataset_sync import remote_state
from lab.core.datasets import dataset_root, validate_dataset_name
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.remote import Remote, RemoteError
from lab.core.runner import Runner

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("ssh", "local")
RUN_FILES = ("status.json", "metrics.jsonl", "checkpoints.json", "result.json", "config.json")
# Set by lab from the target and dataset, never by the form.
LAB_PARAMS = {"dataset", "dataset_root", "output", "fake"}
CHECK_TIMEOUT_S = 120
ENTRYPOINT = "scripts/train.sh"


class TrainError(ValueError):
    """A malformed request: unknown target, parameter or value."""


class CheckFailed(RuntimeError):
    def __init__(self, result: dict):
        super().__init__("checks failed")
        self.result = result


@dataclass(frozen=True)
class TrainRequest:
    target: str
    dataset_backend: str
    dataset_name: str
    run_name: str
    fake: bool = False
    # describe() param name -> value, only what the user set.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolved:
    target: str
    fake: bool
    args: list[str]  # train.sh options, without the entrypoint
    run_dir: str
    host: Optional[str]
    checks: list[dict]  # lab-side checks; a failing one means train.sh --check is not run


def _check(id: str, status: str, message: str, **extra) -> dict:
    return {"id": id, "status": status, "message": message, **extra}


def _failed(checks: list[dict]) -> bool:
    return any(c["status"] == "fail" for c in checks)


# --- describe ---

_describe_lock = threading.Lock()
_describe_cache: dict[Path, dict] = {}


def describe(repo_root: Path = REPO_ROOT) -> dict:
    """train.sh --describe of this checkout, cached for the server's lifetime."""
    with _describe_lock:
        if repo_root not in _describe_cache:
            cp = subprocess.run([sys.executable, "-m", "research.train", "--describe"], cwd=repo_root,
                                capture_output=True, timeout=60)
            if cp.returncode != 0:
                raise RuntimeError(f"train --describe failed: {cp.stderr.decode(errors='replace').strip()}")
            _describe_cache[repo_root] = json.loads(cp.stdout)
        return _describe_cache[repo_root]


# --- resolve ---

def _option(param: dict, value: Any) -> str:
    """One `--flag=value` from a form value, validated against describe()."""
    name, kind, flag = param["name"], param["type"], param["flag"]
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise TrainError(f"{name} must be a non-empty string")
        if param.get("choices") and value not in param["choices"]:
            raise TrainError(f"{name} must be one of {param['choices']}")
        text = value
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TrainError(f"{name} must be an integer")
        if param.get("min") is not None and value < param["min"]:
            raise TrainError(f"{name} must be at least {param['min']}")
        text = str(value)
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TrainError(f"{name} must be a finite number")
        if param.get("min") is not None and value < param["min"]:
            raise TrainError(f"{name} must be at least {param['min']}")
        text = repr(float(value))
    elif kind == "bool":
        if not isinstance(value, bool):
            raise TrainError(f"{name} must be true or false")
        text = "true" if value else "false"
    elif kind == "int_list":
        if not isinstance(value, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
            raise TrainError(f"{name} must be a list of integers")
        text = ",".join(map(str, value))
    else:
        raise TrainError(f"{name}: unsupported type {kind!r}")
    return f"{flag}={text}"


def _local_dataset_root(backend: str, name: str, data_base: Optional[Path]) -> Path:
    if backend == "real" and paths.dataset_dir(name, data_base).is_dir():
        return paths.dataset_dir(name, data_base)
    return dataset_root(backend, name, data_base)


def resolve(req: TrainRequest, spec: dict, remote: Optional[Remote], data_base: Optional[Path] = None,
            not_configured: str = "") -> Resolved:
    """Turn a form request into train.sh options and lab-side checks. Raises TrainError."""
    if req.target not in TARGETS:
        raise TrainError(f"target must be one of {TARGETS}")
    if req.dataset_backend not in ("real", "fake"):
        raise TrainError("dataset backend must be real or fake")
    try:
        validate_dataset_name(req.dataset_name)
        validate_dataset_name(req.run_name)
    except ValueError:
        raise TrainError("dataset and run names may contain letters, digits, '_', '.' and '-'") from None

    params = {p["name"]: p for p in spec["params"]}
    for name in req.params:
        if name not in params or name in LAB_PARAMS or params[name]["group"] == "fixed":
            raise TrainError(f"{name} is not a settable parameter")
        if params[name]["group"] == "fake" and not req.fake:
            raise TrainError(f"{name} applies only to fake runs")
    checks: list[dict] = []
    missing = [p["label"] for p in spec["params"]
               if p["required"] and p["name"] not in LAB_PARAMS and req.params.get(p["name"]) in (None, "")]
    if missing:
        checks.append(_check("params", "fail", f"Required: {', '.join(missing)}"))

    host = None
    dataset_path = None
    if req.target == "ssh":
        if remote is None:
            checks.append(_check("remote", "fail", f"GPU host not configured: {not_configured}"))
            run_dir = ""
        else:
            host = remote.label
            run_dir = posixpath.join(remote.remote_runs_dir(), req.run_name)
            ok, message = remote.ping()
            checks.append(_check("remote", "ok" if ok else "fail", message))
            if ok and req.dataset_backend != "real":
                checks.append(_check("dataset_sync", "fail", "Only real datasets are sent to the GPU host."))
            elif ok:
                state = remote_state(remote, "real", req.dataset_name, data_base)
                if state["state"] == "synced":
                    dataset_path = state["remote_path"]
                    checks.append(_check("dataset_sync", "ok", f"{req.dataset_name} is on the GPU host, "
                                         f"fingerprint {state['remote_fingerprint'][:12]}"))
                else:
                    detail = state["message"] or state["reason"] or f"state {state['state']}"
                    checks.append(_check("dataset_sync", "fail", "Sync this dataset to the GPU host first "
                                         f"({detail}).", link="datasets"))
    else:
        run_dir = str(paths.fake_runs_dir(data_base) / req.run_name)
        if not req.fake:
            checks.append(_check("target", "fail", "This machine has no GPU for pi05: real training runs "
                                 "only on the GPU host. Turn on fake mode to run here."))
        try:
            dataset_path = str(_local_dataset_root(req.dataset_backend, req.dataset_name, data_base))
        except FileNotFoundError:
            checks.append(_check("dataset", "fail", f"Dataset {req.dataset_backend}/{req.dataset_name} not found."))

    values = {**req.params, "dataset": f"local/{req.dataset_name}", "dataset_root": dataset_path,
              "output": run_dir}
    args = []
    for p in spec["params"]:
        value = values.get(p["name"])
        if p["name"] == "fake" or value is None or value == "":
            continue
        args.append(_option(p, value))
    if req.fake:
        args.append("--fake")
    return Resolved(req.target, req.fake, args, run_dir, host, checks)


# --- check ---

def _parse_check(cp: subprocess.CompletedProcess, where: str) -> dict:
    try:
        out = json.loads(cp.stdout)
        if isinstance(out, dict) and isinstance(out.get("checks"), list):
            return out
    except ValueError:
        pass
    err = cp.stderr.decode("utf-8", "replace").strip().splitlines() if isinstance(cp.stderr, bytes) else []
    raise RemoteError(f"train.sh --check on {where} exited {cp.returncode}: {err[-1] if err else 'no output'}")


def _target_check(res: Resolved, remote: Optional[Remote], repo_root: Path) -> dict:
    if res.target == "ssh":
        assert remote is not None
        cp = remote.transport.run(f"{remote.remote_prefix()} bash {ENTRYPOINT} {shlex.join([*res.args, '--check'])}",
                                  timeout=CHECK_TIMEOUT_S)
        if cp.returncode == 255:
            raise RemoteError(f"cannot reach {remote.label}")
        return _parse_check(cp, remote.label)
    cp = subprocess.run([sys.executable, "-m", "research.train", *res.args, "--check"], cwd=repo_root,
                        capture_output=True, timeout=CHECK_TIMEOUT_S)
    return _parse_check(cp, "this machine")


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout.strip() or None


def commit_check(remote: Remote, repo_root: Path = REPO_ROOT) -> dict:
    """The GPU host's checkout runs train.sh; warn when it is not this checkout's commit."""
    repo = shlex.quote(remote.config.repo_dir)
    try:
        # -uno: only modified tracked files make a run untraceable.
        cp = remote.transport.run(f"git -C {repo} rev-parse HEAD && git -C {repo} status --porcelain -uno",
                                  timeout=30)
    except RemoteError as e:
        return _check("remote_commit", "warn", f"Cannot read the GPU host's commit: {e}")
    lines = cp.stdout.decode("utf-8", "replace").splitlines()
    if cp.returncode != 0 or not lines:
        return _check("remote_commit", "warn", "Cannot read the GPU host's commit "
                      f"({remote.config.repo_dir} is not a git checkout?)")
    remote_head, dirty = lines[0].strip(), [line for line in lines[1:] if line.strip()]
    local_head = _git_head(repo_root)
    problems = []
    if local_head and remote_head != local_head:
        problems.append(f"the GPU host is at {remote_head[:7]}, this checkout at {local_head[:7]}")
    if dirty:
        problems.append(f"the GPU host has {len(dirty)} uncommitted change(s)")
    if problems:
        return _check("remote_commit", "warn", "; ".join(problems).capitalize()
                      + ". Run git pull on the GPU host; the run records the host's commit.")
    return _check("remote_commit", "ok", f"GPU host at {remote_head[:7]}, same as this checkout")


def _submitted_command(res: Resolved, explicit: str) -> str:
    """The job command from --check's fully explicit `scripts/train.sh ...` line."""
    argv = shlex.split(explicit)
    if not argv or argv[0] != ENTRYPOINT:
        raise RemoteError(f"train.sh --check returned an unexpected command: {explicit!r}")
    if res.target == "ssh":
        # Runs on the host from repo_dir (the ssh executor adds remote_prefix()).
        return f"bash {explicit}"
    # Local jobs never go through bash, which Windows lacks.
    local = [sys.executable, "-m", "research.train", *argv[1:]]
    return subprocess.list2cmdline(local) if os.name == "nt" else shlex.join(local)


def check(req: TrainRequest, spec: dict, remote: Optional[Remote], data_base: Optional[Path] = None,
          repo_root: Path = REPO_ROOT, not_configured: str = "") -> dict:
    """Lab-side checks plus train.sh --check on the target. Raises TrainError for a malformed request."""
    res = resolve(req, spec, remote, data_base, not_configured)
    result = {"ok": False, "checks": list(res.checks), "estimated_seconds": None, "command": None,
              "target": res.target, "host": res.host, "run_dir": res.run_dir, "fake": res.fake}
    if _failed(res.checks):
        return result
    if res.target == "ssh":
        assert remote is not None
        result["checks"].append(commit_check(remote, repo_root))
    try:
        out = _target_check(res, remote, repo_root)
        command = _submitted_command(res, out.get("command") or "")
    except (RemoteError, subprocess.SubprocessError, OSError) as e:
        result["checks"].append(_check("train_check", "fail", str(e)))
        return result
    result["checks"] += out["checks"]
    result.update(ok=not _failed(result["checks"]), estimated_seconds=out.get("estimated_seconds"),
                  command=command)
    return result


def start(req: TrainRequest, spec: dict, remote: Optional[Remote], runner: Runner, lock: threading.Lock,
          data_base: Optional[Path] = None, repo_root: Path = REPO_ROOT, not_configured: str = "") -> Job:
    """Check again on the target, then submit. Raises CheckFailed, TrainError or JobConflict."""
    result = check(req, spec, remote, data_base, repo_root, not_configured)
    if not result["ok"]:
        raise CheckFailed(result)
    job_spec = JobSpec(kind=JobKind.TRAIN, cmd=result["command"],
                       executor=Executor.SSH if req.target == "ssh" else Executor.LOCAL,
                       label=req.run_name, run_dir=result["run_dir"], host=result["host"])
    with lock:
        return runner.submit(job_spec)


# --- run files ---

class RunDir(Protocol):
    def read_text(self, name: str) -> Optional[str]: ...
    def read_from(self, name: str, offset: int) -> Optional[bytes]: ...


def _name(name: str) -> str:
    # Confines every read to the run directory: only these names are joined to it.
    if name not in RUN_FILES:
        raise ValueError(f"not a run file: {name}")
    return name


class LocalRunDir:
    def __init__(self, path: str):
        self.path = Path(path)

    def read_text(self, name):
        p = self.path / _name(name)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def read_from(self, name, offset):
        p = self.path / _name(name)
        if not p.is_file():
            return None
        with p.open("rb") as f:
            f.seek(offset)
            return f.read()


class RemoteRunDir:
    def __init__(self, remote: Remote, path: str):
        self.remote, self.path = remote, path

    def read_text(self, name):
        return self.remote.transport.read_text(posixpath.join(self.path, _name(name)))

    def read_from(self, name, offset):
        return self.remote.transport.read_from(posixpath.join(self.path, _name(name)), offset)


def run_dir_of(job: Job, remote: Optional[Remote]) -> RunDir:
    """Raises LookupError for a job without run files, RemoteError if its host is not configured."""
    if job.spec.kind != JobKind.TRAIN or not job.spec.run_dir:
        raise LookupError("not a training run")
    if job.spec.executor == Executor.SSH:
        if remote is None:
            raise RemoteError("GPU host not configured")
        return RemoteRunDir(remote, job.spec.run_dir)
    return LocalRunDir(job.spec.run_dir)


def _json(text: Optional[str]) -> Optional[Any]:
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def read_json(run_dir: RunDir, name: str) -> Optional[Any]:
    """A run file's JSON, or None if it does not exist yet."""
    return _json(run_dir.read_text(name))


def run_state(job: Job, status: Optional[dict]) -> str:
    """What the page shows: status.json's word, corrected by the job's fate.

    A process killed without a chance to write leaves status.json at
    "running". If the job is no longer active, the run ended with unknown
    status, unless the runner itself stopped it (Windows kills without
    signalling, so train.py cannot write "stopped" there).
    """
    word = status.get("status") if isinstance(status, dict) else None
    if word is None:
        return "starting" if job.is_active() else job.status.value
    if word == "running" and not job.is_active():
        return "stopped" if job.status == JobStatus.STOPPED else "ended_unknown"
    return word


def run_status(job: Job, run_dir: RunDir) -> dict:
    status = read_json(run_dir, "status.json")
    return {"job": job.to_dict(), "status": status, "state": run_state(job, status)}


def read_metrics(run_dir: RunDir, offset: int = 0) -> dict:
    """Complete metrics.jsonl lines from a byte offset, and the offset to ask for next."""
    data = run_dir.read_from("metrics.jsonl", max(0, offset))
    if not data:
        return {"metrics": [], "offset": max(0, offset)}
    end = data.rfind(b"\n") + 1  # a last line without a newline is still being written
    metrics = []
    for line in data[:end].splitlines():
        try:
            metrics.append(json.loads(line))
        except ValueError:
            continue
    return {"metrics": metrics, "offset": max(0, offset) + end}


def list_runs(jobs: list[Job], remote: Optional[Remote]) -> list[dict]:
    """Every training job with its status.json and checkpoints.json; one remote call for all ssh runs."""
    train = [j for j in jobs if j.spec.kind == JobKind.TRAIN and j.spec.run_dir]
    files: dict[str, Optional[str]] = {}
    unreachable = None
    remote_paths = [posixpath.join(j.spec.run_dir, name) for j in train if j.spec.executor == Executor.SSH
                    for name in ("status.json", "checkpoints.json")]
    if remote_paths:
        if remote is None:
            unreachable = "GPU host not configured"
        else:
            try:
                files.update(remote.transport.read_texts(remote_paths))
            except RemoteError as e:
                unreachable = str(e)
    out = []
    for job in train:
        if job.spec.executor == Executor.SSH:
            status = _json(files.get(posixpath.join(job.spec.run_dir, "status.json")))
            checkpoints = _json(files.get(posixpath.join(job.spec.run_dir, "checkpoints.json")))
            note = job.unreachable or unreachable
        else:
            local = LocalRunDir(job.spec.run_dir)
            status, checkpoints = read_json(local, "status.json"), read_json(local, "checkpoints.json")
            note = job.unreachable
        count = len(checkpoints["checkpoints"]) if isinstance(checkpoints, dict) else None
        out.append({"job": job.to_dict(), "status": status, "state": run_state(job, status),
                    "checkpoint_count": count, "unreachable": note})
    return out
