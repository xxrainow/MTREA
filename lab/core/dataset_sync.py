"""Send a training-ready dataset to the GPU host, verified by content fingerprint.

    python -m lab.core.dataset_sync --name=<name> [--data-base=<data dir>]

Submitted by lab as a JobKind.SYNC job, so the job log, stop and Jobs page
apply. Steps, stopping at the first failure:
1. Resolve the ready root (see ready_root) and its manifest (cached).
2. Look at the host. <datasets>/<name>.manifest.json with the same
   fingerprint: already synced, exit 0. A different fingerprint: refused,
   exit 2. Datasets on the host are immutable; a changed dataset gets a new
   name.
3. Transfer into <datasets>/.incoming/<name>/ (rsync, resumable; tar over
   ssh when either side lacks rsync).
4. Hash the copy on the host; it must equal the local manifest.
5. Move .incoming/<name> to <name>, then write <name>.manifest.json beside it.

stdout carries only parseable lines; everything else goes to stderr:
    STATUS started
    PROGRESS phase=hash|transfer|verify percent=<n>
    SYNCED fingerprint=<sha256> path=<remote root> already=true|false
    STATUS done|failed|stopped exit=<code>
Exit codes: 0 synced or already synced, 1 failed, 2 refused (not sendable,
not configured, different remote copy), 143 stopped. A stopped transfer
resumes from .incoming on the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from lab.core import dataset_manifest, paths
from lab.core.datasets import Backend, list_datasets, validate_dataset_name
from lab.core.remote import Remote, RemoteError, RemoteNotConfigured, TransferStopped, load_remote

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_VERSION = 1

# pi05 normalizes these with QUANTILES; the same rule research/train.py checks.
# Read here directly because lab never imports research/.
QUANTILE_FEATURES = ("observation.state", "action")

STATES = ("absent", "incoming", "synced", "different", "unknown")


class NotSendable(RuntimeError):
    """The dataset cannot be sent; the message is for the user."""


@dataclass(frozen=True)
class Sendability:
    sendable: bool
    reason: Optional[str]
    root: Optional[Path]


def _read_json(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotSendable(f"{what} is missing") from None
    except (OSError, ValueError) as e:
        raise NotSendable(f"cannot read {what}: {e}") from None
    if not isinstance(data, dict):
        raise NotSendable(f"cannot read {what}: expected a JSON object")
    return data


def ready_root(name: str, data_base: Optional[Path] = None) -> Path:
    """The directory to send for a real dataset; raises NotSendable with a plain reason.

    data/datasets/<name> (experiments/prepare_dataset.py's recomputed copy) if
    it exists, else data/raw/<name> when its stats already carry the quantiles
    (prepare_dataset uses the raw root as-is then).
    """
    validate_dataset_name(name)
    base = paths.root(data_base)
    copy, raw = paths.dataset_dir(name, base), paths.raw_dir(base) / name
    if copy.is_dir() and not copy.is_symlink():
        root, where = copy, f"data/datasets/{name}"
    elif raw.is_dir() and not raw.is_symlink():
        root, where = raw, f"data/raw/{name}"
    else:
        raise NotSendable("dataset not found")
    info = _read_json(root / "meta" / "info.json", f"{where}/meta/info.json")
    if info.get("fake") is True:
        raise NotSendable("not sent: fake")
    try:
        stats = _read_json(root / "meta" / "stats.json", f"{where}/meta/stats.json")
    except NotSendable as e:
        raise NotSendable(f"{e}; run experiments/prepare_dataset.py first") from None
    missing = [f for f in QUANTILE_FEATURES
               if not isinstance(stats.get(f), dict) or not {"q01", "q99"} <= stats[f].keys()]
    if missing:
        raise NotSendable(f"{where}/meta/stats.json lacks q01/q99 for {', '.join(missing)}; "
                          "run experiments/prepare_dataset.py first")
    return root


def sendability(backend: Backend, name: str, data_base: Optional[Path] = None) -> Sendability:
    if backend == "fake":
        return Sendability(False, "not sent: fake", None)
    try:
        return Sendability(True, None, ready_root(name, data_base))
    except NotSendable as e:
        return Sendability(False, str(e), None)


def _cache(name: str, data_base: Optional[Path]) -> str:
    return str(paths.manifest_cache("real", name, data_base))


def remote_state(remote: Optional[Remote], backend: Backend, name: str,
                 data_base: Optional[Path] = None, not_configured: str = "") -> dict:
    """What the Datasets page shows for one dataset. Never hashes; at most one remote call."""
    validate_dataset_name(name)
    send = sendability(backend, name, data_base)
    local = (dataset_manifest.valid_cached_manifest(str(send.root), _cache(name, data_base))
             if send.root is not None else None)
    out = {"sendable": send.sendable, "reason": send.reason, "state": "unknown", "message": "",
           "remote_path": None, "local_fingerprint": local["fingerprint"] if local else None,
           "remote_fingerprint": None, "synced_at": None}
    if remote is None:
        out["message"] = not_configured or "GPU host not configured"
        return out
    out["remote_path"] = remote.remote_dataset_root(name)
    try:
        kind, text = remote.transport.probe(remote.remote_datasets_dir(), name)
    except RemoteError as e:
        out["message"] = str(e)
        return out
    if kind == "manifest":
        try:
            manifest = json.loads(text or "")
            out["remote_fingerprint"] = manifest["fingerprint"]
            out["synced_at"] = manifest.get("synced_at")
        except (ValueError, KeyError, TypeError):
            out["message"] = "the remote manifest is unreadable"
            return out
        if local is None:
            out["message"] = ("the local fingerprint has not been computed on this machine; "
                              "Sync computes it and checks the remote copy")
        else:
            out["state"] = "synced" if local["fingerprint"] == out["remote_fingerprint"] else "different"
    elif kind == "orphan":
        out["message"] = "the remote directory exists without a manifest; Sync checks it"
    elif kind in ("incoming", "absent"):
        out["state"] = kind
    else:
        out["message"] = f"unexpected answer from the GPU host: {kind!r}"
    return out


def remote_markers(remote: Optional[Remote], data_base: Optional[Path] = None) -> dict[str, str]:
    """dataset id -> synced | different | not_synced | not_sent_fake | unknown, with one remote call."""
    datasets = list_datasets(data_base)
    manifests: Optional[dict] = None
    if remote is not None:
        try:
            manifests = remote.transport.list_manifests(remote.remote_datasets_dir())
        except RemoteError:
            manifests = None
    out = {}
    for d in datasets:
        if d.backend == "fake":
            out[d.id] = "not_sent_fake"
            continue
        if manifests is None:
            out[d.id] = "unknown"
            continue
        entry = manifests.get(d.name)
        if entry is None:
            out[d.id] = "not_synced"
            continue
        send = sendability("real", d.name, data_base)
        local = (dataset_manifest.valid_cached_manifest(str(send.root), _cache(d.name, data_base))
                 if send.root is not None else None)
        if local is None:
            out[d.id] = "unknown"
        else:
            out[d.id] = "synced" if local["fingerprint"] == entry.get("fingerprint") else "different"
    return out


# --- the job ---

def _git_commit() -> Optional[str]:
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                            timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout.strip() or None


def _say(line: str) -> None:
    print(line, flush=True)


def _err(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class _Progress:
    """PROGRESS lines, one per whole-percent change."""

    def __init__(self, phase: str, stop: Callable[[], bool]):
        self.phase, self.stop, self.last = phase, stop, -1

    def percent(self, n: int) -> None:
        if self.stop():
            raise TransferStopped("stopped")
        if n != self.last:
            self.last = n
            _say(f"PROGRESS phase={self.phase} percent={n}")

    def bytes(self, done: int, total: int) -> None:
        self.percent(100 if total == 0 else done * 100 // total)


def _differences(local: list, remote: list, limit: int = 5) -> list[str]:
    a, b = {e[0]: e for e in local}, {e[0]: e for e in remote}
    out = [f"missing on the GPU host: {p}" for p in sorted(a.keys() - b.keys())]
    out += [f"extra on the GPU host: {p}" for p in sorted(b.keys() - a.keys())]
    out += [f"content differs: {p}" for p in sorted(a.keys() & b.keys()) if a[p] != b[p]]
    return out[:limit]


def sync(name: str, remote: Remote, data_base: Optional[Path] = None,
         stopped: Callable[[], bool] = lambda: False) -> int:
    """Steps 1-5 of the module docstring. Returns the exit code; prints the protocol lines."""
    try:
        root = ready_root(name, data_base)
    except NotSendable as e:
        _err(f"error: {name} is not sendable: {e}")
        return 2

    _err(f"local: {root}")
    local = dataset_manifest.cached_manifest(str(root), _cache(name, data_base),
                                             _Progress("hash", stopped).bytes)
    fingerprint = local["fingerprint"]
    _err(f"local fingerprint {fingerprint} ({len(local['files'])} files, {local['total_bytes']} bytes)")

    datasets_dir = remote.remote_datasets_dir()
    final, incoming = remote.remote_dataset_root(name), remote.remote_incoming_root(name)
    manifest_path = remote.remote_manifest_path(name)
    kind, text = remote.transport.probe(datasets_dir, name)
    if kind == "manifest":
        try:
            remote_fp = json.loads(text or "")["fingerprint"]
        except (ValueError, KeyError, TypeError):
            _err(f"error: {manifest_path} is unreadable; inspect it on the GPU host")
            return 2
        if remote_fp == fingerprint:
            _err(f"already synced: {final}")
            _say(f"SYNCED fingerprint={fingerprint} path={final} already=true")
            return 0
        _err(f"error: {final} already exists on the GPU host with fingerprint {remote_fp}, "
             f"not {fingerprint}. Datasets there are immutable: give the changed dataset a new name.")
        return 2

    if kind == "orphan":
        # A previous run moved the copy into place but did not write the manifest.
        _err(f"{final} exists without a manifest; checking it")
        found = remote.transport.manifest(final)
        if found.get("fingerprint") != fingerprint:
            _err(f"error: {final} exists on the GPU host without a manifest and differs from {root}; "
                 "inspect it there, or give the dataset a new name")
            return 2
        method = "existing"
    else:
        if kind == "incoming":
            _err(f"resuming from {incoming}")
        remote.transport.makedirs(posix_parent(incoming))
        method = remote.transport.transfer(root, incoming, _Progress("transfer", stopped).percent, _err)
        _err(f"transferred with {method}")

        verify = _Progress("verify", stopped)
        verify.percent(0)
        found = remote.transport.manifest(incoming)
        if found.get("files") != local["files"] or found.get("fingerprint") != fingerprint:
            _err(f"error: the copy on the GPU host does not match {root}; nothing was moved")
            for line in _differences(local["files"], found.get("files") or []):
                _err(f"  {line}")
            return 1
        verify.percent(100)
        remote.transport.rename(incoming, final)

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    record = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "fingerprint": fingerprint,
        "total_bytes": local["total_bytes"],
        "file_count": len(local["files"]),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "source_path": str(root),
        "source_host": socket.gethostname(),
        "git_commit": _git_commit(),
        "method": method,
        "synced_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": local["files"],
    }
    remote.transport.write_text(manifest_path, json.dumps(record, indent=2) + "\n")
    _err(f"synced: {final}")
    _say(f"SYNCED fingerprint={fingerprint} path={final} already=false")
    return 0


def posix_parent(path: str) -> str:
    return path.rsplit("/", 1)[0]


def sync_command(name: str, data_base: Path, remote_config: Optional[Path] = None,
                 transport: str = "ssh") -> str:
    """The JobSpec.cmd for one sync, runnable as-is in this machine's shell."""
    argv = [sys.executable, "-m", "lab.core.dataset_sync", f"--name={validate_dataset_name(name)}",
            f"--data-base={data_base}"]
    if remote_config is not None:
        argv.append(f"--remote-config={remote_config}")
    if transport != "ssh":
        argv.append(f"--transport={transport}")
    # LocalExecutor runs the command through the platform shell.
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def _name(value: str) -> str:
    try:
        return validate_dataset_name(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid dataset name {value!r}") from None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lab.core.dataset_sync", description=__doc__.split("\n")[0])
    p.add_argument("--name", required=True, type=_name, help="dataset name under data/raw/")
    p.add_argument("--data-base", type=Path, default=None, help="local data/ root (default: the configured one)")
    p.add_argument("--remote-config", type=Path, default=None, help="default robot/configs/remote.yaml")
    p.add_argument("--transport", choices=["ssh", "local"], default="ssh",
                   help="local: data_dir is a local directory (tests)")
    args = p.parse_args(argv)

    _say("STATUS started")
    try:
        remote = load_remote(args.remote_config, args.transport)
    except RemoteNotConfigured as e:
        _err(f"error: {e}")
        _say("STATUS failed exit=2")
        return 2

    stop = {"requested": False}

    def on_signal(signum, _frame):
        stop["requested"] = True
        remote.transport.terminate()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = sync(args.name, remote, args.data_base, lambda: stop["requested"])
    except TransferStopped:
        code = 143
    except (RemoteError, dataset_manifest.ManifestError, OSError) as e:
        code = 143 if stop["requested"] else 1
        if code == 1:
            _err(f"error: {e}")
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    if stop["requested"] and code != 0:
        code = 143
    status = {0: "done", 143: "stopped"}.get(code, "failed")
    if status == "stopped":
        _err("stopped; a later run resumes from .incoming")
    _say(f"STATUS {status} exit={code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
