"""The GPU host over SSH: configuration, command construction, transports.

This is the only module that builds ssh, rsync or tar-over-ssh command lines.
Everything that touches the GPU host (dataset sync, the SSH job executor,
reading run files back) goes through it, so quoting and connection options
live in one place.

    remote = load_remote()                       # robot/configs/remote.yaml
    remote.ping()                                # (ok, message)
    remote.run(remote.remote_prefix() + " " + shlex.join(cmd))

SSH executor contract: every job command runs as `remote_prefix() + " " +
<command>`, so it starts in repo_dir with the configured venv first on PATH.
A non-interactive ssh shell reads no profile, so without the prefix neither
`python` nor `lerobot-train` resolve to the venv.

Remote paths are POSIX strings. Every value interpolated into a remote shell
command is shlex-quoted here; callers never assemble remote shell strings.

Two transports share one interface. SshTransport is the real one.
LocalTransport treats data_dir as a local directory and performs the same
operations in Python, so the sync flow is testable without a server.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Sequence

import yaml

from lab.core import dataset_manifest
from lab.core.datasets import validate_dataset_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "robot" / "configs" / "remote.yaml"
MANIFEST_SCRIPT = Path(dataset_manifest.__file__).read_bytes()

CONNECT_TIMEOUT_S = 10
INCOMING = ".incoming"
MANIFEST_SUFFIX = ".manifest.json"

# Reads every <name>.manifest.json in a datasets dir; run remotely from stdin.
_LIST_MANIFESTS = b"""
import glob, json, os, sys
out = {}
for path in glob.glob(os.path.join(sys.argv[1], "*.manifest.json")):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        continue
    if isinstance(data, dict):
        name = os.path.basename(path)[:-len(".manifest.json")]
        out[name] = {"fingerprint": data.get("fingerprint"), "synced_at": data.get("synced_at")}
print(json.dumps(out))
"""

# Reads several small text files in one round trip; run remotely from stdin.
_READ_TEXTS = b"""
import json, sys
out = {}
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            out[path] = f.read()
    except OSError:
        out[path] = None
print(json.dumps(out))
"""

Progress = Callable[[int], None]
Log = Callable[[str], None]


class RemoteNotConfigured(RuntimeError):
    """robot/configs/remote.yaml is missing or incomplete."""


class RemoteError(RuntimeError):
    """A remote command failed or the host could not be reached."""


class TransferStopped(RuntimeError):
    """A transfer was stopped on request."""


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    user: str
    repo_dir: str
    data_dir: str
    python: str
    ssh_key: Optional[str] = None
    policy_port: Optional[int] = None
    gpu_count: int = 1
    port: int = 22

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RemoteConfig":
        path = Path(path) if path is not None else DEFAULT_CONFIG
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RemoteNotConfigured(
                f"{path.name} not found: copy robot/configs/remote.yaml.example to "
                "robot/configs/remote.yaml and fill it in (see robot/configs/README.md)") from None
        except (OSError, yaml.YAMLError) as e:
            raise RemoteNotConfigured(f"cannot read {path.name}: {e}") from None
        if not isinstance(raw, dict):
            raise RemoteNotConfigured(f"{path.name} must be a mapping of fields")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RemoteNotConfigured(f"{path.name}: unknown fields {unknown}")
        missing = [k for k in ("host", "user", "repo_dir", "data_dir", "python")
                   if not isinstance(raw.get(k), str) or not raw[k].strip()]
        if missing:
            raise RemoteNotConfigured(f"{path.name}: missing {missing}")
        for key in ("port", "policy_port", "gpu_count"):
            if raw.get(key) is not None and (type(raw[key]) is not int or raw[key] < 1):
                raise RemoteNotConfigured(f"{path.name}: {key} must be a positive integer")
        if raw.get("ssh_key") is not None and not isinstance(raw["ssh_key"], str):
            raise RemoteNotConfigured(f"{path.name}: ssh_key must be a path")
        return cls(**{k: v for k, v in raw.items() if v is not None})

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}" + (f":{self.port}" if self.port != 22 else "")

    def ssh_options(self) -> list[str]:
        """`ssh` and its options, without destination or command (also rsync's -e)."""
        key = ["-i", os.path.expanduser(self.ssh_key)] if self.ssh_key else []
        return ["ssh", *key, "-p", str(self.port), "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}"]

    def ssh_argv(self, remote_command: str | Sequence[str]) -> list[str]:
        """argv running `remote_command` on the host.

        A sequence is one argv and is quoted here. A string is a shell command
        line and must already be quoted; build it from shlex.quote()d parts.
        `--` keeps a command starting with "-" from being read as an option.
        """
        command = remote_command if isinstance(remote_command, str) else shlex.join(remote_command)
        return [*self.ssh_options(), f"{self.user}@{self.host}", "--", command]

    def remote_prefix(self) -> str:
        """Shell prefix for job commands; see the module docstring."""
        bin_dir = posixpath.dirname(self.python)
        return f"cd {shlex.quote(self.repo_dir)} && export PATH={shlex.quote(bin_dir)}:$PATH &&"


# --- transports ---

class Transport(ABC):
    """Operations on the GPU host's filesystem. Paths are the host's paths."""

    @abstractmethod
    def run(self, command: str, *, input: Optional[bytes] = None,
            timeout: Optional[float] = 60) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def ping(self, timeout: float = 6) -> tuple[bool, str]: ...

    @abstractmethod
    def read_text(self, path: str) -> Optional[str]:
        """File content, or None if it does not exist."""

    @abstractmethod
    def read_from(self, path: str, offset: int) -> Optional[bytes]:
        """Bytes from `offset` to the end, or None if the file does not exist."""

    @abstractmethod
    def read_texts(self, paths: Sequence[str]) -> dict[str, Optional[str]]:
        """read_text of several files in one round trip."""

    @abstractmethod
    def write_text(self, path: str, text: str) -> None:
        """Atomic: temp file, then rename."""

    @abstractmethod
    def makedirs(self, path: str) -> None: ...

    @abstractmethod
    def rename(self, src: str, dst: str) -> None:
        """Refuses an existing destination."""

    @abstractmethod
    def probe(self, datasets_dir: str, name: str) -> tuple[str, Optional[str]]:
        """("manifest", text) | ("orphan" | "incoming" | "absent", None) for one dataset.

        orphan: <name> exists but <name>.manifest.json does not.
        """

    @abstractmethod
    def list_manifests(self, datasets_dir: str) -> dict[str, dict]:
        """name -> {"fingerprint", "synced_at"} for every <name>.manifest.json."""

    @abstractmethod
    def manifest(self, root: str) -> dict:
        """dataset_manifest.describe() of a directory on the host."""

    @abstractmethod
    def transfer(self, local_root: Path, dest: str, progress: Progress, log: Log) -> str:
        """Make `dest` (inside .incoming/) an exact copy of local_root. Returns the method used."""

    @abstractmethod
    def terminate(self) -> None:
        """Stop a running transfer from a signal handler."""


def _require_incoming(dest: str) -> None:
    # The only paths ever deleted or overwritten wholesale on the host.
    p = PurePosixPath(dest.replace("\\", "/"))
    if p.parent.name != INCOMING:
        raise ValueError(f"refusing to replace {dest}: not inside {INCOMING}/")
    validate_dataset_name(p.name)


class SshTransport(Transport):
    def __init__(self, config: RemoteConfig):
        self.config = config
        self._procs: list[subprocess.Popen] = []
        self._stopped = False

    def _argv(self, command: str) -> list[str]:
        return self.config.ssh_argv(command)

    def run(self, command, *, input=None, timeout=60):
        try:
            return subprocess.run(self._argv(command), input=input, capture_output=True, timeout=timeout,
                                  **({} if input is not None else {"stdin": subprocess.DEVNULL}))
        except subprocess.TimeoutExpired:
            raise RemoteError(f"no answer from {self.config.label} within {timeout:.0f} s") from None
        except OSError as e:
            raise RemoteError(f"cannot run ssh: {e}") from None

    def _check(self, cp: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
        if cp.returncode != 0:
            err = cp.stderr.decode("utf-8", "replace").strip().splitlines()
            tail = err[-1] if err else "no message"
            if cp.returncode == 255:
                raise RemoteError(f"cannot reach {self.config.label}: {tail}")
            raise RemoteError(f"{what} failed on {self.config.label} (exit {cp.returncode}): {tail}")
        return cp

    def ping(self, timeout=6):
        try:
            self._check(self.run("true", timeout=timeout), "ping")
        except RemoteError as e:
            return False, str(e)
        return True, f"connected to {self.config.label}"

    def read_text(self, path):
        q = shlex.quote(path)
        cp = self.run(f"if [ -f {q} ]; then cat {q}; else exit 3; fi")
        if cp.returncode == 3:
            return None
        return self._check(cp, f"read {path}").stdout.decode("utf-8")

    def read_from(self, path, offset):
        q = shlex.quote(path)
        cp = self.run(f"if [ -f {q} ]; then tail -c +{int(offset) + 1} {q}; else exit 3; fi")
        if cp.returncode == 3:
            return None
        return self._check(cp, f"read {path}").stdout

    def read_texts(self, paths):
        if not paths:
            return {}
        cp = self._check(self.run(f"{shlex.quote(self.config.python)} - {shlex.join(paths)}",
                                  input=_READ_TEXTS), "read files")
        return _json_object(cp.stdout, "read files")

    def write_text(self, path, text):
        q = shlex.quote(path)
        self._check(self.run(f'tmp={q}.tmp.$$ && cat > "$tmp" && mv -f "$tmp" {q}',
                             input=text.encode("utf-8")), f"write {path}")

    def makedirs(self, path):
        self._check(self.run(f"mkdir -p {shlex.quote(path)}"), f"mkdir {path}")

    def rename(self, src, dst):
        qs, qd = shlex.quote(src), shlex.quote(dst)
        self._check(self.run(f"if [ -e {qd} ]; then echo 'destination exists' >&2; exit 4; fi; mv -- {qs} {qd}"),
                    f"rename {src}")

    def probe(self, datasets_dir, name):
        validate_dataset_name(name)
        m = shlex.quote(posixpath.join(datasets_dir, name + MANIFEST_SUFFIX))
        r = shlex.quote(posixpath.join(datasets_dir, name))
        i = shlex.quote(posixpath.join(datasets_dir, INCOMING, name))
        out = self._check(self.run(
            f"if [ -f {m} ]; then echo manifest; cat {m}; elif [ -e {r} ]; then echo orphan; "
            f"elif [ -d {i} ]; then echo incoming; else echo absent; fi"), "probe").stdout.decode("utf-8")
        kind, _, rest = out.partition("\n")
        return (kind, rest) if kind == "manifest" else (kind.strip(), None)

    def list_manifests(self, datasets_dir):
        cp = self._check(self.run(f"{shlex.quote(self.config.python)} - {shlex.quote(datasets_dir)}",
                                  input=_LIST_MANIFESTS), "list manifests")
        return _json_object(cp.stdout, "list manifests")

    def manifest(self, root):
        # The configured interpreter, fed this checkout's dataset_manifest.py on
        # stdin: the host's own checkout version does not matter.
        cp = self._check(self.run(f"{shlex.quote(self.config.python)} - {shlex.quote(root)}",
                                  input=MANIFEST_SCRIPT, timeout=None), f"manifest of {root}")
        return _json_object(cp.stdout, "manifest")

    def remote_has(self, program: str) -> bool:
        cp = self._check(self.run(f"command -v {shlex.quote(program)} >/dev/null 2>&1 && echo yes || echo no"),
                         f"look for {program}")
        return cp.stdout.decode().strip() == "yes"

    def transfer(self, local_root, dest, progress, log):
        _require_incoming(dest)
        # rsync with a Windows path ("C:\...") would read "C" as a host name.
        local = os.name != "nt" and shutil.which("rsync") is not None
        remote = self.remote_has("rsync")
        if local and remote:
            self._rsync(local_root, dest, progress)
            return "rsync"
        missing = " and ".join(side for side, ok in (("this machine", local), ("the GPU host", remote)) if not ok)
        log(f"rsync is not available on {missing}; falling back to tar over ssh "
            "(not resumable: an interrupted transfer starts over)")
        self._tar(local_root, dest, progress)
        return "tar"

    def rsync_argv(self, local_root: Path, dest: str) -> list[str]:
        # -s: the destination path reaches the remote rsync without a second
        # round of shell word splitting. --delete only ever acts inside dest
        # (.incoming/<name>), dropping leftovers of an earlier attempt.
        return ["rsync", "-a", "--partial", "--delete", "-s", "--info=progress2",
                "-e", shlex.join(self.config.ssh_options()),
                f"{local_root.as_posix().rstrip('/')}/", f"{self.config.user}@{self.config.host}:{dest}/"]

    def _rsync(self, local_root: Path, dest: str, progress: Progress) -> None:
        self.makedirs(posixpath.dirname(dest))
        proc = self._start(self.rsync_argv(local_root, dest), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail: deque[str] = deque(maxlen=10)
        percent = re.compile(rb"\s(\d{1,3})%\s")
        buf = b""
        while chunk := proc.stdout.read1(65536):
            buf += chunk
            *pieces, buf = re.split(rb"[\r\n]", buf)
            for piece in pieces:
                if m := percent.search(piece):
                    progress(min(100, int(m[1])))
                elif piece.strip():
                    tail.append(piece.decode("utf-8", "replace").strip())
        rc = proc.wait()
        self._finish(proc)
        if rc != 0:
            raise RemoteError(f"rsync exited {rc}: {' | '.join(tail) or 'no message'}")

    def _tar(self, local_root: Path, dest: str, progress: Progress) -> None:
        total = max(1, sum(row[1] for row in dataset_manifest.listing(str(local_root))))
        qd = shlex.quote(dest)
        reader = self._start(["tar", "-C", str(local_root), "-cf", "-", "."], stdout=subprocess.PIPE)
        writer = self._start(self._argv(f"rm -rf -- {qd} && mkdir -p {qd} && tar -C {qd} -xf -"),
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        errors: list[bytes] = []
        drain = threading.Thread(target=lambda: errors.append(writer.stderr.read()), daemon=True)
        drain.start()
        sent = 0
        try:
            while chunk := reader.stdout.read(1 << 20):
                writer.stdin.write(chunk)
                sent += len(chunk)
                progress(min(99, sent * 100 // total))  # tar headers make the stream a bit larger
        except (BrokenPipeError, OSError):
            pass  # the remote side died; its exit code says why
        finally:
            try:
                writer.stdin.close()
            except OSError:
                pass
        rc_reader, rc_writer = reader.wait(), writer.wait()
        drain.join(timeout=5)
        self._finish(reader, writer)
        if rc_reader != 0 or rc_writer != 0:
            message = b"".join(errors).decode("utf-8", "replace").strip().splitlines()
            raise RemoteError(f"tar over ssh failed (local tar {rc_reader}, remote {rc_writer}): "
                              f"{message[-1] if message else 'no message'}")
        progress(100)

    def _start(self, argv: list[str], **kwargs) -> subprocess.Popen:
        if self._stopped:
            raise TransferStopped("stopped")
        try:
            proc = subprocess.Popen(argv, **kwargs)
        except OSError as e:
            raise RemoteError(f"cannot run {argv[0]}: {e}") from None
        self._procs.append(proc)
        return proc

    def _finish(self, *procs: subprocess.Popen) -> None:
        for proc in procs:
            if proc in self._procs:
                self._procs.remove(proc)
        if self._stopped:
            raise TransferStopped("stopped")

    def terminate(self):
        self._stopped = True
        for proc in list(self._procs):
            if proc.poll() is None:
                proc.terminate()


class LocalTransport(Transport):
    """The GPU host's side as local directories, for tests and dry runs.

    Same semantics as SshTransport: atomic writes, refused renames, a
    resumable transfer that skips files whose size and mtime already match
    (rsync -a's quick check) and drops extraneous ones (--delete).
    The manifest runs dataset_manifest.py from stdin in a fresh interpreter,
    exactly as on the host.
    """

    def __init__(self):
        self._stopped = False
        self.transferred: list[str] = []  # relpaths copied by the last transfer()

    def run(self, command, *, input=None, timeout=60):
        return subprocess.run(command, shell=True, input=input, capture_output=True, timeout=timeout)

    def ping(self, timeout=6):
        return True, "local transport (no SSH)"

    def read_text(self, path):
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def read_from(self, path, offset):
        p = Path(path)
        if not p.is_file():
            return None
        with p.open("rb") as f:
            f.seek(offset)
            return f.read()

    def read_texts(self, paths):
        return {path: self.read_text(path) for path in paths}

    def write_text(self, path, text):
        p = Path(path)
        tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)

    def makedirs(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def rename(self, src, dst):
        if os.path.lexists(dst):
            raise RemoteError(f"rename {src} failed: destination exists")
        os.rename(src, dst)

    def probe(self, datasets_dir, name):
        validate_dataset_name(name)
        base = Path(datasets_dir)
        manifest = base / (name + MANIFEST_SUFFIX)
        if manifest.is_file():
            return "manifest", manifest.read_text(encoding="utf-8")
        if os.path.lexists(base / name):
            return "orphan", None
        if (base / INCOMING / name).is_dir():
            return "incoming", None
        return "absent", None

    def list_manifests(self, datasets_dir):
        cp = subprocess.run([sys.executable, "-", datasets_dir], input=_LIST_MANIFESTS, capture_output=True)
        if cp.returncode != 0:
            raise RemoteError(f"list manifests failed: {cp.stderr.decode(errors='replace').strip()}")
        return _json_object(cp.stdout, "list manifests")

    def manifest(self, root):
        cp = subprocess.run([sys.executable, "-", root], input=MANIFEST_SCRIPT, capture_output=True)
        if cp.returncode != 0:
            raise RemoteError(f"manifest of {root} failed: {cp.stderr.decode(errors='replace').strip()}")
        return _json_object(cp.stdout, "manifest")

    def transfer(self, local_root, dest, progress, log):
        _require_incoming(dest)
        dest_root = Path(dest)
        dest_root.mkdir(parents=True, exist_ok=True)
        rows = dataset_manifest.listing(str(local_root))
        total = max(1, sum(size for _, size, _ in rows))
        done = 0
        self.transferred = []
        wanted = set()
        for rel, size, mtime_ns in rows:
            if self._stopped:
                raise TransferStopped("stopped")
            wanted.add(rel)
            target = dest_root / rel
            st = target.stat() if target.is_file() else None
            if st is None or st.st_size != size or st.st_mtime_ns != mtime_ns:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                shutil.copy2(local_root / rel, partial)
                os.replace(partial, target)
                self.transferred.append(rel)
            done += size
            progress(done * 100 // total)
        for path in sorted(dest_root.rglob("*"), reverse=True):
            rel = path.relative_to(dest_root).as_posix()
            if path.is_file() and rel not in wanted:
                path.unlink()
        progress(100)
        return "local"

    def terminate(self):
        self._stopped = True


def _json_object(raw: bytes, what: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError:
        raise RemoteError(f"{what}: unreadable output") from None
    if not isinstance(data, dict):
        raise RemoteError(f"{what}: expected a JSON object")
    return data


# --- facade ---

class Remote:
    """A configured GPU host and the transport that reaches it."""

    def __init__(self, config: RemoteConfig, transport: Optional[Transport] = None):
        self.config = config
        self.transport = transport if transport is not None else SshTransport(config)

    @property
    def label(self) -> str:
        return self.config.label

    def ssh_argv(self, remote_command: str | Sequence[str]) -> list[str]:
        return self.config.ssh_argv(remote_command)

    def remote_prefix(self) -> str:
        return self.config.remote_prefix()

    def run(self, remote_command: str, timeout: Optional[float] = 60) -> subprocess.CompletedProcess:
        return self.transport.run(remote_command, timeout=timeout)

    def ping(self, timeout: float = 6) -> tuple[bool, str]:
        return self.transport.ping(timeout)

    def remote_datasets_dir(self) -> str:
        return posixpath.join(self.config.data_dir, "datasets")

    def remote_dataset_root(self, name: str) -> str:
        return posixpath.join(self.remote_datasets_dir(), validate_dataset_name(name))

    def remote_incoming_root(self, name: str) -> str:
        return posixpath.join(self.remote_datasets_dir(), INCOMING, validate_dataset_name(name))

    def remote_manifest_path(self, name: str) -> str:
        return self.remote_dataset_root(name) + MANIFEST_SUFFIX

    def remote_runs_dir(self) -> str:
        return posixpath.join(self.config.data_dir, "runs")

    def remote_logs_dir(self) -> str:
        return posixpath.join(self.config.data_dir, "logs")


def load_remote(config_path: Optional[Path] = None, transport: str = "ssh") -> Remote:
    """Raises RemoteNotConfigured. transport "local" is for tests: data_dir is a local directory."""
    config = RemoteConfig.load(config_path)
    if transport == "ssh":
        return Remote(config)
    if transport == "local":
        return Remote(config, LocalTransport())
    raise ValueError(f"unknown transport {transport!r}")
