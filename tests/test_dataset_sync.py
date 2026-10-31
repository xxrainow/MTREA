"""Dataset sync to the GPU host: manifest, sendability, the sync job, SSH command
construction and the API. No network: LocalTransport stands in for the host,
with data_dir a tmp directory."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.core import dataset_manifest, dataset_sync, paths
from lab.core.dataset_sync import NotSendable, ready_root, remote_state, sendability, sync
from lab.core.datasets import validate_dataset_name
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor, JobKind, JobSpec
from lab.core.remote import (INCOMING, LocalTransport, Remote, RemoteConfig, RemoteNotConfigured,
                             SshTransport, load_remote)
from lab.core.runner import JobConflict, Runner
from lab.core.store import JobStore
from lab.server.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANTILES = {f: {"mean": [0.0], "q01": [-1.0], "q99": [1.0]} for f in ("observation.state", "action")}
NO_QUANTILES = {f: {"mean": [0.0], "std": [1.0]} for f in ("observation.state", "action")}


def write_dataset(root: Path, stats=QUANTILES, info=None) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "wrist").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        info or {"total_episodes": 2, "total_frames": 60, "fps": 30}), encoding="utf-8")
    if stats is not None:
        (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "tasks": ["Pick."]}) + "\n" for i in range(2)), encoding="utf-8")
    (root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(os.urandom(50_000))
    (root / "videos" / "wrist" / "episode_000000.mp4").write_bytes(os.urandom(200_000))
    (root / "videos" / "wrist" / "episode_000001.mp4").write_bytes(os.urandom(150_000))
    return root


@pytest.fixture
def base(tmp_path) -> Path:
    b = tmp_path / "data"
    paths.ensure_dirs(b)
    return b


@pytest.fixture
def config_file(tmp_path) -> Path:
    path = tmp_path / "remote.yaml"
    path.write_text(f"host: gpu.tailnet.ts.net\nuser: lab\nrepo_dir: /home/lab/MTREA\n"
                    f"data_dir: {(tmp_path / 'gpu').as_posix()}\npython: {sys.executable.replace(os.sep, '/')}\n",
                    encoding="utf-8")
    return path


@pytest.fixture
def remote(config_file) -> Remote:
    return load_remote(config_file, "local")


def raw(base: Path, name: str = "pick_v1") -> Path:
    return paths.raw_dir(base) / name


def run_main(base, config_file, name="pick_v1", capsys=None) -> tuple[int, list[str], str]:
    rc = dataset_sync.main([f"--name={name}", f"--data-base={base}", f"--remote-config={config_file}",
                            "--transport=local"])
    out = capsys.readouterr() if capsys else None
    return rc, out.out.splitlines() if out else [], out.err if out else ""


# --- manifest ---

def test_manifest_is_deterministic_and_order_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, names in ((a, ["x", "y/z", "y/a"]), (b, ["y/a", "x", "y/z"])):
        for n in names:
            (root / n).parent.mkdir(parents=True, exist_ok=True)
            (root / n).write_bytes(n.encode() * 100)
    ma = dataset_manifest.manifest(str(a))
    assert ma == dataset_manifest.manifest(str(a)) == dataset_manifest.manifest(str(b))
    assert [e[0] for e in ma] == ["x", "y/a", "y/z"]
    assert dataset_manifest.fingerprint(ma) == dataset_manifest.fingerprint(dataset_manifest.manifest(str(b)))


def test_fingerprint_changes_with_any_byte(tmp_path):
    root = write_dataset(tmp_path / "d")
    before = dataset_manifest.fingerprint(dataset_manifest.manifest(str(root)))
    video = root / "videos" / "wrist" / "episode_000001.mp4"
    data = bytearray(video.read_bytes())
    data[len(data) // 2] ^= 1  # same size, one bit
    video.write_bytes(bytes(data))
    assert dataset_manifest.fingerprint(dataset_manifest.manifest(str(root))) != before


def test_renaming_a_file_changes_the_fingerprint(tmp_path):
    root = write_dataset(tmp_path / "d")
    before = dataset_manifest.fingerprint(dataset_manifest.manifest(str(root)))
    (root / "meta" / "episodes.jsonl").rename(root / "meta" / "episodes2.jsonl")
    assert dataset_manifest.fingerprint(dataset_manifest.manifest(str(root))) != before


def test_symlinks_are_refused(tmp_path):
    root = write_dataset(tmp_path / "d")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    try:
        (root / "data" / "link.bin").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not permitted here")
    with pytest.raises(dataset_manifest.ManifestError, match="symlink"):
        dataset_manifest.manifest(str(root))


def test_standalone_from_stdin_matches_in_process(tmp_path):
    root = write_dataset(tmp_path / "d")
    # -I: no repo on sys.path, no site-packages, no PYTHON* env; the host runs it the same way.
    cp = subprocess.run([sys.executable, "-I", "-", str(root)], input=Path(dataset_manifest.__file__).read_bytes(),
                        capture_output=True, cwd=tmp_path, check=True)
    remote = json.loads(cp.stdout)
    local = dataset_manifest.describe(str(root), dataset_manifest.manifest(str(root)))
    assert remote["files"] == local["files"]
    assert remote["fingerprint"] == local["fingerprint"]
    assert remote["total_bytes"] == local["total_bytes"] == 400_000 + sum(
        (root / "meta" / n).stat().st_size for n in ("info.json", "stats.json", "episodes.jsonl"))


def test_manifest_cache_is_reused_until_the_listing_changes(tmp_path, monkeypatch):
    root = write_dataset(tmp_path / "d")
    cache = str(tmp_path / "cache" / "real_d.json")
    first = dataset_manifest.cached_manifest(str(root), cache)
    assert dataset_manifest.valid_cached_manifest(str(root), cache) == first

    def no_hashing(*a, **k):
        raise AssertionError("hashed despite a valid cache")
    monkeypatch.setattr(dataset_manifest, "manifest", no_hashing)
    assert dataset_manifest.cached_manifest(str(root), cache) == first

    video = root / "videos" / "wrist" / "episode_000000.mp4"
    st = video.stat()
    os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert dataset_manifest.valid_cached_manifest(str(root), cache) is None
    monkeypatch.undo()
    assert dataset_manifest.cached_manifest(str(root), cache)["fingerprint"] == first["fingerprint"]


# --- which dataset is sendable ---

def test_ready_root_prefers_the_prepared_copy(base):
    write_dataset(raw(base), stats=NO_QUANTILES)
    copy = write_dataset(paths.dataset_dir("pick_v1", base))
    assert ready_root("pick_v1", base) == copy


def test_raw_with_quantiles_is_sendable(base):
    root = write_dataset(raw(base))
    assert ready_root("pick_v1", base) == root
    assert sendability("real", "pick_v1", base).sendable


def test_raw_without_quantiles_is_refused(base):
    write_dataset(raw(base), stats=NO_QUANTILES)
    with pytest.raises(NotSendable, match="q01/q99.*experiments/prepare_dataset.py first"):
        ready_root("pick_v1", base)


def test_raw_without_stats_is_refused(base):
    write_dataset(raw(base), stats=None)
    send = sendability("real", "pick_v1", base)
    assert not send.sendable and "stats.json is missing" in send.reason and "prepare_dataset" in send.reason


def test_fake_is_refused(base):
    write_dataset(base / "fake" / "collection" / "raw" / "pick_v1")
    assert sendability("fake", "pick_v1", base).reason == "not sent: fake"
    write_dataset(raw(base, "fake_in_real"), info={"total_episodes": 1, "fake": True})
    assert sendability("real", "fake_in_real", base).reason == "not sent: fake"


def test_unreadable_metadata_is_refused(base):
    root = write_dataset(raw(base))
    (root / "meta" / "info.json").write_text("{", encoding="utf-8")
    assert "cannot read data/raw/pick_v1/meta/info.json" in sendability("real", "pick_v1", base).reason


# --- the sync job ---

PROTOCOL = r"^(STATUS (started|(done|failed|stopped) exit=\d+)|PROGRESS phase=(hash|transfer|verify) percent=\d+|SYNCED .+)$"


def test_sync_happy_path(base, config_file, tmp_path, capsys):
    write_dataset(raw(base))
    rc, out, err = run_main(base, config_file, capsys=capsys)
    assert rc == 0, err
    assert all(re.match(PROTOCOL, line) for line in out), out
    assert out[0] == "STATUS started" and out[-1] == "STATUS done exit=0"
    assert {"PROGRESS phase=hash percent=100", "PROGRESS phase=transfer percent=100",
            "PROGRESS phase=verify percent=100"} <= set(out)
    assert out[-2].startswith("SYNCED fingerprint=") and out[-2].endswith("already=false")

    datasets = tmp_path / "gpu" / "datasets"
    assert sorted(p.name for p in datasets.iterdir()) == [INCOMING, "pick_v1", "pick_v1.manifest.json"]
    assert list((datasets / INCOMING).iterdir()) == []
    local = dataset_manifest.describe(str(raw(base)), dataset_manifest.manifest(str(raw(base))))
    assert dataset_manifest.manifest(str(datasets / "pick_v1")) == local["files"]

    record = json.loads((datasets / "pick_v1.manifest.json").read_text(encoding="utf-8"))
    assert record["fingerprint"] == local["fingerprint"] and record["files"] == local["files"]
    assert (record["episodes"], record["frames"], record["file_count"]) == (2, 60, 6)
    assert record["source_path"] == str(raw(base)) and record["method"] == "local"
    for key in ("git_commit", "synced_at", "source_host"):
        assert key in record
    # The manifest sits beside the copy, so it never changes the copy's fingerprint.
    assert not (datasets / "pick_v1" / "pick_v1.manifest.json").exists()
    assert (base / "cache" / "manifests" / "real_pick_v1.json").is_file()


def test_rerun_says_already_synced(base, config_file, capsys, monkeypatch):
    write_dataset(raw(base))
    assert run_main(base, config_file, capsys=capsys)[0] == 0

    def no_transfer(*a, **k):
        raise AssertionError("transferred again")
    monkeypatch.setattr(LocalTransport, "transfer", no_transfer)
    rc, out, err = run_main(base, config_file, capsys=capsys)
    assert rc == 0 and "already synced" in err
    assert out[-2].endswith("already=true") and out[-1] == "STATUS done exit=0"


def test_different_remote_fingerprint_is_refused(base, config_file, tmp_path, capsys):
    root = write_dataset(raw(base))
    assert run_main(base, config_file, capsys=capsys)[0] == 0
    remote_copy = tmp_path / "gpu" / "datasets" / "pick_v1"
    before = dataset_manifest.manifest(str(remote_copy))
    (root / "meta" / "episodes.jsonl").write_text("changed\n", encoding="utf-8")
    rc, out, err = run_main(base, config_file, capsys=capsys)
    assert rc == 2 and out[-1] == "STATUS failed exit=2"
    assert "immutable" in err and "new name" in err
    assert dataset_manifest.manifest(str(remote_copy)) == before


def test_interrupted_transfer_resumes_from_incoming(base, remote, tmp_path, capsys):
    root = write_dataset(raw(base))
    transport: LocalTransport = remote.transport
    # Stop as soon as the first file has been copied.
    rc = None
    with pytest.raises(dataset_sync.TransferStopped):
        rc = sync("pick_v1", remote, base, stopped=lambda: len(transport.transferred) >= 1)
    assert rc is None
    incoming = tmp_path / "gpu" / "datasets" / INCOMING / "pick_v1"
    first = transport.transferred[:]
    assert len(first) == 1 and (incoming / first[0]).is_file()
    assert not (tmp_path / "gpu" / "datasets" / "pick_v1").exists()
    (incoming / "stale_from_earlier_attempt.bin").write_bytes(b"x")

    assert sync("pick_v1", remote, base) == 0
    assert "resuming from" in capsys.readouterr().err
    assert first[0] not in transport.transferred
    assert len(transport.transferred) == 5
    final = tmp_path / "gpu" / "datasets" / "pick_v1"
    assert dataset_manifest.manifest(str(final)) == dataset_manifest.manifest(str(root))


def test_verify_mismatch_fails_and_moves_nothing(base, config_file, tmp_path, capsys):
    write_dataset(raw(base))

    class Corrupting(LocalTransport):
        def transfer(self, local_root, dest, progress, log):
            method = super().transfer(local_root, dest, progress, log)
            (Path(dest) / "meta" / "info.json").write_text("{}", encoding="utf-8")
            return method

    remote = Remote(RemoteConfig.load(config_file), Corrupting())
    assert sync("pick_v1", remote, base) == 1
    err = capsys.readouterr().err
    assert "does not match" in err and "content differs: meta/info.json" in err
    datasets = tmp_path / "gpu" / "datasets"
    assert not (datasets / "pick_v1").exists() and not (datasets / "pick_v1.manifest.json").exists()
    assert (datasets / INCOMING / "pick_v1").is_dir()


def test_copy_without_manifest_is_recovered(base, config_file, tmp_path, capsys):
    write_dataset(raw(base))
    assert run_main(base, config_file, capsys=capsys)[0] == 0
    manifest = tmp_path / "gpu" / "datasets" / "pick_v1.manifest.json"
    manifest.unlink()  # as if the run died between the move and the manifest write
    rc, out, err = run_main(base, config_file, capsys=capsys)
    assert rc == 0 and "without a manifest" in err and manifest.is_file()


def test_not_sendable_and_not_configured_exit_2(base, tmp_path, capsys):
    write_dataset(raw(base), stats=NO_QUANTILES)
    rc, out, err = run_main(base, tmp_path / "remote.yaml.missing", capsys=capsys)
    assert rc == 2 and "remote.yaml.example" in err and out[-1] == "STATUS failed exit=2"


def test_stop_sets_status_stopped(base, config_file, capsys, monkeypatch):
    write_dataset(raw(base))

    def stop_mid_transfer(self, local_root, dest, progress, log):
        import signal
        signal.raise_signal(signal.SIGTERM)  # as lab's runner would
        progress(10)
        raise AssertionError("progress should have raised")
    monkeypatch.setattr(LocalTransport, "transfer", stop_mid_transfer)
    rc, out, err = run_main(base, config_file, capsys=capsys)
    assert rc == 143 and out[-1] == "STATUS stopped exit=143"
    assert "resumes from .incoming" in err


# --- names and paths ---

@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b", "a\\b", ".hidden", "a b", "x;rm -rf ~", "-rf"])
def test_bad_names_are_refused(name, remote, base):
    with pytest.raises(ValueError):
        validate_dataset_name(name)
    with pytest.raises(ValueError):
        remote.remote_dataset_root(name)
    with pytest.raises(SystemExit):
        dataset_sync.main([f"--name={name}", f"--data-base={base}"])


def test_transfer_only_replaces_inside_incoming(remote, base, tmp_path):
    root = write_dataset(raw(base))
    for dest in (remote.remote_dataset_root("pick_v1"), remote.remote_datasets_dir(),
                 f"{remote.remote_datasets_dir()}/{INCOMING}/../pick_v1"):
        with pytest.raises(ValueError):
            remote.transport.transfer(root, dest, lambda n: None, lambda s: None)


def test_dataset_root_on_the_host():
    config = RemoteConfig(host="h", user="u", repo_dir="/r", data_dir="/data/mtrea", python="/v/bin/python")
    r = Remote(config, LocalTransport())
    assert r.remote_datasets_dir() == "/data/mtrea/datasets"
    assert r.remote_dataset_root("pick_v1") == "/data/mtrea/datasets/pick_v1"
    assert r.remote_incoming_root("pick_v1") == "/data/mtrea/datasets/.incoming/pick_v1"
    assert r.remote_manifest_path("pick_v1") == "/data/mtrea/datasets/pick_v1.manifest.json"


# --- configuration and SSH command construction ---

def test_example_config_loads():
    config = RemoteConfig.load(REPO_ROOT / "robot" / "configs" / "remote.yaml.example")
    assert config.port == 22 and config.policy_port == 8000 and config.gpu_count == 1


def test_missing_config_is_not_configured(tmp_path):
    with pytest.raises(RemoteNotConfigured, match="remote.yaml.example"):
        RemoteConfig.load(tmp_path / "remote.yaml")


@pytest.mark.parametrize("text,match", [
    ("host: h\nuser: u\n", "missing"),
    ("host: h\nuser: u\nrepo_dir: /r\ndata_dir: /d\npython: /p\nhostname: x\n", "unknown fields"),
    ("host: h\nuser: u\nrepo_dir: /r\ndata_dir: /d\npython: /p\nport: '22'\n", "port"),
    ("- a list\n", "mapping"),
])
def test_bad_config_is_not_configured(tmp_path, text, match):
    path = tmp_path / "remote.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RemoteNotConfigured, match=match):
        RemoteConfig.load(path)


def ssh_config(**over) -> RemoteConfig:
    return RemoteConfig(**{"host": "gpu.tailnet.ts.net", "user": "lab", "repo_dir": "/home/lab/MT REA",
                           "data_dir": "/data/my data", "python": "/home/lab/.venv/bin/python",
                           "ssh_key": "~/.ssh/id_ed25519", "port": 2222, **over})


def test_ssh_argv_quotes_every_argument():
    argv = ssh_config().ssh_argv(["ls", "a b; rm -rf /", "$(id)"])
    assert argv[:9] == ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-p", "2222",
                        "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert argv[9:11] == ["lab@gpu.tailnet.ts.net", "--"]
    assert shlex.split(argv[11]) == ["ls", "a b; rm -rf /", "$(id)"]
    assert "-i" not in ssh_config(ssh_key=None).ssh_argv("true")
    assert ssh_config(port=22).label == "lab@gpu.tailnet.ts.net"


def test_remote_prefix_puts_the_venv_on_path():
    prefix = ssh_config().remote_prefix()
    assert prefix == "cd '/home/lab/MT REA' && export PATH=/home/lab/.venv/bin:$PATH &&"


def test_rsync_argv(tmp_path):
    argv = SshTransport(ssh_config()).rsync_argv(tmp_path / "pick_v1", "/data/my data/datasets/.incoming/pick_v1")
    assert argv[:6] == ["rsync", "-a", "--partial", "--delete", "-s", "--info=progress2"]
    assert argv[6] == "-e" and shlex.split(argv[7])[:5] == ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
                                                            "-p", "2222"]
    assert argv[8] == f"{(tmp_path / 'pick_v1').as_posix()}/"
    assert argv[9] == "lab@gpu.tailnet.ts.net:/data/my data/datasets/.incoming/pick_v1/"  # -s: no remote re-splitting


def test_ssh_commands_quote_remote_paths(monkeypatch):
    seen = []

    def fake_run(self, command, *, input=None, timeout=60):
        seen.append(command)
        return subprocess.CompletedProcess([], 0, b"absent\n", b"")
    monkeypatch.setattr(SshTransport, "run", fake_run)
    t = SshTransport(ssh_config())
    assert t.probe("/data/my data/datasets", "pick_v1") == ("absent", None)
    assert "'/data/my data/datasets/pick_v1.manifest.json'" in seen[-1]
    assert "'/data/my data/datasets/.incoming/pick_v1'" in seen[-1]
    with pytest.raises(ValueError):
        t.probe("/data/my data/datasets", "x'; rm -rf ~; '")


@pytest.mark.skipif(os.name == "nt" or shutil.which("tar") is None, reason="POSIX sh and tar")
def test_ssh_transport_shell_commands_run_on_a_posix_shell(base, tmp_path, monkeypatch):
    """SshTransport's real command strings, run by a local sh instead of over ssh (tar fallback)."""
    class Loopback(SshTransport):
        def _argv(self, command):
            return ["sh", "-c", command]
    write_dataset(raw(base))
    config = RemoteConfig(host="h", user="u", repo_dir=str(tmp_path), data_dir=str(tmp_path / "gpu"),
                          python=sys.executable)
    transport = Loopback(config)
    monkeypatch.setattr(transport, "remote_has", lambda program: False)
    remote = Remote(config, transport)
    assert sync("pick_v1", remote, base) == 0
    assert sync("pick_v1", remote, base) == 0  # already synced, read back over the same commands
    assert remote_state(remote, "real", "pick_v1", base)["state"] == "synced"
    assert transport.list_manifests(remote.remote_datasets_dir())["pick_v1"]["fingerprint"]


# --- runner ---

def test_runner_refuses_a_second_sync(tmp_path):
    paths.ensure_dirs(tmp_path)
    executor = LocalExecutor(repo_root=tmp_path)
    runner = Runner(JobStore(paths.jobs_ledger(tmp_path)), {Executor.LOCAL: executor}, tmp_path, tmp_path)
    sleep = subprocess.list2cmdline([sys.executable, "-c", "import time; time.sleep(30)"])
    runner.submit(JobSpec(kind=JobKind.SYNC, cmd=sleep, label="sync a"))
    try:
        with pytest.raises(JobConflict):
            runner.submit(JobSpec(kind=JobKind.SYNC, cmd=sleep, label="sync b"))
        runner.submit(JobSpec(kind=JobKind.CONVERT, cmd=sleep, label="other kinds are not blocked"))
    finally:
        for proc in executor._procs.values():
            proc.kill()
            proc.wait()


# --- API ---

def wait_done(client, job_id, timeout_s=60) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("sync job did not finish")


def test_api_not_configured(base, tmp_path):
    write_dataset(raw(base))
    with TestClient(create_app(base, remote_config=tmp_path / "absent.yaml", remote_transport="local")) as client:
        status = client.get("/api/remote/status")
        assert status.status_code == 200
        assert status.json()["configured"] is False and "remote.yaml" in status.json()["message"]
        state = client.get("/api/datasets/real/pick_v1/remote").json()
        assert state["state"] == "unknown" and state["sendable"] is True and state["remote_path"] is None
        response = client.post("/api/datasets/real/pick_v1/sync")
        assert response.status_code == 409 and "not configured" in response.json()["detail"]
        assert client.get("/api/remote/datasets").json() == {"real/pick_v1": "unknown"}


def test_api_sync_then_synced_then_different(base, config_file, tmp_path):
    root = write_dataset(raw(base))
    write_dataset(base / "fake" / "collection" / "raw" / "demo")
    with TestClient(create_app(base, remote_config=config_file, remote_transport="local")) as client:
        status = client.get("/api/remote/status").json()
        assert status["configured"] is True and status["reachable"] is True
        assert status["host"] == "lab@gpu.tailnet.ts.net"
        assert client.get("/api/datasets/real/pick_v1/remote").json()["state"] == "absent"

        job = client.post("/api/datasets/real/pick_v1/sync").json()
        assert job["spec"]["kind"] == "sync" and job["spec"]["label"] == "sync pick_v1"
        finished = wait_done(client, job["id"])
        log = client.get(f"/api/jobs/{job['id']}/logs", params={"tail_lines": 1000}).json()["content"]
        assert finished["status"] == "done", log
        assert "PROGRESS phase=transfer percent=100" in log

        state = client.get("/api/datasets/real/pick_v1/remote").json()
        assert state["state"] == "synced"
        assert state["remote_path"] == f"{(tmp_path / 'gpu').as_posix()}/datasets/pick_v1"
        assert state["local_fingerprint"] == state["remote_fingerprint"] and state["synced_at"]
        assert client.get("/api/remote/datasets").json() == {"real/pick_v1": "synced",
                                                             "fake/demo": "not_sent_fake"}

        (root / "meta" / "episodes.jsonl").write_text("changed\n", encoding="utf-8")
        # Not hashed by a status request: the changed dataset is unknown until a sync looks at it.
        assert client.get("/api/datasets/real/pick_v1/remote").json()["state"] == "unknown"
        refused = wait_done(client, client.post("/api/datasets/real/pick_v1/sync").json()["id"])
        assert refused["status"] == "failed" and refused["exit_code"] == 2
        state = client.get("/api/datasets/real/pick_v1/remote").json()
        assert state["state"] == "different" and state["local_fingerprint"] != state["remote_fingerprint"]


def test_api_refuses_fake_unsendable_and_concurrent(base, config_file):
    write_dataset(raw(base))
    write_dataset(raw(base, "no_stats"), stats=NO_QUANTILES)
    write_dataset(base / "fake" / "collection" / "raw" / "demo")
    with TestClient(create_app(base, remote_config=config_file, remote_transport="local")) as client:
        fake = client.post("/api/datasets/fake/demo/sync")
        assert fake.status_code == 409 and "not sent: fake" in fake.json()["detail"]
        fake_state = client.get("/api/datasets/fake/demo/remote").json()
        assert fake_state["reason"] == "not sent: fake"
        assert fake_state["remote_path"] is None and fake_state["state"] == "unknown"
        unsendable = client.post("/api/datasets/real/no_stats/sync")
        assert unsendable.status_code == 409 and "prepare_dataset" in unsendable.json()["detail"]
        assert client.post("/api/datasets/real/missing/sync").status_code == 404
        assert client.get("/api/datasets/real/..%2Fetc/remote").status_code == 404

        runner = client.app.state.runner
        sleep = subprocess.list2cmdline([sys.executable, "-c", "import time; time.sleep(30)"])
        runner.submit(JobSpec(kind=JobKind.SYNC, cmd=sleep, label="sync other"))
        try:
            busy = client.post("/api/datasets/real/pick_v1/sync")
            assert busy.status_code == 409 and "sync" in busy.json()["detail"]
        finally:
            for proc in runner.executors[Executor.LOCAL]._procs.values():
                proc.kill()
                proc.wait()
