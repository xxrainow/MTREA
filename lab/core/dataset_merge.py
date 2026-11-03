"""Create a new local dataset, preserving every source and its provenance."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from lab.core import dataset_manifest, paths
from lab.core.dataset_metadata import read_episode_tasks, read_tasks
from lab.core.datasets import Backend, dataset_root, get_dataset, validate_dataset_name
from lab.core.hardware.lock import HardwareLock


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def _storage(backend: Backend, base: Path) -> Path:
    if backend not in ("real", "fake"):
        raise ValueError("Invalid backend")
    storage = base / ("raw" if backend == "real" else "fake/collection/raw")
    if not storage.resolve().is_relative_to(base.resolve()):
        raise ValueError("Storage points outside data directory")
    return storage


def new_dataset_path(backend: Backend, name: str, base: Path) -> Path:
    validate_dataset_name(name)
    storage = _storage(backend, base)
    target = storage / name
    # Prepared copies and collection ledgers also reserve a name.
    reserved = [target, base / "datasets" / name,
                storage / f"{name}.collection.jsonl", storage / f"{name}.collection.input.json"]
    if any(path.exists() or path.is_symlink() for path in reserved):
        raise ValueError("Dataset name already exists; choose a new name")
    return target


def _require_lerobot() -> None:
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        raise ValueError("Real merge requires LeRobot 0.6.1 in the server's Python environment") from None
    if installed != "0.6.1":
        raise ValueError(f"Real merge requires LeRobot 0.6.1; installed: {installed}")


def check_merge(backend: Backend, names: list[str], name: str, data_base: Path | None = None) -> dict:
    base = paths.root(data_base).resolve()
    target = new_dataset_path(backend, name, base)
    if not 2 <= len(names) <= 50 or len(set(names)) != len(names):
        raise ValueError("Select 2–50 distinct source datasets")
    if backend == "real":
        _require_lerobot()
    infos, details, listings = [], [], []
    for source in names:
        validate_dataset_name(source)
        root = dataset_root(backend, source, base)
        listings.append(dataset_manifest.listing(root))
        detail = get_dataset(backend, source, base)
        if detail.dataset.issues:
            raise ValueError(f"{source}: {'; '.join(detail.dataset.issues)}")
        if not detail.episodes or [ep.index for ep in detail.episodes] != list(range(len(detail.episodes))):
            raise ValueError(f"{source}: expected non-empty, contiguous episode indices")
        info = _json(root / "meta/info.json")
        if backend == "fake":
            if info.get("fake") is not True:
                raise ValueError(f"{source}: not a fake dataset")
        else:
            if info.get("fake") or info.get("codebase_version") != "v3.0":
                raise ValueError(f"{source}: real merge supports LeRobot v3.0 datasets only; convert older copies first")
            if not detail.dataset.fps or not info.get("features") or not detail.dataset.robot_type:
                raise ValueError(f"{source}: missing recording settings")
            if any("/" in key or "\\" in key or key in (".", "..") for key in info["features"]):
                raise ValueError(f"{source}: invalid feature name")
            if not (root / "meta/tasks.parquet").is_file() or not (root / "meta/stats.json").is_file():
                raise ValueError(f"{source}: missing v3 task metadata or statistics")
            if not any((root / "data").rglob("*.parquet")):
                raise ValueError(f"{source}: no recorded data")
            if type(info.get("total_frames")) is not int or info["total_frames"] <= 0:
                raise ValueError(f"{source}: invalid frame count")
        if infos:
            for key in ("fps", "robot_type", "features"):
                if info.get(key) != infos[0].get(key):
                    raise ValueError(f"{source}: {key} differs from the first source")
        infos.append(info)
        details.append(detail)
    return {
        "backend": backend, "name": name, "sources": names,
        "episodes": sum(len(d.episodes) for d in details),
        "total_frames": (sum(i["total_frames"] for i in infos)
                         if all(type(i.get("total_frames")) is int for i in infos) else None),
        "location": str(target.relative_to(base)),
        # Captures the checked source versions for the worker's second preflight.
        "source_versions": [dataset_manifest.fingerprint(rows) for rows in listings],
    }


def _merge_real(roots: list[Path], output: Path, name: str) -> None:
    # Sources are local. Missing metadata must fail instead of fetching Hub data.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from lerobot.datasets.aggregate import aggregate_datasets
    aggregate_datasets(
        repo_ids=[f"local/{root.name}" for root in roots], roots=roots,
        aggr_repo_id=f"local/{name}", aggr_root=output,
        concatenate_videos=False, concatenate_data=False,
    )


def _merge_fake(roots: list[Path], output: Path) -> None:
    meta = output / "meta"
    meta.mkdir(parents=True)
    episodes, tasks = [], set()
    for root in roots:
        tasks.update(read_tasks(root))
        for _, instructions in sorted(read_episode_tasks(root).items()):
            episodes.append({"episode_index": len(episodes), "tasks": instructions})
    (meta / "info.json").write_text(json.dumps({"fake": True, "total_episodes": len(episodes)}))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task": task}) + "\n" for task in sorted(tasks)))


def merge_datasets(backend: Backend, names: list[str], name: str, data_base: Path | None = None,
                   expected_versions: list[str] | None = None) -> dict:
    base = paths.root(data_base).resolve()
    lock = HardwareLock(paths.hardware_lock(base))
    lock.acquire("dataset-merge")
    stage = None
    try:
        checked = check_merge(backend, names, name, base)
        if expected_versions is not None and checked["source_versions"] != expected_versions:
            raise ValueError("Source datasets changed after review; check the selection again")
        roots = [dataset_root(backend, source, base) for source in names]
        print("PROGRESS phase=fingerprint", flush=True)
        fingerprints = [dataset_manifest.fingerprint(dataset_manifest.manifest(root)) for root in roots]
        target = new_dataset_path(backend, name, base)
        staging = target.parent / ".merging"
        if staging.is_symlink():
            raise ValueError("Merge staging directory cannot be a symbolic link")
        stage = staging / uuid4().hex
        stage.mkdir(parents=True)
        output = stage / "output"
        print("PROGRESS phase=merge", flush=True)
        if backend == "real":
            _merge_real(roots, output, name)
        else:
            _merge_fake(roots, output)
        print("PROGRESS phase=verify", flush=True)
        info = _json(output / "meta/info.json")
        episodes = read_episode_tasks(output)
        if info.get("total_episodes") != checked["episodes"] or sorted(episodes) != list(range(checked["episodes"])):
            raise ValueError("Merged episode count or indices are invalid")
        expected_tasks = [tasks for root in roots for _, tasks in sorted(read_episode_tasks(root).items())]
        if [episodes[index] for index in sorted(episodes)] != expected_tasks:
            raise ValueError("Merged episode instructions differ from the sources")
        if backend == "real":
            import pyarrow.parquet as pq
            rows = sum(pq.read_metadata(path).num_rows for path in (output / "data").rglob("*.parquet"))
            if info.get("total_frames") != checked["total_frames"] or rows != checked["total_frames"]:
                raise ValueError("Merged frame count differs from sources")
        dataset_manifest.listing(output)
        after = [dataset_manifest.fingerprint(dataset_manifest.manifest(root)) for root in roots]
        if after != fingerprints:
            raise ValueError("Source data changed while merging; output was not published")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=True).stdout.strip()
        provenance = {"schema_version": 1, "git_commit": commit, "backend": backend,
                      "created_at": datetime.now(timezone.utc).isoformat(), "name": name, "sources": []}
        offset = 0
        for root, fingerprint in zip(roots, fingerprints):
            count = len(read_episode_tasks(root))
            provenance["sources"].append({"name": root.name, "location": str(root.relative_to(base)),
                                          "fingerprint": fingerprint, "episodes": count,
                                          "output_episode_start": offset})
            offset += count
        (output / "meta/merge.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # The hardware lock serializes collection, deletion and other merge workers.
        new_dataset_path(backend, name, base)
        output.rename(target)
        print(f"MERGED backend={backend} name={name} episodes={checked['episodes']}", flush=True)
        return checked
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        lock.release()


def merge_command(checked: dict, data_base: Path | None = None) -> str:
    args = [sys.executable, "-m", "lab.core.dataset_merge", f"--backend={checked['backend']}",
            f"--name={checked['name']}", f"--data-base={paths.root(data_base).resolve()}",
            f"--expected-versions={json.dumps(checked['source_versions'])}"]
    args.extend(f"--source={name}" for name in checked["sources"])
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["real", "fake"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--expected-versions", required=True)
    args = parser.parse_args()
    def stopped(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stopped)
    try:
        merge_datasets(args.backend, args.source, args.name, args.data_base, json.loads(args.expected_versions))
    except Exception as exc:
        print(f"ERROR {exc}", flush=True)
        return 1
    print("STATUS done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
