"""Archive datasets and their collection records without erasing them."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from lab.core import paths
from lab.core.datasets import Backend, dataset_root
from lab.core.hardware.lock import HardwareLock


def archive_dataset(
    backend: Backend,
    name: str,
    data_base: Path | None = None,
) -> dict:
    base = paths.root(data_base).resolve()
    lock = HardwareLock(paths.hardware_lock(base))
    lock.acquire("dataset-delete")

    try:
        root = dataset_root(backend, name, base)

        candidates = [
            root,
            root.parent / f"{name}.collection.jsonl",
            root.parent / f"{name}.collection.input.json",
        ]

        sources = []
        for source in candidates:
            if source.is_symlink():
                raise ValueError("Cannot archive a symbolic link")

            if not source.resolve().is_relative_to(base):
                raise ValueError("Dataset path is outside data storage")

            if source.exists():
                sources.append(source)

        archive_id = uuid4().hex
        destination = paths.dataset_trash_dir(base) / archive_id

        if not destination.resolve().is_relative_to(base):
            raise ValueError("Archive path is outside data storage")

        destination.mkdir(parents=True, exist_ok=False)

        manifest = {
            "archive_id": archive_id,
            "backend": backend,
            "name": name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "original_paths": [
                str(source.relative_to(base))
                for source in sources
            ],
        }

        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        moved = []
        try:
            for source in sources:
                target = destination / source.name
                source.rename(target)
                moved.append((source, target))

        except OSError:
            for source, target in reversed(moved):
                target.rename(source)
            raise

        return {
            "id": f"{backend}/{name}",
            "archive_id": archive_id,
            "archive_location": str(destination.relative_to(base)),
        }

    finally:
        lock.release()
