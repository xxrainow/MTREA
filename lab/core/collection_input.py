import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
import subprocess
from importlib import metadata

from lab.core.hardware import HardwareError
from lab.core import paths
from lab.core.fake_recorder import fake_collection_enabled


def save_collection_input(
    raw: dict,
    *,
    data_base: Path | None = None,
) -> dict:
    config = deepcopy(raw)
    session_index = config.pop("session_index")
    dataset = config["dataset"]

    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]*",
        dataset,
    ):
        raise ValueError("Invalid dataset name")

    if not 1 <= session_index <= len(config["sessions"]):
        raise ValueError("Selected session does not exist")

    # This snapshot records the input, not research-protocol approval.
    config["schema_version"] = 1
    config["purpose"] = "pipeline_check"

    content = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"

    raw_dir = paths.raw_dir(data_base)
    dataset_root = raw_dir / dataset
    ledger_path = raw_dir / f"{dataset}.collection.jsonl"
    config_path = raw_dir / f"{dataset}.collection.input.json"

    if not config_path.exists() and (
        dataset_root.exists() or ledger_path.exists()
    ):
        raise ValueError(
            "Existing data has no matching web collection config; "
            "use a new dataset name"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        with config_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(content)
    except FileExistsError:
        if config_path.read_bytes() != content.encode("utf-8"):
            raise ValueError(
                "Collection settings changed; use a new dataset name"
            ) from None

    return {
        "config_path": config_path,
        "plan_sha": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
        "dataset_root": dataset_root,
        "ledger_path": ledger_path,
        "session_index": session_index,
        "session": deepcopy(
            config["sessions"][session_index - 1]
        ),
        "seed": config["seed"],
    }

def prepare_start_arguments(
    raw: dict,
    *,
    data_base: Path | None = None,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    fake_mode = fake_collection_enabled()

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

        lerobot_version = (
            "fake" if fake_mode else metadata.version("lerobot")
        )
    except (
        OSError,
        subprocess.SubprocessError,
        metadata.PackageNotFoundError,
    ) as exc:
        raise HardwareError(
            f"Cannot prepare recording environment: {exc}"
        ) from None

    collection_base = paths.root(data_base)
    if fake_mode:
        collection_base = collection_base / "fake" / "collection"

    saved = save_collection_input(
        raw,
        data_base=collection_base,
    )
    session = saved["session"]

    return {
        "dataset_root": saved["dataset_root"],
        "ledger_path": saved["ledger_path"],
        "plan_sha": saved["plan_sha"],
        "session_index": saved["session_index"],
        "planned_episodes": session["num_episodes"],
        "repo_id": f"local/{raw['dataset']}",
        "instruction": session["instruction"],
        "episode_time_s": session["episode_time_s"],
        "reset_time_s": session["reset_time_s"],
        "record_context": {
            "object": None,
            "plan": str(saved["config_path"]),
            "git_commit": commit,
            "git_dirty": bool(dirty),
            "lerobot_version": lerobot_version,
            "seed": saved["seed"],
            "purpose": "pipeline_check",
            "layouts": [
                {
                    "target_position": None,
                    "distractor": None,
                    "distractor_position": None,
                    "distractor_kind": None,
                }
                for _ in range(session["num_episodes"])
            ],
            "backend": "fake" if fake_mode else "real",
        },
    }
