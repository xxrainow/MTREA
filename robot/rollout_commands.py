"""Per-trial command files; submission is distinct from execution acknowledgement."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import psutil

from lab.core.recording_telemetry import atomic_write


def submit_command(trial_dir: Path, action: str) -> str:
    if action not in {"stop", "release"}:
        raise ValueError("Expected stop or release")
    status = json.loads((trial_dir / "status.json").read_text())
    if not status.get("process_alive"):
        raise RuntimeError("Trial process is not accepting commands")
    try:
        if abs(psutil.Process(status["pid"]).create_time() - status["process_started"]) > 0.01:
            raise RuntimeError("Trial process identity has changed")
    except psutil.Error as exc:
        raise RuntimeError("Trial process is no longer available") from exc
    command_id = uuid4().hex
    directory = trial_dir / "commands"
    directory.mkdir(exist_ok=True)
    atomic_write(directory / f"{command_id}.json", json.dumps({
        "id": command_id, "trial_token": status["trial_token"], "action": action,
    }).encode())
    return command_id


class CommandInbox:
    def __init__(self, root: Path, token: str):
        self.root = root
        self.token = token
        self.commands = root / "commands"
        self.results = root / "command_results"
        self.commands.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)

    def poll(self):
        for path in sorted(self.commands.glob("*.json")):
            result = self.results / path.name
            if result.exists():
                continue
            try:
                data = json.loads(path.read_text())
                if (not isinstance(data, dict) or data.get("id") != path.stem
                        or data.get("trial_token") != self.token
                        or not isinstance(data.get("action"), str)
                        or data.get("action") not in {"stop", "release"}):
                    raise ValueError("Invalid command or stale trial token")
            except (OSError, ValueError) as exc:
                self.ack(path.stem, "rejected", str(exc))
                continue
            yield data

    def ack(self, command_id: str, state: str, error=None):
        atomic_write(self.results / f"{command_id}.json", json.dumps({
            "id": command_id, "state": state, "error": error,
        }).encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Submit a rollout Stop or Release command.")
    parser.add_argument("action", choices=["stop", "release"])
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"command_id": submit_command(args.trial_dir, args.action),
                      "state": "queued"}), flush=True)


if __name__ == "__main__":
    main()
