"""Drive a policy server without a robot, from dataset frames.

    python -m research.serve_client_check --server=127.0.0.1:<port> --checkpoint=<same string the server loaded> \\
        --dataset-root=<dir> --episodes=2 [--task="<instruction>"] [--fps=30] [--actions-per-chunk=50] \\
        [--threshold=0.5] [--out=<report.json>]

This runs lerobot's own RobotClient: its handshake, control loop, queue and
aggregation, chunk threshold and channel options, so the timing is the client's
and not an imitation of it. Only the robot is replaced. DatasetRobot reports the
frame of the action last executed and discards commands. The episode's own
instruction is used unless --task is given, and each episode is a fresh client
(Ready, instructions), as a new robot_client process would be.

Two lerobot behaviours shape the numbers:
- Below the threshold, the control loop sends an observation on every tick, not
  once per chunk, until a chunk arrives. Sending blocks the control loop, so a
  link slower than the peak rate lowers the tick rate during that window.
- Below the threshold the server drops observations whose joints moved less than
  1 unit in norm. An arm at rest therefore runs its queue dry before the
  must_go observation gets through, which shows up here as a short stall.

Exit codes: 0 done, 1 refused by the server or failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec: the protocol itself
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from research.serve_common import (UNITS_NOTE, ChunkErrors, DatasetError, DemoDataset, latency_summary,
                                   lerobot_log_dir, require_cameras, run_provenance)
from research.train import _write_json


class DatasetRobot:
    """Stands in for an so101_follower: the same observation keys and shapes, no hardware."""

    def __init__(self, dataset: DemoDataset, episode: int, images: dict):
        self.dataset = dataset
        self.episode = episode
        self.images = images
        self.length = dataset.episode_length(episode)
        self.client = None  # the RobotClient, set once it exists: it needs this robot to start
        features = {name: float for name in dataset.state_names}
        for camera in dataset.cameras:
            features[camera] = images[camera][0].shape
        # What RobotClient reads to build lerobot_features and to name actions.
        self.observation_features = features
        self.action_features = {name: float for name in dataset.action_names}

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def current_frame(self) -> int:
        return min(max(self.client.latest_action, 0), self.length - 1)

    def get_observation(self) -> dict:
        frame = self.current_frame()
        images = {camera: self.images[camera][frame] for camera in self.dataset.cameras}
        raw = self.dataset.raw_observation(self.episode, frame, images, task="")
        del raw["task"]  # robot_client sets it
        return raw

    def send_action(self, action: dict) -> dict:
        # The control loop records latest_action right after this returns.
        if self.client.latest_action + 1 >= self.length - 1:
            self.client.shutdown_event.set()
        return action


def make_check_client(robot_client_module):
    class CheckClient(robot_client_module.RobotClient):
        """RobotClient plus measurement. Every override delegates to the original."""

        def __init__(self, config, robot: DatasetRobot):
            self.round_trips_ms: list[float] = []
            self.send_ms: list[float] = []
            self.observation_bytes: Optional[int] = None
            self.observations_sent = 0
            self.chunks: list[list] = []
            self.chunk_bytes: Optional[int] = None
            self.queue_empty_ticks = 0
            self.stall_ms: list[float] = []
            self._stall_start: Optional[float] = None
            self._first_chunk = threading.Event()
            self.refusal: Optional[str] = None
            super().__init__(config)
            robot.client = self

        def send_observation(self, obs) -> bool:
            if self.observation_bytes is None:
                self.observation_bytes = len(pickle.dumps(obs))
            start = time.perf_counter()
            sent = super().send_observation(obs)
            self.send_ms.append((time.perf_counter() - start) * 1000)
            self.observations_sent += 1
            return sent

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            if incoming_actions:
                # The first action's timestamp is the observation's, set by this
                # machine's clock, so this holds across hosts.
                self.round_trips_ms.append((time.time() - incoming_actions[0].get_timestamp()) * 1000)
                self.chunks.append([incoming_actions[0].get_timestep(),
                                    [a.get_action().tolist() for a in incoming_actions]])
                if self.chunk_bytes is None:
                    self.chunk_bytes = len(pickle.dumps(incoming_actions))
            super()._aggregate_action_queues(incoming_actions, aggregate_fn)
            self._first_chunk.set()

        def actions_available(self) -> bool:
            available = super().actions_available()
            if self._first_chunk.is_set():
                now = time.perf_counter()
                if not available:
                    self.queue_empty_ticks += 1
                    if self._stall_start is None:
                        self._stall_start = now
                elif self._stall_start is not None:
                    self.stall_ms.append((now - self._stall_start) * 1000)
                    self._stall_start = None
            return available

    return CheckClient


class _RefusalCapture(logging.Handler):
    """RobotClient.start() logs the server's refusal and returns False, dropping the message."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def run_episode(modules, args, dataset: DemoDataset, episode: int) -> dict:
    robot_client, configs = modules
    start, _ = dataset.episode_range(episode)
    task = args.task if args.task is not None else dataset.task_of(start)
    print(f"EPISODE {episode} decoding {dataset.episode_length(episode)} frames", flush=True)
    images = dataset.episode_images(episode)
    robot = DatasetRobot(dataset, episode, images)
    config = configs.RobotClientConfig(
        policy_type=args.policy_type, pretrained_name_or_path=args.checkpoint, robot=robot,
        actions_per_chunk=args.actions_per_chunk, task=task, server_address=args.server,
        chunk_size_threshold=args.threshold, fps=args.fps)

    client = make_check_client(robot_client)(config, robot)
    capture = _RefusalCapture()
    client.logger.addHandler(capture)
    try:
        if not client.start():
            refusal = next((m for m in capture.messages), "the server refused or could not be reached")
            client.stop()
            return {"episode": episode, "refused": refusal}
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        started = time.perf_counter()
        client.control_loop(task=task)
        wall = time.perf_counter() - started
    finally:
        client.logger.removeHandler(capture)
    client.stop()
    receiver.join(timeout=5)
    return {"episode": episode, "task": task, "frames": robot.length, "wall_s": wall, "client": client}


def compare_chunks(dataset: DemoDataset, episode: int, chunks: list[list], errors: ChunkErrors) -> None:
    for timestep, actions in chunks:
        frame = min(timestep, dataset.episode_length(episode) - 1)
        errors.add(actions, dataset.actions_from(episode, frame, len(actions)).tolist())


def build_report(args, dataset: DemoDataset, results: list[dict], guard_ms: int) -> dict:
    round_trips, sends, stalls = [], [], []
    errors = ChunkErrors(dataset.joint_names)
    sent = empty_ticks = chunk_count = 0
    wall = 0.0
    obs_bytes = chunk_bytes = None
    episodes = []
    for result in results:
        client = result["client"]
        round_trips += client.round_trips_ms
        sends += client.send_ms
        stalls += client.stall_ms
        sent += client.observations_sent
        empty_ticks += client.queue_empty_ticks
        chunk_count += len(client.chunks)
        wall += result["wall_s"]
        obs_bytes = client.observation_bytes or obs_bytes
        chunk_bytes = client.chunk_bytes or chunk_bytes
        compare_chunks(dataset, result["episode"], client.chunks, errors)
        episodes.append({"episode": result["episode"], "task": result["task"], "frames": result["frames"],
                         "wall_s": round(result["wall_s"], 2), "chunks": len(client.chunks),
                         "observations_sent": client.observations_sent,
                         "queue_empty_ticks": client.queue_empty_ticks})
    mbit = (lambda per_s: round(obs_bytes * 8 * per_s / 1e6, 1)) if obs_bytes else (lambda per_s: None)
    return {
        "server": args.server, "checkpoint": args.checkpoint, "fps": args.fps, "threshold": args.threshold,
        "actions_per_chunk": args.actions_per_chunk, "dataset_root": args.dataset_root, "units": UNITS_NOTE,
        "run": run_provenance(), "command": sys.argv,
        "episodes": episodes, "chunks": chunk_count,
        "round_trip_ms": latency_summary(round_trips),
        "observation_bytes": obs_bytes, "chunk_bytes": chunk_bytes,
        "observations_sent": sent, "observations_per_s": round(sent / wall, 2) if wall else None,
        "observation_send_ms": latency_summary(sends),
        "mbit_per_s": {
            "average": mbit(sent / wall) if wall else None,
            "while_sending_every_tick": mbit(args.fps),
            "note": "average over the run; while_sending is the rate needed in the below-threshold window, "
                    "during which each send blocks the control loop",
        },
        "stall": {"ran_empty": empty_ticks > 0, "queue_empty_ticks": empty_ticks, "stalls": len(stalls),
                  "duration_ms": latency_summary(stalls),
                  "over_guardrail": sum(1 for s in stalls if s > guard_ms), "guardrail_ms": guard_ms},
        "vs_dataset": errors.summary(),
    }


def print_summary(report: dict) -> None:
    print(f"chunks={report['chunks']} round_trip_ms={report['round_trip_ms']}")
    print(f"observation_bytes={report['observation_bytes']} observations_per_s={report['observations_per_s']} "
          f"send_ms={report['observation_send_ms']}")
    print(f"mbit_per_s average={report['mbit_per_s']['average']} "
          f"while_sending_every_tick={report['mbit_per_s']['while_sending_every_tick']}")
    print(f"stall={report['stall']}")
    errors = report["vs_dataset"]
    print(f"vs_dataset mean MAE first_1={errors['first_1']['mean_mae']} first_10={errors['first_10']['mean_mae']} "
          f"first_50={errors['first_50']['mean_mae']} ({UNITS_NOTE})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.serve_client_check", description=__doc__.split("\n")[0])
    p.add_argument("--server", required=True, help="host:port; through the tunnel that is 127.0.0.1:8765")
    p.add_argument("--checkpoint", required=True, help="the same string the server loaded")
    p.add_argument("--dataset-root", required=True, help="the dataset whose frames stand in for the robot")
    p.add_argument("--episodes", type=int, default=2, help="the first N episodes")
    p.add_argument("--task", default=None, help="instruction; default: each episode's own")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--actions-per-chunk", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.5, help="robot_client's chunk_size_threshold")
    p.add_argument("--policy-type", default="pi05")
    p.add_argument("--out", default=None, help="also write the report JSON here, atomically")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from robot.config import CAMERAS, RolloutGuardrails

        dataset = DemoDataset(Path(args.dataset_root))
        require_cameras(dataset, [c.name for c in CAMERAS])
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with lerobot_log_dir():
        from lerobot.async_inference import configs, robot_client

    # RobotClient connects to whatever this returns; the "config" here is the robot itself.
    robot_client.make_robot_from_config = lambda robot: robot

    results = []
    for episode in dataset.episodes[:args.episodes]:
        result = run_episode((robot_client, configs), args, dataset, episode)
        if "refused" in result:
            print(f"REFUSED {result['refused']}", flush=True)
            if args.out:
                _write_json(Path(args.out), {"server": args.server, "refused": result["refused"]})
            return 1
        results.append(result)
    report = build_report(args, dataset, results, RolloutGuardrails().queue_starved_ms)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, report)
    print_summary(report)
    print("REPORT " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
