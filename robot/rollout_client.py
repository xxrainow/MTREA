"""Rollout helpers that retain torque and device ownership after Stop.

Release arm is explicit so stopping a trial does not drop an unsupported arm.
"""

import argparse
import logging
import math
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread

from lab.core.hardware.cameras import load_camera_configs
from lab.core.hardware.devices import load_arm_config, safe_disconnect
from lab.core.hardware.lock import HardwareLock
from robot.config import (
    FPS,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
    RolloutGuardrails,
)
from robot.rollout_records import TrialRecorder
from robot.rollout_stalls import QueueStallTracker


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        ) from None

    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        )

    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one policy trial on the SO-101 follower."
    )
    parser.add_argument(
        "--server",
        default=f"{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument(
        "--max-relative-target",
        type=positive_float,
        required=True,
    )
    parser.add_argument(
        "--time-limit-s",
        type=positive_float,
        default=RolloutGuardrails().max_episode_seconds,
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def build_client_config(args: argparse.Namespace):
    if not args.checkpoint.strip():
        raise ValueError("checkpoint must not be empty")
    if not args.task.strip():
        raise ValueError("task must not be empty")
    if not math.isfinite(args.threshold):
        raise ValueError("threshold must be finite")

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    arm = load_arm_config("follower")
    if not arm.use_degrees:
        raise ValueError("Rollouts require use_degrees=true")

    cameras = {
        camera.name: OpenCVCameraConfig(
            index_or_path=camera.index_or_path,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
        )
        for camera in load_camera_configs()
    }

    robot = SO101FollowerConfig(
        port=arm.port,
        id=arm.id,
        use_degrees=arm.use_degrees,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=False,
        cameras=cameras,
    )

    return RobotClientConfig(
        policy_type="pi05",
        pretrained_name_or_path=args.checkpoint,
        robot=robot,
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        server_address=args.server,
        fps=args.fps,
        chunk_size_threshold=args.threshold,
        client_device="cpu",
    )


def create_locked_client(
    config,
    *,
    lock: HardwareLock,
    client_type,
):
    class ManagedClient(client_type):
        def __init__(self, config):
            self._io_lock = Lock()
            self._stop_requested = Event()
            self._stop_finished = Event()
            self._holding = False
            self._released = False
            self._stats_lock = Lock()
            self._chunks_received = 0
            self._actions_executed = 0
            self._last_latency_ms = None
            self._max_latency_ms = None
            self._stalls = QueueStallTracker(RolloutGuardrails().queue_starved_ms)

            try:
                super().__init__(config)
            except BaseException:
                channel = getattr(self, "channel", None)
                if channel is not None:
                    with suppress(Exception):
                        channel.close()

                safe_disconnect(
                    getattr(self, "robot", None),
                    "rollout client initialization",
                )
                raise

        @property
        def running(self):
            return (
                not self._stop_requested.is_set()
                and super().running
            )

        def start(self):
            if self._stop_requested.is_set():
                raise RuntimeError("Stopped clients cannot be restarted")
            accepted = super().start()
            if accepted and not self._stop_requested.is_set():
                self._stalls.start()
            return accepted

        def control_loop_action(self, verbose=False):
            with self._io_lock:
                if self._stop_requested.is_set():
                    return {}

                result = super().control_loop_action(verbose)

                with self._stats_lock:
                    self._actions_executed += 1

                self._stalls.action_executed()
                return result

        def _aggregate_action_queues(
            self, incoming_actions, aggregate_fn=None
        ):
            latency_ms = None
            if incoming_actions:
                latency_ms = (
                    time.time() - incoming_actions[0].get_timestamp()
                ) * 1000

            result = super()._aggregate_action_queues(
                incoming_actions, aggregate_fn
            )

            if incoming_actions:
                with self._stats_lock:
                    self._chunks_received += 1

                    if (
                        latency_ms is not None
                        and math.isfinite(latency_ms)
                        and latency_ms >= 0
                    ):
                        self._last_latency_ms = latency_ms
                        self._max_latency_ms = (
                            latency_ms
                            if self._max_latency_ms is None
                            else max(self._max_latency_ms, latency_ms)
                        )

            return result

        def metrics_snapshot(self) -> dict:
            with self.action_queue_lock:
                queue_size = self.action_queue.qsize()

            with self._stats_lock:
                snapshot = {
                    "chunks_received": self._chunks_received,
                    "actions_executed": self._actions_executed,
                    "queue_size": queue_size,
                    "last_latency_ms": self._last_latency_ms,
                    "max_latency_ms": self._max_latency_ms,
                }
            snapshot.update(self._stalls.snapshot())
            return snapshot

        def stall_events(self) -> list[dict]:
            return self._stalls.poll()

        def control_loop_observation(self, task, verbose=False):
            with self._io_lock:
                if self._stop_requested.is_set():
                    return None
                return super().control_loop_observation(task, verbose)

        def stop(self):
            self._stop_requested.set()
            self.shutdown_event.set()
            self._stalls.finish()

            try:
                try:
                    self.channel.close()
                finally:
                    with self._io_lock:
                        if not self._holding and not self._released:
                            positions = self.robot.bus.sync_read(
                                "Present_Position",
                                num_retry=self.robot.config.num_read_retries,
                            )
                            self.robot.bus.sync_write(
                                "Goal_Position",
                                positions,
                            )
                            self._holding = True
            finally:
                self._stop_finished.set()

        def release_arm(self):
            if not self._stop_finished.is_set():
                raise RuntimeError(
                    "Stop the client before releasing the arm"
                )

            with self._io_lock:
                if any(
                    thread.is_alive()
                    for thread in getattr(self, "_workers", ())
                ):
                    raise RuntimeError(
                        "Wait for policy workers to stop before releasing the arm"
                    )

                if self._released:
                    return

                bus = self.robot.bus
                if bus.is_connected:
                    bus.disable_torque(num_retry=5)
                    bus.disconnect(disable_torque=False)

                for camera in self.robot.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()

                self._released = True
                lock.release()

    lock.acquire("rollout")

    try:
        return ManagedClient(config)
    except BaseException:
        lock.release()
        raise


class PolicyConnectionError(RuntimeError):
    pass


class ClientErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def connect_policy(client) -> None:
    capture = ClientErrorCapture()
    client.logger.addHandler(capture)

    try:
        if not client.start():
            message = "\n".join(capture.messages)
            raise PolicyConnectionError(
                message or "The policy server refused or could not be reached"
            )
    finally:
        client.logger.removeHandler(capture)


def run_policy(
    client,
    *,
    task: str,
    time_limit_s: float,
    stop_requested: Event | None = None,
    on_state: Callable[[str], None] | None = None,
    on_progress: Callable[[], None] | None = None,
    on_stall: Callable[[dict], None] | None = None,
) -> str:
    if stop_requested is None:
        stop_requested = Event()

    errors = Queue()
    capture = ClientErrorCapture()
    client.logger.addHandler(capture)
    workers = []
    client._workers = workers

    def worker(target):
        try:
            target()
        except BaseException as exc:
            if client.running:
                errors.put(exc)

    try:
        if not math.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("time_limit_s must be a finite positive number")

        if stop_requested.is_set():
            return "user_stop"

        if on_state is not None:
            on_state("connecting")

        connect_policy(client)

        if on_state is not None:
            on_state("running")

        deadline = time.monotonic() + time_limit_s
        next_progress = time.monotonic() + 0.2

        for target in (
            client.receive_actions,
            lambda: client.control_loop(task=task),
        ):
            thread = Thread(
                target=worker,
                args=(target,),
                daemon=True,
            )
            thread.start()
            workers.append(thread)

        while True:
            try:
                error = errors.get_nowait()
            except Empty:
                pass
            else:
                raise error

            if capture.messages:
                message = "\n".join(capture.messages)
                if message.startswith((
                    "Error receiving actions:",
                    "Error sending observation",
                )):
                    raise PolicyConnectionError(message)
                raise RuntimeError(message)

            if stop_requested.is_set():
                return "user_stop"

            if not client.running:
                return "stopped"

            if any(not thread.is_alive() for thread in workers):
                raise RuntimeError("A policy worker ended unexpectedly")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "time_limit"

            for event in client.stall_events():
                if on_stall is not None:
                    on_stall(event)

            if (
                on_progress is not None
                and time.monotonic() >= next_progress
            ):
                on_progress()
                next_progress = time.monotonic() + 0.2

            stop_requested.wait(min(0.02, remaining))
    finally:
        try:
            client.stop()
        finally:
            try:
                client.start_barrier.abort()

                join_deadline = time.monotonic() + 5.0
                for thread in workers:
                    thread.join(
                        max(0.0, join_deadline - time.monotonic())
                    )

                if any(thread.is_alive() for thread in workers):
                    raise RuntimeError(
                        "Policy workers did not stop; hardware remains locked"
                    )
            finally:
                client.logger.removeHandler(capture)


def run_recorded_policy(
    client,
    *,
    task: str,
    time_limit_s: float,
    recorder: TrialRecorder,
    stop_requested: Event | None = None,
) -> str:
    connected = False

    def report(state: str) -> None:
        nonlocal connected
        connected = state == "running"

        recorder.event("accept" if connected else "connect")
        recorder.write_status(
            state=state,
            task=task,
            **client.metrics_snapshot(),
        )

    def progress() -> None:
        recorder.write_status(
            state="running",
            task=task,
            **client.metrics_snapshot(),
        )

    def record_stall(details: dict) -> None:
        recorder.event("stall", **details)

    def finish(state: str, reason: str, error=None) -> None:
        event = {
            "stopped": "stop",
            "aborted": "abort",
            "failed": "failure",
        }[state]

        for details in client.stall_events():
            record_stall(details)
        recorder.event(event, reason=reason, error=error)
        recorder.write_status(
            state=state,
            task=task,
            reason=reason,
            error=error,
            hold_command_sent=client._holding,
            arm_released=client._released,
            workers_stopped=all(
                not thread.is_alive()
                for thread in client._workers
            ),
            **client.metrics_snapshot(),
        )

    try:
        reason = run_policy(
            client,
            task=task,
            time_limit_s=time_limit_s,
            stop_requested=stop_requested,
            on_state=report,
            on_progress=progress,
            on_stall=record_stall,
        )
    except PolicyConnectionError as exc:
        finish(
            "aborted" if connected else "failed",
            "connection",
            str(exc),
        )
        raise
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            finish("stopped", "interrupted")
        else:
            finish("failed", "error", str(exc))
        raise
    else:
        finish("stopped", reason)
        return reason
