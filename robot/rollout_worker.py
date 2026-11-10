"""Dedicated robot owner: Stop retains ownership until an explicit Release."""

import json
import os
import signal
import time
from threading import Event, Thread
from uuid import uuid4

import psutil

from lab.core.hardware.lock import HardwareLock
from robot.rollout_client import (
    build_client_config, build_parser, create_locked_client, run_recorded_policy,
)
from robot.rollout_commands import CommandInbox
from robot.rollout_preview import TrialPreview
from robot.rollout_records import TrialRecorder, build_trial_metadata


def create_real_client(config, lock, trial_dir):
    # RobotClient creates a log file at import time. This runs before any workers.
    previous = os.getcwd()
    try:
        os.chdir(trial_dir)
        from lerobot.async_inference import robot_client as module
    finally:
        os.chdir(previous)

    original = module.make_robot_from_config

    def make_calibrated_robot(robot_config):
        robot = original(robot_config)
        if not robot.calibration:
            raise RuntimeError("No follower calibration; calibrate in Devices first")

        def refuse_calibration():
            raise RuntimeError("Follower calibration mismatch; check calibration in Devices")

        # Native connect calls calibrate on mismatch, which would otherwise prompt.
        robot.calibrate = refuse_calibration
        return robot

    module.make_robot_from_config = make_calibrated_robot
    try:
        return create_locked_client(config, lock=lock, client_type=module.RobotClient)
    finally:
        module.make_robot_from_config = original


def run_process(config, *, trial_dir, time_limit_s, client_factory=create_real_client,
                lock=None, stop_requested=None):
    recorder = TrialRecorder(trial_dir)
    token = uuid4().hex
    recorder.set_context(trial_token=token, pid=os.getpid(), process_alive=True,
                         process_started=psutil.Process().create_time())
    recorder.write_session(build_trial_metadata(config, time_limit_s=time_limit_s))
    recorder.update_status()
    inbox = CommandInbox(recorder.root, token)
    stop = stop_requested if stop_requested is not None else Event()
    finished = Event()
    lock = lock if lock is not None else HardwareLock()
    client = None
    failed = False
    pending_stops = []
    interrupted_channel = False

    def execute():
        nonlocal failed
        try:
            run_recorded_policy(
                client, task=config.task, time_limit_s=time_limit_s,
                recorder=recorder, stop_requested=stop,
                preview=TrialPreview(recorder.root), session_prepared=True,
            )
        except BaseException:
            failed = True
        finally:
            finished.set()

    try:
        if stop.is_set():
            recorder.update_status(state="stopped", reason="user_stop",
                                   process_alive=False, arm_released=True)
            return 0
        client = client_factory(config, lock, recorder.root)
    except Exception as exc:
        recorder.event("failure", reason="initialization", error=str(exc))
        recorder.update_status(state="failed", error=str(exc), process_alive=False,
                               arm_released=not lock.held)
        return 1

    worker = Thread(target=execute, name="rollout-policy", daemon=True)
    worker.start()
    try:
        while True:
            if finished.is_set():
                for command_id in pending_stops:
                    inbox.ack(command_id, "completed" if not failed else "failed",
                              "Policy cleanup failed; inspect status" if failed else None)
                pending_stops.clear()
            for command in inbox.poll():
                command_id = command["id"]
                if command["action"] == "stop":
                    stop.set()
                    if finished.is_set():
                        inbox.ack(command_id, "completed" if not failed else "failed")
                    else:
                        pending_stops.append(command_id)
                        # Consumption is distinct from completed worker cleanup.
                        inbox.ack(command_id, "accepted")
                elif not finished.is_set():
                    inbox.ack(command_id, "rejected", "Stop must finish before Release")
                else:
                    try:
                        client.release_arm()
                    except Exception as exc:
                        recorder.event("release_failed", error=str(exc))
                        recorder.update_status(release_error=str(exc))
                        inbox.ack(command_id, "failed", str(exc))
                    else:
                        recorder.event("release")
                        recorder.update_status(arm_released=True, process_alive=False,
                                               release_error=None)
                        inbox.ack(command_id, "completed")
                        return 1 if failed else 0
            if stop.is_set() and not interrupted_channel:
                # Cancel pending gRPC calls; motor operations remain in the policy worker.
                client.channel.close()
                interrupted_channel = True
            time.sleep(0.05)
    finally:
        # Normal exit is only possible after Release. Signals request Stop, not exit.
        worker.join(timeout=0)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_client_config(args)
    stop = Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        return run_process(config, trial_dir=args.trial_dir,
                           time_limit_s=args.time_limit_s, stop_requested=stop)
    except Exception as exc:
        print(json.dumps({"state": "failed", "error": str(exc)}), flush=True)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
