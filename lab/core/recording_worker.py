"""Web adapter around the installed LeRobot recorder; run only as a child process."""

import inspect
import signal
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
import json
from threading import Event, Thread

from lab.core.recording_telemetry import (
    FramePublisher,
    PREPARATION_SECONDS,
    TelemetryWriter,
    atomic_write,
)

def read_recording_command(
    root: Path,
    *,
    phase: str,
    episode: int,
    phase_id: str,
) -> str | None:
    command_path = root / "command.json"

    try:
        content = command_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    command_id = None
    action = None
    error = None

    try:
        payload = json.loads(content)

        if not isinstance(payload, dict):
            raise ValueError("Command must be an object")

        command_id = payload.get("id")

        if not isinstance(command_id, str) or not command_id:
            raise ValueError("Command ID is required")

        if payload.get("expected_phase") != phase:
            raise ValueError("Recording phase has changed")

        if payload.get("expected_episode") != episode:
            raise ValueError("Current episode has changed")

        if payload.get("expected_phase_id") != phase_id:
            raise ValueError("Recording attempt has changed")

        allowed_actions = {
            "preparing": (
                "start_episode",
            ),
            "recording": (
                "finish_episode",
                "retry_episode",
            ),
            "resetting": (
                "start_episode",
            ),
        }

        requested_action = payload.get("action")

        if requested_action not in allowed_actions.get(phase, ()):
            raise ValueError(
                "Command is not allowed in the current phase"
            )

        action = requested_action

    except ValueError as exc:
        error = str(exc)

    atomic_write(
        root / "command_result.json",
        json.dumps({
            "id": command_id,
            "status": "rejected" if error else "accepted",
            "error": error,
        }).encode("utf-8"),
    )

    command_path.unlink()
    return action


@contextmanager
def recording_commands(root, *, phase, episode, events, phase_id=None):
    if root is None:
        yield
        return

    if phase_id is None:
        raise ValueError("Recording commands require a phase ID")

    done = Event()
    errors = []

    def watch():
        try:
            while not done.is_set():
                if events["stop_recording"]:
                    return

                action = read_recording_command(
                    root,
                    phase=phase,
                    episode=episode,
                    phase_id=phase_id,
                )

                if action is not None:
                    if action == "retry_episode":
                        events["rerecord_episode"] = True

                    events["exit_early"] = True
                    return

                done.wait(0.05)

        except Exception as exc:
            errors.append(exc)
            events["exit_early"] = True

    watcher = Thread(target=watch, daemon=True)
    watcher.start()

    try:
        yield
    finally:
        done.set()
        watcher.join()

        if not events["stop_recording"]:
            events["exit_early"] = False

    if errors:
        raise errors[0]


@contextmanager
def recording_hooks(
    module,
    writer,
    publisher,
    events,
    *,
    command_root: Path | None = None,
):
    original_loop = module.record_loop
    original_factory = module.make_robot_from_config
    original_listener = module.init_keyboard_listener
    original_save = module.LeRobotDataset.save_episode

    dataset_config = getattr(module, "DatasetRecordConfig", None)
    original_stamp = getattr(dataset_config, "stamp_repo_id", None)

    target = inspect.unwrap(original_loop)
    signature = inspect.signature(target)

    required = {"robot", "events", "dataset", "control_time_s"}

    # LeRobot 0.6.0's image-writer wrapper omits functools.wraps.
    if not required.issubset(signature.parameters):
        wrapped = inspect.getclosurevars(target).nonlocals.get("func")
        if callable(wrapped):
            signature = inspect.signature(wrapped)

    if not required.issubset(signature.parameters):
        raise RuntimeError("Unsupported LeRobot record_loop signature")

    observations = []
    first = True

    def make_robot(config):
        if events["stop_recording"]:
            raise KeyboardInterrupt

        robot = original_factory(config)
        original_observation = robot.get_observation

        def observe():
            observation = original_observation()
            publisher.submit(observation)
            return observation

        observations.append((robot, original_observation))
        robot.get_observation = observe
        return robot

    def loop(*args, **kwargs):
        nonlocal first

        bound = signature.bind(*args, **kwargs)
        dataset = bound.arguments.get("dataset")
        duration = bound.arguments["control_time_s"]
        writer.current = writer.saved + 1

        if first and dataset is not None:
            first = False
            writer.stage("preparing", PREPARATION_SECONDS)

            preparation = dict(
                bound.arguments,
                dataset=None,
                control_time_s=PREPARATION_SECONDS,
            )

            with recording_commands(
                command_root,
                phase="preparing",
                episode=writer.current,
                events=events,
                phase_id=writer.phase_id,
            ):
                original_loop(**preparation)

        if events["stop_recording"]:
            raise KeyboardInterrupt

        phase = (
            "recording"
            if dataset is not None
            else "resetting"
        )
        writer.stage(phase, duration)

        with recording_commands(
            command_root,
            phase=phase,
            episode=writer.current,
            events=events,
            phase_id=writer.phase_id,
        ):
            result = original_loop(*args, **kwargs)

        if dataset is not None:
            writer.stage("saving")

        return result

    def save(dataset, *args, **kwargs):
        writer.stage("saving")
        result = original_save(dataset, *args, **kwargs)
        writer.saved += 1
        writer.stage("saving")
        return result

    module.record_loop = loop
    module.make_robot_from_config = make_robot
    module.init_keyboard_listener = lambda: (None, events)
    module.LeRobotDataset.save_episode = save

    if original_stamp is not None:
        dataset_config.stamp_repo_id = lambda self: None

    try:
        yield
    finally:
        module.record_loop = original_loop
        module.make_robot_from_config = original_factory
        module.init_keyboard_listener = original_listener
        module.LeRobotDataset.save_episode = original_save

        if original_stamp is not None:
            dataset_config.stamp_repo_id = original_stamp

        for robot, observe in observations:
            robot.get_observation = observe


def main() -> int:
    root = Path(sys.argv.pop(1))
    writer = TelemetryWriter(root)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def stop(_signum, _frame):
        events["stop_recording"] = True
        events["exit_early"] = True

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)
    publisher = FramePublisher(root)
    publisher.start()
    try:
        version = metadata.version("lerobot")
        if version not in {"0.6.0", "0.6.1"}:
            raise RuntimeError(f"Web recording supports LeRobot 0.6.0/0.6.1; found {version}")
        from lerobot.scripts import lerobot_record
        if "no_stamp" not in lerobot_record.DatasetRecordConfig.__dataclass_fields__:
            sys.argv = [arg for arg in sys.argv if not arg.startswith("--dataset.no_stamp=")]
        # Web owns controls; desktop listeners/viewers must not contend with it.
        sys.argv.extend(["--display_data=false", "--play_sounds=false"])
        with recording_hooks(
            lerobot_record,
            writer,
            publisher,
            events,
            command_root=root,
        ):
            lerobot_record.main()
        writer.stage("stopped" if events["stop_recording"] else "completed")
        return 0
    except KeyboardInterrupt:
        writer.stage("stopped")
        return 130
    except EOFError as exc:
        message = "LeRobot requested terminal input. Complete both arms' calibration in Devices before recording."
        writer.stage("failed", error=message)
        raise RuntimeError(message) from exc
    except Exception as exc:
        writer.stage("failed", error=str(exc))
        raise
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
