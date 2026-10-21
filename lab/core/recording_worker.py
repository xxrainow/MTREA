"""Web adapter around the installed LeRobot recorder; run only as a child process."""

import inspect
import signal
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from lab.core.recording_telemetry import FramePublisher, PREPARATION_SECONDS, TelemetryWriter


@contextmanager
def recording_hooks(module, writer, publisher, events):
    original_loop = module.record_loop
    original_factory = module.make_robot_from_config
    original_listener = module.init_keyboard_listener
    original_save = module.LeRobotDataset.save_episode
    dataset_config = getattr(module, "DatasetRecordConfig", None)
    original_stamp = getattr(dataset_config, "stamp_repo_id", None)
    target = inspect.unwrap(original_loop)
    signature = inspect.signature(target)
    # LeRobot's safe_stop_image_writer omits functools.wraps in 0.6.0.
    if not {"robot", "events", "dataset", "control_time_s"}.issubset(signature.parameters):
        wrapped = inspect.getclosurevars(target).nonlocals.get("func")
        if callable(wrapped):
            signature = inspect.signature(wrapped)
    if not {"robot", "events", "dataset", "control_time_s"}.issubset(signature.parameters):
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
        if first and dataset is not None:
            first = False
            writer.stage("preparing", PREPARATION_SECONDS)
            preparation = dict(bound.arguments, dataset=None, control_time_s=PREPARATION_SECONDS)
            original_loop(**preparation)
        if events["stop_recording"]:
            raise KeyboardInterrupt
        writer.current = writer.saved + 1
        writer.stage("recording" if dataset is not None else "resetting", duration)
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
        # The web config already fixes identity; never add a timestamp suffix.
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
        with recording_hooks(lerobot_record, writer, publisher, events):
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
