"""Terminal entry to the hardware layer, so the arms can be tested without the UI.

    python -m lab.core.hardware.cli ports
    python -m lab.core.hardware.cli detect --arm=follower [--save]
    python -m lab.core.hardware.cli calibrate --arm=leader
    python -m lab.core.hardware.cli teleop [--seconds=30]

Uses the same HardwareService as the server, so the same lock applies: this
refuses to run while the server or experiments/collect.py holds the arms.
MTREA_HARDWARE=fake runs it against fake arms.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from lab.core.hardware import ARMS, HardwareError
from lab.core.hardware.calibration import ACTIVE, HOMING, RECORDING
from lab.core.hardware.service import HardwareService


def _ports(svc: HardwareService, _args) -> int:
    config = svc.config()
    print(f"backend: {config['backend']}")
    ports = config["ports"]
    print("serial ports:" if ports else "serial ports: none found")
    for port in ports or []:
        print(f"  {port}")
    for arm, info in config["arms"].items():
        if "error" in info:
            print(f"{arm}: {info['error']}")
            continue
        present = {True: "present", False: "NOT present", None: "unknown"}[info["port_present"]]
        calibrated = "yes" if info["calibration_present"] else "no"
        print(f"{arm}: port {info['port']} ({present}), id {info['id']}, calibration file: {calibrated} "
              f"({info['calibration_file']})")
        if info["legacy_calibration_present"] and not info["calibration_present"]:
            print(f"  note: a pre-0.6 calibration exists for id {info['id']}, but lerobot v0.6.1 does not read it")
    return 0


def _detect(svc: HardwareService, args) -> int:
    started = svc.detect_start(args.arm)
    print(f"backend: {started['backend']}; {len(started['ports_before'])} ports now")
    print(f"{started['instruction']} Waiting up to 15 s...", flush=True)
    found = svc.detect_finish(args.arm)
    print(f"{args.arm} port: {found['port']}")
    print(f"Plug the {args.arm} arm back in.")
    if args.save:
        saved = svc.save_port(args.arm, found["port"])
        print(f"saved to {saved['config_file']}")
    else:
        print(f"Not saved; rerun with --save, or: python -m lab.core.hardware.cli detect --arm={args.arm} --save")
    return 0


def _print_joints(status: dict) -> None:
    for joint, j in status["joints"].items():
        if j["recorded"]:
            print(f"  {joint:<14} {j['current']:>5}   min {j['min']:>5}   max {j['max']:>5}   "
                  f"range {j['max'] - j['min']:>5}")
        else:
            print(f"  {joint:<14} {j['current']:>5}")


def _read_enter(pressed: threading.Event, closed: threading.Event) -> None:
    try:
        input()
        pressed.set()
    except EOFError:
        closed.set()


def _calibrate(svc: HardwareService, args) -> int:
    session = svc.calibration
    session.start(args.arm)
    shown_state = None
    try:
        while True:
            status = session.status()
            if status["state"] != shown_state:
                shown_state = status["state"]
                print(f"\n[{status['backend']}] {shown_state}: {status['instruction']}", flush=True)
            if status["state"] in (HOMING, RECORDING):
                pressed, closed = threading.Event(), threading.Event()
                threading.Thread(target=_read_enter, args=(pressed, closed), daemon=True).start()
                print("(press Enter for Next, Ctrl+C to cancel)", flush=True)
                while not (pressed.wait(1.5) or closed.is_set()):
                    status = session.status()
                    if status["state"] != shown_state:
                        break
                    _print_joints(status)
                    print(flush=True)
                if closed.is_set():
                    raise KeyboardInterrupt  # stdin closed: nobody can press Next
                if pressed.is_set():
                    session.next()
                    # Wait for the worker to leave this step, so one Enter is one Next.
                    while session.status()["state"] == shown_state:
                        time.sleep(0.05)
                continue
            if status["state"] not in ACTIVE:
                _print_joints(status)
                return 0 if status["state"] == "done" else 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        if session.active:
            session.cancel()
        print(f"\n{session.status()['instruction']}")
        return 130


def _teleop(svc: HardwareService, args) -> int:
    session = svc.teleop
    status = session.start()
    print(f"[{status['backend']}] teleoperation running at target {status['target_fps']} fps; "
          f"max_relative_target {status['max_relative_target']}")
    print(f"Note: {status['torque_on_stop']}.")
    print("Ctrl+C stops." if not args.seconds else f"Stops after {args.seconds} s (Ctrl+C stops earlier).")
    latest: dict = {}
    unsubscribe = svc.stream.subscribe(lambda frame: latest.update(frame))
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while session.running and (deadline is None or time.monotonic() < deadline):
            time.sleep(1.0)
            status = session.status()
            sample = ""
            if latest.get("follower"):
                sample = "  follower " + " ".join(f"{j[:5]}={v:6.1f}" for j, v in latest["follower"].items())
            print(f"fps {status['fps']:>3}{sample}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        unsubscribe()
        if session.running:
            session.stop()
    status = session.status()
    if status["error"]:
        print(status["error"], file=sys.stderr)
        return 1
    print("teleoperation stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.core.hardware.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="list serial ports and the configured arms")
    detect = sub.add_parser("detect", help="find an arm's port by unplugging it")
    detect.add_argument("--arm", required=True, choices=ARMS)
    detect.add_argument("--save", action="store_true", help="write the port into robot/configs/so101_<arm>.yaml")
    calibrate = sub.add_parser("calibrate", help="calibrate one arm, step by step")
    calibrate.add_argument("--arm", required=True, choices=ARMS)
    teleop = sub.add_parser("teleop", help="leader -> follower teleoperation")
    teleop.add_argument("--seconds", type=float, default=0, help="stop after this long (default: until Ctrl+C)")
    args = parser.parse_args(argv)

    commands = {"ports": _ports, "detect": _detect, "calibrate": _calibrate, "teleop": _teleop}
    try:
        svc = HardwareService()
        return commands[args.command](svc, args)
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
