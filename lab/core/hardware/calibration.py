"""Step-wise calibration of one arm, driven by Next presses instead of Enter.

    idle -> connecting -> homing -> recording_ranges -> saving -> done
                                    any active state  -> error | cancelled

The steps are those of SOFollower/SOLeader.calibrate() in lerobot v0.6.1:
torque off, position mode, half-turn homing at the pose the operator chose,
then min/max recording for every joint except wrist_roll, which is a full-turn
joint and gets 0..4095. A worker thread owns the serial port; start(), next()
and cancel() only signal it, so none of them block on the arm.

From LeLab's calibrate.py: invalid readings (outside 0..5000) are ignored, a
single-frame jump over 2000 steps is an encoder wrap-around (the arm was not
started near the middle), and a range under 100 steps means the joint was not
moved. Either refuses to save and names the joint.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

from lab.core.hardware import HardwareError, JointStream, SessionStateError, check_arm
from lab.core.hardware.devices import CONFIGS_DIR, load_arm_config
from lab.core.hardware.lock import HardwareLock
from robot.config import JOINT_ORDER

FULL_TURN_JOINT = "wrist_roll"  # as SOFollower.calibrate(): not recorded, range 0..4095
FULL_TURN_RANGE = (0, 4095)
VALID_RANGE = (0, 5000)  # exclusive; Present_Position is 12-bit
MAX_JUMP = 2000
MIN_RANGE = 100

IDLE, CONNECTING, HOMING, RECORDING, SAVING = "idle", "connecting", "homing", "recording_ranges", "saving"
DONE, ERROR, CANCELLED = "done", "error", "cancelled"
ACTIVE = (CONNECTING, HOMING, RECORDING, SAVING)


class _Cancelled(Exception):
    pass


def _instruction(state: str, arm: str | None, detail: str) -> str:
    return {
        IDLE: "Choose an arm to calibrate.",
        CONNECTING: f"Connecting to the {arm} arm...",
        HOMING: (f"Torque is off: hold the {arm} arm. Move it to the middle of its range of motion, "
                 "every joint roughly centred, then press Next."),
        RECORDING: (f"Move every joint of the {arm} arm except {FULL_TURN_JOINT} slowly through its full "
                    "range, end to end, then press Next."),
        SAVING: "Checking the ranges and saving...",
        DONE: f"Calibration saved to {detail}.",
        ERROR: detail,
        CANCELLED: "Calibration cancelled. Nothing was saved.",
    }[state]


def assess_joint_range(
    joint: str,
    minimum: int | None,
    maximum: int | None,
) -> dict:
    if joint not in JOINT_ORDER:
        raise ValueError(f"Unknown joint: {joint}")

    span = (
        maximum - minimum
        if minimum is not None and maximum is not None
        else None
    )

    if joint == FULL_TURN_JOINT:
        span = None
        status = "fixed"
        message = "Uses a fixed range; movement measurement is not required."
        range_ok = True

    elif span is None:
        status = "missing"
        message = "No valid range readings."
        range_ok = False

    elif span < 0:
        status = "invalid"
        message = "Maximum is smaller than minimum."
        range_ok = False

    elif span < MIN_RANGE:
        status = "insufficient"
        message = (
            f"Moved {span} steps; "
            f"at least {MIN_RANGE} steps are required."
        )
        range_ok = False

    else:
        status = "minimum_met"
        message = (
            "Minimum movement check passed. "
            "Full-range coverage is not verified."
        )
        range_ok = True

    return {
        "span": span,
        "range_status": status,
        "range_message": message,
        "range_ok": range_ok,
    }
    

class CalibrationSession:
    def __init__(self, backend: Any, lock: HardwareLock, stream: JointStream,
                 configs_dir: Path = CONFIGS_DIR, poll_s: float = 0.05):
        self.backend, self.lock, self.stream = backend, lock, stream
        self.configs_dir, self.poll_s = configs_dir, poll_s
        self._mutex = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._reset(IDLE, None)

    def _reset(self, state: str, arm: str | None) -> None:
        self._state, self._arm = state, arm
        self._advance_from: str | None = None  # the state a Next press applies to
        self._detail = ""
        self._current: dict[str, int] = {}
        self._mins: dict[str, int] = {}
        self._maxes: dict[str, int] = {}
        self._saved_path: Path | None = None

    # --- control (any thread) ---

    def start(self, arm: str) -> dict:
        check_arm(arm)
        with self._mutex:
            if self._state in ACTIVE:
                raise SessionStateError(f"calibration of the {self._arm} arm is already running")
            cfg = load_arm_config(arm, self.configs_dir)
            self.lock.acquire(f"calibration of the {arm} arm")
            self._reset(CONNECTING, arm)
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, args=(cfg,), name=f"calibrate-{arm}", daemon=True)
            self._thread.start()
        return self.status()

    def next(self) -> dict:
        with self._mutex:
            if self._state not in (HOMING, RECORDING):
                raise SessionStateError(f"Next only applies while homing or recording ranges; "
                                        f"calibration is {self._state}")
            self._advance_from = self._state
        return self.status()

    def cancel(self, timeout_s: float = 5.0) -> dict:
        with self._mutex:
            if self._state not in ACTIVE:
                raise SessionStateError(f"no calibration is running (state: {self._state})")
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        return self.status()

    def shutdown(self) -> None:
        if self.active:
            self.cancel()

    @property
    def active(self) -> bool:
        return self._state in ACTIVE

    def status(self) -> dict:
        with self._mutex:
            state, arm = self._state, self._arm
            joints = {}
        
            if self._current:
                for joint in JOINT_ORDER:
                    recorded = (
                        state != HOMING
                        and joint != FULL_TURN_JOINT
                        and joint in self._mins
                        and joint in self._maxes
                    )
        
                    minimum = self._mins[joint] if recorded else None
                    maximum = self._maxes[joint] if recorded else None
        
                    assessment = assess_joint_range(
                        joint,
                        minimum,
                        maximum,
                    )
        
                    joints[joint] = {
                        "current": self._current.get(joint),
                        "min": minimum,
                        "max": maximum,
                        "recorded": recorded,
                        **assessment,
                    }
        
            range_check_passed = (
                state == RECORDING
                and len(joints) == len(JOINT_ORDER)
                and all(
                    value["range_ok"]
                    for value in joints.values()
                )
            )
        
            return {
                "backend": self.backend.name,
                "state": state,
                "arm": arm,
                "active": state in ACTIVE,
                "can_next": state in (HOMING, RECORDING),
                "instruction": _instruction(state, arm, self._detail),
                "error": self._detail if state == ERROR else None,
                "saved_path": (
                    str(self._saved_path)
                    if self._saved_path
                    else None
                ),
                "joints": joints,
                "range_check_passed": range_check_passed,
            }

    # --- worker ---

    def _set(self, state: str, detail: str = "") -> None:
        with self._mutex:
            self._state, self._detail = state, detail

    def _wait_for_next(self, arm: Any, step: str, on_read) -> None:
        while self._advance_from != step:
            if self._cancel.is_set():
                raise _Cancelled
            positions = arm.read_raw()
            with self._mutex:
                on_read(positions)
            self._publish()
            time.sleep(self.poll_s)

    def _publish(self) -> None:
        status = self.status()
        self.stream.publish({"type": "calibration", "t": time.time(), "arm": status["arm"],
                             "state": status["state"], "joints": status["joints"]})

    def _run(self, cfg) -> None:
        arm = None
        final, detail = ERROR, "calibration stopped unexpectedly"
        try:
            arm = self.backend.calibration_arm(cfg)
            arm.connect()
            if self._cancel.is_set():
                raise _Cancelled
            self._set(HOMING)
            self._wait_for_next(arm, HOMING, lambda p: self._current.update(p))
            homing = arm.set_half_turn_homings()

            start = {j: v for j, v in arm.read_raw().items() if VALID_RANGE[0] < v < VALID_RANGE[1]}
            with self._mutex:
                self._current = dict(start)
                self._mins = {j: v for j, v in start.items() if j != FULL_TURN_JOINT}
                self._maxes = dict(self._mins)
                self._state = RECORDING
            self._wait_for_next(arm, RECORDING, self._record)

            self._set(SAVING)
            problems = self._range_problems()
            if problems:
                raise HardwareError("Not saved: " + "; ".join(problems) + ". Start the calibration again.")
            mins = {**self._mins, FULL_TURN_JOINT: FULL_TURN_RANGE[0]}
            maxes = {**self._maxes, FULL_TURN_JOINT: FULL_TURN_RANGE[1]}
            self._saved_path = arm.save(homing, mins, maxes)
            final, detail = DONE, str(self._saved_path)
        except _Cancelled:
            final, detail = CANCELLED, ""
        except HardwareError as exc:
            final, detail = ERROR, str(exc)
        except Exception as exc:
            final, detail = ERROR, f"calibration failed: {exc.__class__.__name__}: {exc}"
            traceback.print_exc()
        finally:
            try:
                if arm is not None:
                    arm.disconnect()
            finally:
                self.lock.release()
                self._set(final, detail)
                self._publish()

    def _record(self, positions: dict[str, int]) -> None:
        """Runs under the mutex: fold one reading into min/max, catching wrap-arounds."""
        for joint, pos in positions.items():
            if not VALID_RANGE[0] < pos < VALID_RANGE[1]:
                continue
            previous = self._current.get(joint)
            self._current[joint] = pos
            if joint == FULL_TURN_JOINT or joint not in self._mins:
                continue
            if previous is not None and abs(pos - previous) > MAX_JUMP:
                raise HardwareError(
                    f"Not saved: {joint} jumped from {previous} to {pos} (encoder wrap-around). Put the arm "
                    "in the middle of its range, every joint centred, and start the calibration again.")
            self._mins[joint] = min(self._mins[joint], pos)
            self._maxes[joint] = max(self._maxes[joint], pos)

        
    def _range_problems(self) -> list[str]:
        problems = []

        for joint in JOINT_ORDER:
            assessment = assess_joint_range(
                joint,
                self._mins.get(joint),
                self._maxes.get(joint),
            )

            if assessment["range_ok"]:
                continue

            status = assessment["range_status"]

            if status == "missing":
                problems.append(
                    f"{joint} gave no valid reading"
                )
            elif status == "insufficient":
                span = assessment["span"]
                problems.append(
                    f"{joint} moved only {span} steps "
                    f"(needs at least {MIN_RANGE})"
                )
            else:
                problems.append(
                    f"{joint}: {assessment['range_message']}"
                )

        return problems
