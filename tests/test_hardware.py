"""lab/core/hardware with fake arms: ports, lock, calibration, teleop. No lerobot, no robot."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import yaml

from lab.core.hardware import HardwareBusy, HardwareError, PortDetectionError, SessionStateError
from lab.core.hardware.calibration import FULL_TURN_JOINT
from lab.core.hardware.devices import CONFIGS_DIR, calibration_path, load_arm_config
from lab.core.hardware.fake import FakeBackend, default_trajectory, fake_calibration_path
from lab.core.hardware.lock import HardwareLock
from lab.core.hardware.ports import PortDetection, save_port
from lab.core.hardware.service import HardwareService
from robot.config import JOINT_ORDER


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("HF_HOME", "HF_LEROBOT_HOME", "HF_LEROBOT_CALIBRATION", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def configs(tmp_path):
    target = tmp_path / "configs"
    target.mkdir()
    for arm in ("follower", "leader"):
        shutil.copyfile(CONFIGS_DIR / f"so101_{arm}.yaml", target / f"so101_{arm}.yaml")
    return target


def make_service(tmp_path, configs, trajectory=default_trajectory):
    return HardwareService(data_base=tmp_path / "data", backend=FakeBackend(trajectory), configs_dir=configs)


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


def wait_state(session, state, timeout=5.0):
    return wait_for(lambda: session.status() if session.status()["state"] == state else None, timeout)


# --- ports ---

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def detection(snapshots):
    lists = iter(snapshots)
    last = {}

    def lister():
        last["v"] = next(lists, last.get("v"))
        return last["v"]

    clock = Clock()
    return PortDetection(lister=lister, sleep=clock.sleep, clock=clock)


def test_detection_finds_the_one_removed_port():
    d = detection([["COM1", "COM3", "COM4"], ["COM1", "COM3", "COM4"], ["COM1", "COM4"]])
    assert d.start() == ["COM1", "COM3", "COM4"]
    assert d.finish() == "COM3"


def test_detection_times_out_when_nothing_is_unplugged():
    d = detection([["COM1", "COM3"]])
    d.start()
    with pytest.raises(PortDetectionError, match="no port disappeared within 15 s"):
        d.finish()


def test_detection_refuses_several_removed_ports():
    d = detection([["COM1", "COM3", "COM4"], ["COM1"]])
    d.start()
    with pytest.raises(PortDetectionError, match="more than one port disappeared"):
        d.finish()


def test_detection_finish_needs_start():
    with pytest.raises(SessionStateError):
        detection([["COM1"]]).finish()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_save_port_rewrites_only_the_port_line(configs, newline):
    path = configs / "so101_follower.yaml"
    original = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", newline)
    path.write_bytes(original.encode("utf-8"))

    save_port("follower", "COM7", configs)

    after = path.read_bytes().decode("utf-8")
    before_lines, after_lines = original.split(newline), after.split(newline)
    changed = [(b, a) for b, a in zip(before_lines, after_lines) if b != a]
    assert len(before_lines) == len(after_lines)
    assert changed == [(next(line for line in before_lines if line.startswith("port:")), "port: COM7")]
    assert "\r\n" in after if newline == "\r\n" else "\r\n" not in after
    assert load_arm_config("follower", configs).port == "COM7"


def test_save_port_keeps_an_inline_comment_and_quotes_when_needed(configs):
    path = configs / "so101_leader.yaml"
    text = path.read_text(encoding="utf-8")
    port_line = next(line for line in text.splitlines() if line.startswith("port:"))
    path.write_text(text.replace(port_line, "port: /dev/old   # USB hub, left slot"), encoding="utf-8")
    save_port("leader", "/dev/tty: odd #name", configs)
    new_line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("port:"))
    assert new_line == 'port: "/dev/tty: odd #name"   # USB hub, left slot'
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["port"] == "/dev/tty: odd #name"


def test_save_port_refuses_a_file_without_one_port_line(configs):
    path = configs / "so101_leader.yaml"
    path.write_text("type: so101_leader\nid: x\nuse_degrees: true\n", encoding="utf-8")
    with pytest.raises(HardwareError, match="exactly one top-level 'port:' line"):
        save_port("leader", "COM3", configs)


def test_max_relative_target_int_becomes_float(configs):
    path = configs / "so101_follower.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("max_relative_target: null", "max_relative_target: 10"),
                    encoding="utf-8")
    assert load_arm_config("follower", configs).max_relative_target == 10.0


# --- lock ---

def test_second_session_refused_in_process(tmp_path):
    path = tmp_path / "locks" / "hardware.lock"
    first, second = HardwareLock(path), HardwareLock(path)
    first.acquire("calibration of the follower arm")
    with pytest.raises(HardwareBusy, match="calibration of the follower arm"):
        second.acquire("teleoperation")
    first.release()
    assert not path.exists()
    second.acquire("teleoperation")
    second.release()


def _write_lock(path, pid, create_time, owner="experiments/collect.py"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "create_time": create_time, "owner": owner, "since": "earlier"}))


def test_stale_lock_file_is_reclaimed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    path = tmp_path / "hardware.lock"
    _write_lock(path, dead.pid, 0.0)
    lock = HardwareLock(path)
    assert lock.holder() is None
    lock.acquire("teleoperation")
    assert json.loads(path.read_text())["owner"] == "teleoperation"
    lock.release()


def test_live_pid_lock_refused_and_reused_pid_reclaimed(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = tmp_path / "hardware.lock"
        _write_lock(path, other.pid, psutil.Process(other.pid).create_time())
        lock = HardwareLock(path)
        assert lock.holder()["pid"] == other.pid
        with pytest.raises(HardwareBusy, match=f"experiments/collect.py \\(pid {other.pid}"):
            lock.acquire("teleoperation")
        assert not lock.held

        # Same pid but a different start time: the pid was reused, the lock is stale.
        _write_lock(path, other.pid, 12345.0)
        lock.acquire("teleoperation")
        lock.release()
    finally:
        other.kill()
        other.wait()


# --- calibration ---

def test_calibration_happy_path_writes_a_calibration_file(tmp_path, configs, home):
    svc = make_service(tmp_path, configs)
    session = svc.calibration
    status = session.start("follower")
    assert status["backend"] == "fake" and status["state"] in ("connecting", "homing")
    wait_state(session, "homing")
    assert svc.lock.holder()["owner"] == "calibration of the follower arm"
    session.next()
    wait_state(session, "recording_ranges")
    time.sleep(0.5)
    status = session.status()
    assert status["joints"]["shoulder_pan"]["recorded"] and not status["joints"][FULL_TURN_JOINT]["recorded"]
    session.next()
    status = wait_state(session, "done")

    expected = fake_calibration_path("follower", "so-arm1")
    assert expected.parent == calibration_path("follower", "so-arm1").parent
    assert expected.is_relative_to(home / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower")
    assert status["saved_path"] == str(expected)
    saved = json.loads(expected.read_text())
    assert list(saved) == JOINT_ORDER
    assert (saved[FULL_TURN_JOINT]["range_min"], saved[FULL_TURN_JOINT]["range_max"]) == (0, 4095)
    assert all(saved[j]["range_max"] - saved[j]["range_min"] >= 100 for j in JOINT_ORDER)
    assert svc.lock.holder() is None and not svc.lock.path.exists()


def test_next_in_a_wrong_state_is_refused(tmp_path, configs, home):
    svc = make_service(tmp_path, configs)
    with pytest.raises(SessionStateError, match="calibration is idle"):
        svc.calibration.next()
    svc.calibration.start("leader")
    with pytest.raises(SessionStateError, match="already running"):
        svc.calibration.start("follower")
    svc.calibration.cancel()


def test_cancel_releases_the_lock(tmp_path, configs, home):
    svc = make_service(tmp_path, configs)
    svc.calibration.start("leader")
    wait_state(svc.calibration, "homing")
    status = svc.calibration.cancel()
    assert status["state"] == "cancelled"
    assert svc.lock.holder() is None
    with pytest.raises(SessionStateError):
        svc.calibration.cancel()
    svc.teleop.start()  # the arms are free again
    svc.teleop.stop()
    assert not fake_calibration_path("leader", "so-arm1").exists()


def test_discontinuity_refuses_to_save_and_names_the_joint(tmp_path, configs, home):
    def jumpy(t):
        positions = default_trajectory(t)
        if t > 0.6:  # alternating frames: a > 2000-step jump whatever the timing
            positions["shoulder_lift"] = 4600 if int(t * 20) % 2 else 1000
        return positions

    svc = make_service(tmp_path, configs, jumpy)
    svc.calibration.start("follower")
    wait_state(svc.calibration, "homing")
    svc.calibration.next()
    status = wait_state(svc.calibration, "error")
    assert "shoulder_lift jumped" in status["error"] and "Not saved" in status["error"]
    assert not fake_calibration_path("follower", "so-arm1").exists()
    assert svc.lock.holder() is None


def test_too_small_range_refuses_to_save_and_names_the_joint(tmp_path, configs, home):
    def stiff_elbow(t):
        return {**default_trajectory(t), "elbow_flex": 2047}

    svc = make_service(tmp_path, configs, stiff_elbow)
    svc.calibration.start("leader")
    wait_state(svc.calibration, "homing")
    svc.calibration.next()
    wait_state(svc.calibration, "recording_ranges")
    time.sleep(0.3)
    svc.calibration.next()
    status = wait_state(svc.calibration, "error")
    assert "elbow_flex moved only 0 steps" in status["error"]
    assert "shoulder_pan" not in status["error"]
    assert not fake_calibration_path("leader", "so-arm1").exists()


# --- teleop ---

def test_teleop_streams_frames_and_reports_fps(tmp_path, configs, home):
    svc = make_service(tmp_path, configs)
    frames = []
    unsubscribe = svc.stream.subscribe(frames.append)
    status = svc.teleop.start()
    assert status["running"] and status["backend"] == "fake" and status["target_fps"] == 30
    assert "torque" in status["torque_on_stop"]
    time.sleep(1.2)
    status = svc.teleop.status()
    assert 20 <= status["fps"] <= 35
    unsubscribe()
    status = svc.teleop.stop()
    assert not status["running"] and status["error"] is None
    assert svc.lock.holder() is None
    frame = frames[-1]
    assert frame["type"] == "teleop" and isinstance(frame["t"], float)
    assert list(frame["leader"]) == JOINT_ORDER and list(frame["follower"]) == JOINT_ORDER
    with pytest.raises(SessionStateError):
        svc.teleop.stop()


def test_teleop_respects_max_relative_target(tmp_path, configs, home):
    path = configs / "so101_follower.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("max_relative_target: null", "max_relative_target: 2"),
                    encoding="utf-8")
    svc = make_service(tmp_path, configs)
    frames = []
    svc.stream.subscribe(frames.append)
    svc.teleop.start()
    time.sleep(0.5)
    svc.teleop.stop()
    for previous, current in zip(frames, frames[1:]):
        for joint in JOINT_ORDER:
            assert abs(current["follower"][joint] - previous["follower"][joint]) <= 2.0 + 1e-6


def test_teleop_start_while_calibrating_is_refused(tmp_path, configs, home):
    svc = make_service(tmp_path, configs)
    svc.calibration.start("follower")
    wait_state(svc.calibration, "homing")
    with pytest.raises(HardwareBusy, match="calibration of the follower arm"):
        svc.teleop.start()
    assert not svc.teleop.running
    svc.calibration.cancel()


def test_real_backend_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MTREA_HARDWARE", raising=False)
    assert HardwareService(data_base=tmp_path).backend.name == "real"
    monkeypatch.setenv("MTREA_HARDWARE", "bogus")
    with pytest.raises(HardwareError, match="MTREA_HARDWARE"):
        HardwareService(data_base=tmp_path)


def test_server_and_real_backend_import_without_lerobot():
    code = ("import sys, lab.server.main, lab.core.hardware.cli; "
            "from lab.core.hardware.devices import RealBackend; RealBackend(); "
            "assert not [m for m in sys.modules if m.split('.')[0] == 'lerobot'], 'lerobot imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
