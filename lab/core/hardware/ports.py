"""Serial ports: list them, find an arm's port by unplugging it, save it to the yaml.

Detection works like lerobot-find-port (v0.6.1) and LeLab: snapshot the port
list, the operator unplugs one arm's USB cable, and the one port that
disappeared is that arm's. The list is built the way lerobot-find-port builds
it (pyserial COM ports on Windows, /dev/tty* elsewhere), so a detected name is
exactly what LeRobot opens and what record_test.sh has always used.
"""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

import yaml

from lab.core.hardware import HardwareError, PortDetectionError, SessionStateError, check_arm
from lab.core.hardware.devices import CONFIGS_DIR, config_path

DETECT_TIMEOUT_S = 15.0


def list_ports() -> list[str]:
    if platform.system() == "Windows":
        try:
            from serial.tools import list_ports as serial_ports
        except ImportError:
            raise HardwareError("pyserial is needed to list COM ports: pip install pyserial") from None
        return sorted(p.device for p in serial_ports.comports())
    return sorted(str(p) for p in Path("/dev").glob("tty*"))


class PortDetection:
    def __init__(self, lister: Callable[[], list[str]] = list_ports,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._lister, self._sleep, self._clock = lister, sleep, clock
        self._before: list[str] | None = None

    @property
    def started(self) -> bool:
        return self._before is not None

    def start(self) -> list[str]:
        self._before = list(self._lister())
        return list(self._before)

    def finish(self, timeout_s: float = DETECT_TIMEOUT_S, poll_s: float = 0.3) -> str:
        """Wait until exactly one port from the snapshot is gone, and return it."""
        if self._before is None:
            raise SessionStateError("port detection was not started")
        before, self._before = set(self._before), None
        deadline = self._clock() + timeout_s
        while True:
            gone = sorted(before - set(self._lister()))
            if len(gone) == 1:
                return gone[0]
            if len(gone) > 1:
                raise PortDetectionError(f"more than one port disappeared ({', '.join(gone)}); "
                                         "unplug only the one arm and try again")
            if self._clock() >= deadline:
                raise PortDetectionError(f"no port disappeared within {timeout_s:.0f} s; unplug the arm's "
                                         "USB cable when asked and try again")
            self._sleep(poll_s)


# A top-level `port:` line (LF or CRLF); group "value" stops before an inline comment.
_PORT_LINE = re.compile(
    r"^port:(?P<space>[ \t]*)(?P<value>[^#\r\n]*?)(?P<rest>[ \t]*(?:#[^\r\n]*)?)(?=\r?\n|\Z)", re.M)


def _yaml_scalar(port: str) -> str:
    try:
        if yaml.safe_load(f"port: {port}") == {"port": port} and "#" not in port:
            return port
    except yaml.YAMLError:
        pass
    return json.dumps(port)  # a double-quoted JSON string is valid YAML


def save_port(arm: str, port: str, configs_dir: Path = CONFIGS_DIR) -> Path:
    """Rewrite only the `port:` line of so101_<arm>.yaml; comments and other keys stay as they are."""
    check_arm(arm)
    if not port or port != port.strip() or "\n" in port or "\r" in port:
        raise HardwareError(f"invalid port {port!r}")
    path = config_path(arm, configs_dir)
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    matches = list(_PORT_LINE.finditer(text))
    if len(matches) != 1:
        raise HardwareError(f"{path}: expected exactly one top-level 'port:' line, found {len(matches)}")
    m = matches[0]
    value = _yaml_scalar(port) if m.group("space") else " " + _yaml_scalar(port)
    new_text = text[:m.start("value")] + value + text[m.end("value"):]

    before, after = yaml.safe_load(text), yaml.safe_load(new_text)
    if after.get("port") != port or {k: v for k, v in after.items() if k != "port"} != {
            k: v for k, v in before.items() if k != "port"}:
        raise HardwareError(f"{path}: rewriting the port line would change more than the port; edit it by hand")

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
