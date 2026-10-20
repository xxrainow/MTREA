from dataclasses import replace
from pathlib import Path
from threading import Lock, Thread
from copy import deepcopy
from time import monotonic
from robot.config import CAMERAS

from lab.core import paths
from lab.core.hardware import SessionStateError
from lab.core.hardware.lock import HardwareLock
from lab.core.models import CollectionPhase, CollectionStatus
from lab.core.recorder import (
    RecordingProcess,
    RecordingResult,
    build_hardware_args,
    build_record_command,
    wait_and_verify,
)
from lab.core.collection_ledger import (
    append_record,
    prepare_session,
)
from lab.core.fake_recorder import (
    FakeRecordingProcess,
    fake_collection_enabled,
)


class CollectionService:
    def __init__(self, data_base: Path | None = None) -> None:
        self._lock = Lock()
        self._hardware_lock = HardwareLock(
            paths.hardware_lock(data_base)
        )
        self._status = CollectionStatus()
        self._recording: (
            RecordingProcess | FakeRecordingProcess | None
        ) = None
        self._worker: Thread | None = None
        self._result: RecordingResult | None = None
        self._fake_mode = fake_collection_enabled()
        self._recorded_before = 0
        self._started_clock: float | None = None
        self._status = replace(
            self._status, backend="fake" if self._fake_mode else "real"
        )

    def status(self) -> CollectionStatus:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> CollectionStatus:
        status = self._status
        if self._started_clock is not None:
            status = replace(
                status, session_elapsed_s=monotonic() - self._started_clock
            )
        if self._recording is not None and hasattr(self._recording, "progress"):
            progress = self._recording.progress()
            if "current_episode" in progress:
                progress["current_episode"] = min(
                    status.total_episodes,
                    self._recorded_before + progress["current_episode"],
                )
            if "saved_episodes" in progress:
                progress["saved_episodes"] += self._recorded_before
            status = replace(status, **progress)
        return status

    def read_frame(self, recording_id: str, name: str) -> bytes:
        with self._lock:
            recording = self._recording
            if (
                self._fake_mode or recording is None
                or getattr(recording, "recording_id", None) != recording_id
                or self._status.phase not in (CollectionPhase.STARTING, CollectionPhase.RECORDING)
            ):
                raise FileNotFoundError("Recording preview is inactive")
        return recording.frame(name)

    def result(self) -> RecordingResult | None:
        with self._lock:
            return self._result
    def start(
        self,
        *,
        dataset_root: Path,
        ledger_path: Path,
        plan_sha: str,
        session_index: int,
        planned_episodes: int,
        repo_id: str,
        instruction: str,
        episode_time_s: float,
        reset_time_s: float,
        record_context: dict,
        executable: str = "lerobot-record",       
    ) -> CollectionStatus:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise SessionStateError("Collection is already active")

            self._hardware_lock.acquire("collection")

            try:
                context = deepcopy(record_context)

                required = {
                    "object",
                    "plan",
                    "git_commit",
                    "git_dirty",
                    "lerobot_version",
                    "layouts",
                }
                if not required.issubset(context):
                    raise ValueError("Recording metadata is incomplete")

                layouts = context.pop("layouts")
                if (
                    not isinstance(layouts, list)
                    or len(layouts) != planned_episodes
                    or not all(isinstance(item, dict) for item in layouts)
                ):
                    raise ValueError(
                        "Layouts must cover every planned episode"
                    )
                state = prepare_session(
                    dataset_root=dataset_root,
                    ledger_path=ledger_path,
                    plan_sha=plan_sha,
                    session_index=session_index,
                    planned_episodes=planned_episodes,
                )

                command = build_record_command(
                    executable=executable,
                    hardware_args=(
                        [] if self._fake_mode else build_hardware_args()
                    ),
                    repo_id=repo_id,
                    dataset_root=dataset_root,
                    instruction=instruction,
                    num_episodes=state.remaining_episodes,
                    episode_time_s=episode_time_s,
                    reset_time_s=reset_time_s,
                    resume=state.resume,
                )
                
                ledger_base = {
                                    **context,
                                    "session": session_index,
                                    "instruction": instruction,
                                    "planned_episodes": planned_episodes,
                                    "requested_episodes": state.remaining_episodes,
                                    "episodes_before": state.episodes_before,
                                    "resume": state.resume,
                                    "plan_sha256": plan_sha,
                                    "argv": command,
                                    "layouts": layouts[state.recorded_episodes:],
                                    "recorder_adapter": "fake" if self._fake_mode else "web-hooks-v1",
                                    }

                self._result = None
                self._recorded_before = state.recorded_episodes
                self._started_clock = monotonic()
                self._status = CollectionStatus(
                    phase=CollectionPhase.STARTING,
                    session_index=session_index,
                    backend="fake" if self._fake_mode else "real",
                    dataset=repo_id,
                    instruction=instruction,
                    total_episodes=planned_episodes,
                    saved_episodes=state.recorded_episodes,
                    camera_names=tuple(camera.name for camera in CAMERAS),
                )

                self._worker = Thread(
                    target=self._run,
                    kwargs={
                        "command": command,
                        "dataset_root": dataset_root,
                        "instruction": instruction,
                        "num_episodes": state.remaining_episodes,
                        "episodes_before": state.episodes_before,
                        "ledger_path": ledger_path,
                        "ledger_base": ledger_base,
                    },
                    daemon=False,
                )
                self._worker.start()

            except Exception as exc:
                self._hardware_lock.release()
                self._started_clock = None
                self._status = replace(
                    self._status,
                    phase=CollectionPhase.FAILED,
                    session_index=session_index,
                    error=str(exc),
                )
                raise

            return self._status
    

    def stop(self) -> CollectionStatus:
        with self._lock:
            if self._status.phase not in (
                CollectionPhase.STARTING,
                CollectionPhase.RECORDING,
            ):
                return self._status

            if self._recording is not None:
                self._recording.request_stop()

            self._status = replace(
                self._status,
                phase=CollectionPhase.STOPPING,
                stop_requested=True,
            )
            return self._status

    def _run(
        self,
        *,
        command: list[str],
        dataset_root: Path,
        instruction: str,
        num_episodes: int,
        episodes_before: int,
        ledger_path: Path,
        ledger_base: dict,
    ) -> None:
        error = None

        try:
            with self._lock:
                if self._status.stop_requested:
                    return

                if self._fake_mode:
                    self._recording = FakeRecordingProcess(command)
                else:
                    self._recording = RecordingProcess(command, telemetry=True)
                recording = self._recording
                self._status = replace(
                    self._status,
                    phase=CollectionPhase.RECORDING,
                    recording_id=getattr(recording, "recording_id", None),
                )

            recording.wait()

            with self._lock:
                self._status = self._snapshot_locked()
                self._status = replace(
                    self._status,
                    phase=CollectionPhase.VERIFYING,
                )

            result = wait_and_verify(
                recording,
                dataset_root=dataset_root,
                episodes_before=episodes_before,
                requested_episodes=num_episodes,
                instruction=instruction,
            )

            with self._lock:
                self._result = result
                if result.episodes_after is not None:
                    self._status = replace(
                        self._status,
                        saved_episodes=self._recorded_before + max(
                            0, result.episodes_after - episodes_before
                        ),
                    )
                interrupted = self._status.stop_requested

            added = (
                0
                if result.episodes_after is None
                else max(0, result.episodes_after - episodes_before)
            )
            saved_layouts = [
                {
                    **layout,
                    "episode_index": episodes_before + offset,
                }
                for offset, layout in enumerate(
                    ledger_base["layouts"][:min(added, num_episodes)]
                )
            ]

            append_record(
                ledger_path,
                {
                    **ledger_base,
                    "layouts": saved_layouts,
                    "episodes_after": result.episodes_after,
                    "started_at": result.started_at,
                    "ended_at": result.ended_at,
                    "lerobot_exit_code": result.exit_code,
                    "status": (
                        "mismatch" if result.problems else "ok"
                    ),
                    "ok": not result.problems,
                    "interrupted": interrupted,
                    "problems": list(result.problems),
                },
            )

            if result.problems:
                error = "; ".join((*result.problems, *([self._status.error] if self._status.error else [])))

        except Exception as exc:
            error = str(exc)

        finally:
            # Keep ownership until the recording process has exited.
            recording = self._recording
            if recording is not None:
                recording.wait()

            try:
                self._hardware_lock.release()
            finally:
                with self._lock:
                    self._status = self._snapshot_locked()
                    self._started_clock = None
                    self._recording = None
                    self._status = replace(
                        self._status,
                        phase=(
                            CollectionPhase.FAILED
                            if error
                            else CollectionPhase.FINISHED
                        ),
                        error=error,
                    )
                if recording is not None and hasattr(recording, "close"):
                    recording.close()
                    
    def is_active(self) -> bool:
        with self._lock:
            return (
                self._worker is not None
                and self._worker.is_alive()
            )

    def shutdown(self) -> None:
        self.stop()

        with self._lock:
            worker = self._worker

        if worker is not None:
            worker.join()
