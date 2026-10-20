from dataclasses import replace
from pathlib import Path
from threading import Lock, Thread
from copy import deepcopy

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

    def status(self) -> CollectionStatus:
        with self._lock:
            return self._status

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
                                    }

                self._result = None
                self._status = CollectionStatus(
                    phase=CollectionPhase.STARTING,
                    session_index=session_index,
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
                self._status = CollectionStatus(
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
                    self._recording = RecordingProcess(command)                
                recording = self._recording
                self._status = replace(
                    self._status,
                    phase=CollectionPhase.RECORDING,
                )

            recording.wait()

            with self._lock:
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
                error = "; ".join(result.problems)

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
