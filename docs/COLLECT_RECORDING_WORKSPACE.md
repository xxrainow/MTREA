# Recording workspace: study notes

## Scope

The Collect form opens a dedicated recording view after Start. Both fake and real
web recorders report preparation, recording, saving, and reset stages. The real
web adapter publishes frames already read by LeRobot. Standalone CLI collection
continues to use the ordinary `lerobot-record` command. Simulated illustrations
are never presented as live camera feeds. Real hardware verification is still
required; compatibility and bridge tests do not establish physical robot behavior.

## Follow the data

1. `lab/core/models.py`: `CollectionStatus` separates job lifecycle (`phase`) from
   the episode loop (`episode_phase`). It also carries the dataset, instruction,
   episode counts, camera names, timing, and backend mode.
2. `lab/core/fake_recorder.py`: `_stage()` changes the phase under a progress lock.
   `Event.wait(duration)` implements interruptible preparation/record/reset waits.
   Preparation takes three seconds. Episode and reset durations come from the
   submitted settings. No reset is inserted after the last episode.
3. `lab/core/collection.py`: `status()` snapshots progress while the worker runs.
   It converts per-attempt progress into per-session counts for resumed sessions.
   The worker preserves the final snapshot before releasing the recording object.
   Verification and the append-only ledger remain part of finalization.
4. `lab/web/src/hooks/useCollection.ts`: serial polling receives server snapshots
   and records their browser receipt time. Request versions discard stale reads.
5. `lab/web/src/components/RecordingView.tsx`: interpolates the last server timing
   snapshot with `performance.now()` for smooth display. Interpolation stops at the
   phase limit; the browser never decides that recording or saving has completed.
6. `lab/web/src/pages/CollectPage.tsx`: switches between settings and recording.
   Reopening Collect during an active job opens the recording view from server
   state. Back is available after processing finishes.

## Why these choices matter

- A running process is not proof that an episode has started or been saved.
- Monotonic clocks measure elapsed time without wall-clock adjustments.
- `Event.wait()` lets Stop interrupt a countdown rather than waiting for sleep.
- Progress uses a separate lock: waiting for recording completion must not block
  status reads or stop requests.
- Simulated camera panels are illustrations. A real preview must share frames
  from the recording owner instead of opening the same device again.

## Try it

Run the backend with `MTREA_HARDWARE=fake`. Use a fresh dataset name, an explicit
seed, two episodes, a three-second recording duration, and a two-second reset.
Expect preparation → episode 1 → reset → episode 2 → verification → completion.
Try Stop during preparation and during episode 2. Completed episodes remain saved.

Fake data stays under `data/fake/collection/`. It contains metadata, not training
frames or videos. Partial stops may still produce `failed` plus
`stop_requested=true` because the existing verifier expects the full requested
episode count; the recording view displays the saved count and verification error.

## Validation

```bash
python -m pytest tests/test_fake_recording_progress.py tests/test_collection_service.py tests/test_collection_api.py tests/test_collect.py tests/test_boundary.py
cd lab/web
npm run build
npm run lint
```

The new tests cover stage order and configured durations, interruptible preparation,
final progress retention, and preservation of completed episodes after Stop.
They do not establish real robot behavior or live camera streaming.

## Next steps

Real hardware end-to-end verification; End Episode/Start Next Episode controls;
input restoration after reload; distinguishing expected partial stops from genuine
verification failures. The research protocol and data-quality checks remain separate.

## From Collect to Datasets

Recordings already live on disk; the catalog does not copy them into another
database. `lab/core/datasets.py` reads each dataset's `meta/info.json` to build
the list. Selecting a dataset loads its task and episode metadata with the
existing v2 JSONL/v3 Parquet readers. The list is loaded when entering Datasets
and can be reloaded with Refresh. Detail requests are cancelled when changing
selection, so a slower previous request cannot replace the current selection.

`lab/server/datasets_api.py` exposes GET `/api/datasets` and
`/api/datasets/{backend}/{name}`. It delegates disk access to core, and the React
page only renders the returned metadata. Real and fake storage are both listed
with different IDs. Missing fields stay unknown; web recordings do not establish
a T_sub/held-out role. Invalid metadata produces a per-dataset warning rather
than hiding every healthy dataset.

Joint plots remain future work. Fake collection has neither real frames nor
videos to play. These endpoints only read data and reject dataset or metadata
paths that link outside their allowed storage.

Interview discussion: explain why disk remains the source of truth, why list
and detail reads are separate, and why displaying metadata does not establish
data quality. The integration test records through the Fake start API, waits
for completion, then verifies the catalog and per-episode instructions.

## Recorded video playback

`lab/core/dataset_video.py` resolves videos from `meta/info.json` and episode
metadata. LeRobot v2 stores a separate episode file; v3 can pack several episodes
into a single MP4. The v3 episode's `videos/<key>/from_timestamp` and
`to_timestamp` therefore define the playable segment. Guessing a filename from
the episode number would select the wrong video after episode zero.

The API exposes an episode's camera segments and serves each resolved MP4 with
Starlette FileResponse. Byte-range requests allow browsers to seek without
downloading the entire file again. The client supplies a dataset, episode, and
camera identity, never an arbitrary filesystem path. File resolution rejects
paths and symlinks outside the selected dataset. Source recordings are unchanged.

`DatasetEpisodeVideos.tsx` fetches camera segments. `DatasetVideoPlayer.tsx` uses
one HTML video element per camera, maps its absolute file time to an
episode-relative seek bar, and pauses at the selected episode's end. Changing
episodes unmounts old players and aborts their metadata request. Cameras have
independent controls, rather than claiming frame-accurate synchronization.
Browser codec support still matters: the saved real sample uses AV1; a browser
that cannot decode it receives an explicit playback error.

Tests cover v2 paths, v3 shared files with different timestamp bounds, HTTP 206
byte ranges, HEAD responses, missing cameras/files, and out-of-dataset symlinks.
Run `python -m pytest tests/test_dataset_video.py tests/test_datasets_api.py
tests/test_boundary.py -q`. Interview topics: metadata-driven media lookup,
range requests, segment timelines, and the difference between two playable
camera views and synchronized playback.

## Live recording bridge: real mode

`RecordingProcess(..., telemetry=True)` launches `python -m
lab.core.recording_worker` with the backend's Python interpreter. Activate the
LeLab environment before starting the backend; installing LeRobot in a different
Python environment does not make it available in this child. The original CLI
arguments and adapter identity are retained in the collection ledger. The
adapter adds a fixed 3-second preparation loop before the first recorded episode.

The child installs process-local wrappers around LeRobot's robot factory,
`record_loop`, and `save_episode`. It does not edit the installed package. LeRobot
still connects/disconnects hardware, applies processors, writes episodes, and
finalizes the dataset. The loop wrapper emits timestamps when preparation,
recording, reset, or saving actually starts. Saved counts increase only after
`save_episode` returns successfully. LeRobot resets before saving in the inspected
version, so a completed episode can remain unsaved during the orange reset stage.
The adapter supports the inspected 0.6.0/0.6.1 interfaces and checks signatures;
upgrading LeRobot requires checking these hooks again. 0.6.0's image-writer decorator
does not preserve the signature, so the adapter inspects its captured function
while retaining the original wrapper's cleanup behavior.
The installed 0.6.0 configuration also lacks `dataset.no_stamp`; the web adapter
removes that unsupported CLI flag and disables its process-local repo-ID stamping
method. The full generated configuration was parser-checked without opening hardware.

The observation wrapper receives the same RGB arrays the recorder reads. A
latest-frame mailbox copies a sampled observation; a separate thread encodes
JPEGs. It never opens a camera or queues every recording frame. Progress JSON and
JPEGs are atomically replaced in one temporary directory per attempt. The parent
reads timing using a monotonic clock. Frame HTTP requests are tied to an attempt
ID and reject frames older than two seconds. Client requests are serialized with
timeouts, preventing slow viewers from piling up work. This is a low-rate live
preview, not a frame-accurate recording stream.

The web path disables desktop keyboard listeners and uses SIGINT/CTRL_BREAK to
request a graceful stop through LeRobot's events. Stop during preparation exits
without recording an empty episode. A stop during recording may save a partial
episode according to LeRobot behavior; always inspect stopped recordings. Child
stdin is disabled so missing calibration cannot leave an invisible prompt waiting
forever. Complete calibration in Devices first. Finalization must exit before
hardware ownership is released and temporary IPC files are removed.

Study files: `recording_worker.py` for the LeRobot adapter,
`recording_telemetry.py` for bounded IPC and JPEG publication, `collection.py`
for attempt lifetime, and `RecordingCamera.tsx` for browser refresh and cleanup.
Tests cover hooks, decorator compatibility, preparation cancellation, frame
freshness, API attempt isolation, and real subprocess IPC cleanup using a harmless
child. These tests do not connect an arm or camera.
