# AGENTS.md

MT-REA — studying whether narrow fine-tuning of a pretrained VLA on a new
embodiment preserves its capability on tasks it was not fine-tuned on.

## Vocabulary

Terms used throughout the code; they are not self-explanatory from names alone.

- **T** — the broad task distribution the base VLA was pretrained on (source embodiment).
- **T_sub** — the subset of tasks we actually fine-tune on with SO-101 demos.
- **held-out** — `T \ T_sub`. Tasks the base model could do but we never fine-tune on.
  These are the measurement target, not a test split of the training data.
- **H0, H1, H2, ...** — fine-tuning methods under comparison. H0 is naive
  single-task fine-tuning and serves as the baseline all others are compared against.
- **rollout** — one execution attempt of one task by a policy on the real robot.
- **episode** — one recorded human teleoperation demo.
- **oracle** — a reference model fine-tuned on a held-out task's own demos. It
  gives the ceiling for that task on SO-101. It is not a baseline and not a
  method under comparison.

## Invariants

- Baseline is H0, never zero-shot. Zero-shot on SO-101 floors near 0% from
  embodiment mismatch alone, so it carries no task-level signal.
- Metrics are absolute success rates, never ratios. Any ratio divides by a
  near-zero number here (H0's held-out success is expected to be low).
- An oracle's demos never enter the training data of H0 or any compared method.
  A held-out task stays held-out for every compared method.

## Tech stack

- Python 3.12+ (lerobot v0.6.1 requires it) · PyTorch
- LeRobot pinned to v0.6.1 (not PyPI latest) for hardware I/O, dataset format,
  training. GPU host: `lerobot[pi,async,training]`; robot side: `lerobot[feetech]`.
- Training is LeRobot-native: `lerobot-train --policy.type=pi05` starting from
  `lerobot/pi05_base`. H0 is full fine-tuning (`train_expert_only=false`,
  `freeze_vision_encoder=false`) at batch 16, the most one A100 MIG 3g.40gb
  slice holds (docs/COMPUTE.md). LoRA is not used (unsupported for π0.5).
- TODO: pin the package manager (uv vs pip) once the env is built

## Commands

```bash
pip install -e .                      # editable install
pytest research/ -v                   # research-side tests (no GPU/robot needed)
pytest tests/test_boundary.py         # enforces lab -> research direction
scripts/train.sh --method=h0 --dataset=<repo_id> --output=data/runs/h0_pickplace_seed0 --seed=0 [--dry-run]
```

Wrapper scripts live in top-level `scripts/*.sh` (not `research/scripts/`).
They take `--flag=value` arguments, print one parseable progress line at a time
to stdout, and exit 0 on success. GPU/robot commands among them need the A100
SSH host or the arm plugged in. They will not run in a plain checkout — check
before invoking.

## Structure

```
lab/core/    robot control, recording, job runner, state — knows nothing about HTTP
lab/server/  FastAPI (:8001) + hardware API/WebSocket — delegates to core
lab/web/     React + Vite (:8080) — buttons and status only, no Python
research/    methods, task definitions, eval, analysis
scripts/     thin shell wrappers over `python -m research.*`
robot/configs/ so101_follower/leader.yaml, cameras.yaml, remote.yaml (gitignored)
experiments/  one-off hardware/data checks, not part of the pipeline
data/        raw, datasets, norm_stats, runs, rollouts, jobs, logs — gitignored, never committed
```

`lab/` is three layers that do not mix: `core/` never imports FastAPI, `server/`
API handlers never contain robot code, and `core/runner.py` never imports
`research/` — it only runs the shell wrappers as subprocesses (local or over SSH).
`lab/` and `research/` share only the `data/` layout and `robot/configs/`.

`research/` must never import from `lab/`. The dependency runs one way so
`research/` can be lifted into a standalone paper repo. `tests/test_boundary.py`
enforces this; a violation is a failing test, not a style opinion.

## Experiment conventions

- `experiments/inspect_dataset.py` performs read-only recording integrity checks
  across all parquet files and within each episode. `--check-videos` additionally
  decodes all MP4s with ffmpeg. Passing does not establish training compatibility,
  camera identity, synchronization, or demonstration quality.
- T_sub demos are recorded with `experiments/collect.py` from a fixed plan in
  `research/collect/plans/*.yaml` (one lerobot-record session per object,
  appended with `--resume`). Its ledger `data/raw/<dataset>.collection.jsonl`
  is append-only, like rollout results.
- Every training run gets a directory under `data/runs/<run_id>/` holding its
  resolved config, `metrics.jsonl`, and checkpoints. Nothing about a run is
  reconstructed from memory or from the command line after the fact.
- Record the git commit with each run. A result that cannot be traced to a
  commit is not usable in the paper.
- Seeds are explicit arguments, never defaulted silently.
- Rollout results are append-only. Never rewrite or delete a past result file
  to "clean up" — the comparison across methods depends on them.
- Task definitions in `research/collect/tasks/*.yaml` carry the T_sub vs held-out
  role. Changing a task's role invalidates every prior result that used it, so
  it is a new task definition, not an edit.

## Conventions

Only deviations from ordinary Python practice are listed; the rest is standard.

- Comments and docstrings in English, and sparse. Explain why, not what.
- Robot constants (FPS, joint order, camera names) live only in `robot/config.py`.
  Nothing else defines them; everything else imports them.
- `POLICY_SERVER_HOST/PORT` in `robot/config.py` (127.0.0.1:8765) is the robot-side
  end of an SSH tunnel, never a remote address. `serve_policy` on the GPU host binds
  its own localhost only; the GPU-side port lives in `robot/configs/remote.yaml`.
- Hardware is assumed to be SO-101 throughout. Supporting another arm means
  changing `robot/config.py`, `lab/core/robot.py`, and `robot/configs/` — not a
  config flag today.
- Analysis code must run on synthetic inputs, so correctness is testable with
  no GPU and no robot.

## Boundaries

- One process at a time holds the SO-101 arms. `lab/core/hardware/lock.py`
  (`data/locks/hardware.lock`) is taken by calibration, teleoperation and
  `experiments/collect.py`; any new code that opens the serial ports takes it
  too.
- Devices and Teleoperate use the hardware API; page changes do not stop an
  active calibration or teleoperation. Return to the page to cancel/stop.
  Vite proxies `/api` and `/ws` (WebSocket) to the backend. Joint order comes
  from `robot/config.py` through `/api/hardware/config`, never a web constant.
- Never commit anything under `data/`, `.env`, or calibration files.
- Never bind the policy server to 0.0.0.0 — it must be 127.0.0.1 behind an SSH
  tunnel.
- Devices and Collect share the backend camera preview service. Page changes do not stop preview. A camera frame-read failure stops all preview cameras and exposes an error in status; reconnecting requires an explicit start.
- Collect opens a dedicated recording workspace. Both modes have a 3-second preparation phase. Fake illustrations are explicitly simulated. Real web collection runs `lab.core.recording_worker` in the server's Python environment, wrapping installed LeRobot 0.6.0/0.6.1 recording hooks to report phase timing and publish existing observation frames. It retains LeRobot dataset writing/finalization. Temporary latest-frame IPC is bounded, JPEG encoding runs outside the control loop, stale frames are rejected, and files are removed after exit. Recording owns cameras exclusively; standalone preview is stopped before collection starts. CLI collection keeps its ordinary `lerobot-record` path. Real hardware end-to-end validation remains required.
- Collect retains timed episode/reset settings, with no separate manual mode. Real recording controls can end preparation, recording, or reset early through the command API and temporary command-file IPC; Fake simulates the same phase advances with an in-memory event and writes metadata only. During recording, Discard & retry confirms discarding the current unsaved take, runs reset, and repeats the same episode without changing previously saved data. Real recording uses LeRobot's rerecord flag and buffer cleanup; Fake omits the discarded take from metadata. Every phase has a new phase_id, required by commands, so stale requests and button locks do not carry into a repeated attempt even with zero reset duration. A queued response is not execution confirmation; the web waits for the reported phase ID to change before allowing another command. Episode saving retains LeRobot's reset-before-save order in both backends. Whole-session stop remains separate.
- Datasets reads local recording metadata from `data/raw/` and `data/fake/collection/raw/` through the read-only `/api/datasets` endpoints. Real and Fake datasets have separate identities, even with matching names. Metadata availability is not validation or research-role approval; Fake datasets contain no recorded video. Dataset browsing does not open hardware.
- After collection ends with saved episodes, View dataset opens that dataset's detail screen directly using its backend and name. Partial saved results can also be inspected after a stopped/failed run. Navigation remains disabled while collection is active; ordinary sidebar navigation to Datasets opens the list.
- Stop recording requires confirmation. After an intentional stop, verification permits fewer episodes than requested but still checks exit status and metadata. Valid partial runs use an interrupted ledger status so they remain resumable. The web shows the verified saved count and remaining target in plain language; it never promises that an in-progress take was saved before verification. CLI verification remains strict.
- Web collection has no seed input or seed metadata in new snapshots/ledger entries: recording does not use randomness. Historical snapshots containing seed remain unchanged and retain their hashes for resuming partial collections. Research plan generation, training and evaluation seed requirements remain separate.
- Dataset playback resolves v2 episode MP4s or v3 shared MP4s and timestamp bounds from saved metadata. Video responses support byte ranges; only paths inside the selected dataset are served. Each camera plays independently, with an episode-relative timeline. Original recordings are never rewritten for playback.


## Ownership

Dev A owns `robot/`, `research/`, `docs/`. Dev B owns `lab/core/`, `lab/server/`,
`lab/web/`, `scripts/`. Changes to the other person's area go through a PR they
review; nobody pushes to main directly.

## Keeping this file current

This file is the project's memory across sessions. When a decision is made or
reversed update this file in the same change, or the next session will repeat the mistake.
