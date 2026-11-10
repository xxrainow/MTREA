# Serving: the contract between the GPU host and the robot side

`scripts/serve_policy.sh` serves one trained checkpoint to LeRobot's
`robot_client`. The server subclasses LeRobot's `PolicyServer`
(`research/serve_backend.py`), so the async inference protocol is unchanged and
the robot side runs `robot_client` as it is. Verified against lerobot v0.6.1.

## 1. Start it

On the GPU host, from the repo root, with the venv on PATH:

```bash
scripts/serve_policy.sh \
  --checkpoint=/data/mtrea/data/runs/<run>/lerobot/checkpoints/last/pretrained_model
```

Wait for `STATUS ready port=8000`. Loading took about 3.5 minutes when measured
(section 7).

| flag | default | |
|---|---|---|
| `--checkpoint` | required | `pretrained_model` directory or Hub id. Clients must send this same string. |
| `--host` | `127.0.0.1` | loopback only; anything else exits 2 |
| `--port` | `8000` | `0` picks a free port, reported in `STATUS ready` |
| `--fps` | `30` | spacing of the action timestamps in a chunk |
| `--device` | `cuda` | the client's `policy_device` is ignored |
| `--session` | `serve_<time>` | must be new; past sessions are never overwritten |
| `--data-dir` | see section 5 | where the session goes |
| `--fake` | `none` | `hold` or `replay`, section 4 |
| `--dataset-root` | none | replay's actions; features for a fake server without a checkpoint config |

## 2. Fixed values

| | |
|---|---|
| server, on the GPU host | `127.0.0.1:8000` |
| tunnel, from the robot PC | `ssh -L 8765:127.0.0.1:8000 root@mtrea-ts` |
| client `server_address` | `127.0.0.1:8765` (`POLICY_SERVER_HOST/PORT` in `robot/config.py`) |
| `policy_type` | `pi05` |
| `pretrained_name_or_path` | exactly the server's `--checkpoint` string |
| `actions_per_chunk` | `50` |
| `fps` | `30` |
| `chunk_size_threshold` | `0.5` is what section 7 measured |
| cameras | `scene` and `wrist`, 640x480 |
| joint units | body joints in degrees, gripper 0-100 (see below) |

**Units.** With `use_degrees: true`, LeRobot's `so_follower.py` reads the five body
joints in degrees and the gripper as `RANGE_0_100`, a percentage of its calibrated
range. The recorded gripper spans 0 to 41 in `pick_and_place`. Every error reported
here is in those units, so a gripper error is not an angle.

**Checkpoint string.** Only the string is compared, or the resolved path on the GPU
host, so `checkpoints/last/...` and `checkpoints/010000/...` count as the same
checkpoint. Nothing is ever loaded from what the client sends.

**Serving and training never share the GPU slice.** A server refuses to start with
more than 1 GiB in use on the device. Nothing stops a training job started while a
server runs, so stop the server first.

## 3. What gets refused

Before anything loads: exit 2, `error: ...` on stderr, `STATUS failed exit=2`, and
no session directory.

| condition | message contains |
|---|---|
| host is not loopback | `is not a loopback address` |
| no `config.json` | `has no config.json` |
| policy type not servable | `is not servable` |
| no weights | `has no weights` |
| no processor file or its stats | `has no processor file`, `has no <file>` |
| checkpoint from a fake run | `comes from a fake run` |
| checkpoint past lerobot's `last` | `is not complete` |
| another process on the GPU | `GPU busy` |
| port taken | `cannot bind` |
| session exists | `never overwritten` |

During load: exit 1 and status `failed`. This includes weights that fail to load.
`PI05Policy.from_pretrained` catches that error, reports it with a print and returns
a randomly initialized model, so `load_policy` refuses unless lerobot prints that
all keys loaded.

From a client, `SendPolicyInstructions` aborts with `FAILED_PRECONDITION`, naming
every mismatch at once. `robot_client`'s `start()` logs the error and returns False.

```
refused: pretrained_name_or_path 'lerobot/pi05_base' is not the loaded checkpoint '/data/...'; this server does not load what a client asks for
refused: camera observation.images.scene is missing from the client; camera observation.images.front is not in the checkpoint (it has [...])
refused: camera observation.images.scene has shape [240, 320, 3] (H, W, C), the checkpoint expects [480, 640, 3]
refused: observation.state has shape [5], the checkpoint expects [6]; observation.state joints are [...], the checkpoint expects [...]
refused: actions_per_chunk 51 must be between 1 and the policy's chunk_size 50
refused: policy_type 'act' is not the loaded 'pi05'
refused: rename_map must be empty, got {...}; send the checkpoint's camera names
```

Observations and `GetActions` before an accepted `SendPolicyInstructions` are also
refused. Each `Ready` resets acceptance.

**Physically swapped cameras cannot be detected.** The protocol carries names and
shapes only, and both cameras are 640x480. If the `wrist` and `scene` devices are
exchanged (`index_or_path` in `robot/config.py`), the server accepts the client and
the policy runs on the wrong views. Check the two views by eye before a rollout.

## 4. Fake server

For developing the robot side with no GPU and no weights. It uses the same gRPC
protocol, the same refusals and the same files, with `"fake": true`. Measured on
Windows CPU; not tested on macOS.

```bash
pip install 'lerobot[async]==0.6.1'            # hold; add [dataset] for replay and the check client

# From the repo root. The string can name the checkpoint on the GPU host, so the client config does not change.
# A path missing on this machine takes its features from the dataset's meta/info.json.
scripts/serve_policy.sh --checkpoint=/data/mtrea/data/runs/<run>/lerobot/checkpoints/last/pretrained_model \
  --fake=hold --dataset-root=data/raw/<dataset>
```

On Windows, run `python -m research.serve_policy` with the same flags.

- **hold** returns chunks repeating the observation's own joint positions, so the
  arm stays where it is.
- **replay** returns the recorded action for each timestep of one episode, then
  holds its last action instead of jumping to the next episode. Each `Ready` moves
  to the next episode, so **restarting `robot_client` changes the trajectory**. It
  needs `--dataset-root`, the dataset's fps must equal `--fps`, and on real hardware
  it moves the arm.

## 5. Status lines and files

stdout carries one line per event, and nothing else. Everything else, lerobot's
logging included, goes to stderr.

```
STATUS loading checkpoint=<...>
STATUS ready port=<n>
CLIENT connected
CLIENT disconnected          no RPC from an accepted client for 5 s
STATUS stopped|failed exit=<n>
```

| exit | meaning | `status.json` |
|---|---|---|
| 1 | failed while loading | `failed`, with `failure` `oom` or `error` |
| 2 | refused before loading | not written |
| 143 / 130 | SIGTERM / SIGINT | `stopped` |

A prediction error does not end the server, because LeRobot's `GetActions` swallows
it and the client just receives no chunk. It shows as
`counts.prediction_errors` and `last_error` in `status.json`.

SIGTERM took 0.8 s on the loaded real server. On Windows `terminate()` and
`taskkill /F` are hard kills: `status.json` stays as it was and `end_time` stays
null, so treat a dead job with a live status as ended with unknown status.

Each session is `<data_dir>/rollouts/serve/<session>/`. `data_dir` is `--data-dir`,
else `ROBOLAB_DATA_ROOT`, else the directory holding the checkpoint's `runs/`, else
the repo's `data/`.

| file | written | content |
|---|---|---|
| `session.json` | at start, ready, end | checkpoint, resolved path, step, run id, method, seed, dataset, train commit, git commit and dirty, lerobot version, host, port, device, fps, GPU memory, start and end time, exit code |
| `status.json` | on every event, atomically | `status` (`loading`, `ready`, `serving`, `stopped`, `failed`), checkpoint, `client_connected`, `counts`, `last_latency_ms`, `last_refusal`, `last_error` |
| `inference.jsonl` | one line per chunk | `timestep`, `timestamp`, `receive_time`, `must_go`, `task`, `prepare_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`, `chunk_length`, `chunk`, `fake` |
| `logs/policy_server_<time>.log` | by lerobot | its full debug log |

The server does not seed torch, so flow-matching noise differs between sessions.
`inference.jsonl` records every chunk that was returned.

## 6. Checks

```bash
# Does the checkpoint reproduce its demos? Needs the GPU to itself.
python -m research.infer_offline --checkpoint=<dir> --dataset-root=<dir> --samples=50 --seed=0 \
  --out=<report.json> --compare=lerobot/pi05_base

# Drive a server without a robot. From the robot PC through the tunnel this measures the link.
python -m research.serve_client_check --server=127.0.0.1:8765 --checkpoint=<same string> \
  --dataset-root=<dir> --episodes=2 --out=<report.json>
```

`serve_client_check` runs LeRobot's own `RobotClient` with a dataset in place of the
robot. It needs `lerobot[async,dataset]` and decodes each episode into memory first
(about 1 GB per 600 frames). Each episode's own instruction is used unless `--task`
is given. Please run it through the tunnel before the first rollout. The numbers
below are loopback only and say nothing about Tailscale.

`--compare` puts the other weights behind the checkpoint's config and processors.
`lerobot/pi05_base` has no SO-101 features or stats of its own, so this is the only
meaningful comparison, and it is the state fine-tuning started from.

## 7. Measurements

H0 checkpoint `h0_pick_and_place_s1_20261103-1457`, step 10000, on one A100 MIG
3g.40gb slice with nothing else on it. Code at commit `1654d42`, clean tree, lerobot
0.6.1, dataset `pick_and_place`. Server session `verify_h0_20261104-134926`.

**Offline**: `infer_offline --samples=50 --seed=0 --compare=lerobot/pi05_base`, one
frame in each of 50 episodes, same noise for both models. Results vary slightly
across seeds. Mean over joints:

| mean MAE | H0 | `pi05_base` |
|---|---|---|
| first action | 0.918 | 23.167 |
| first 10 | 1.024 | 24.604 |
| all 50 | 1.569 | 28.720 |

Per joint, H0 over all 50 actions:

| | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| MAE | 1.137 | 2.863 | 2.907 | 1.209 | 0.680 | 0.618 |
| RMSE | 1.982 | 6.224 | 5.324 | 2.017 | 1.117 | 1.382 |

A low error here means the checkpoint reproduces its demos. It is not a rollout
success rate.

**Latency, bandwidth, memory**: through 127.0.0.1 on the GPU host.

| quantity | value | how measured |
|---|---|---|
| inference per chunk, offline | p50 335 ms, p95 419 ms, max 421 ms | `infer_offline`, 50 samples, warmup excluded |
| inference per chunk, server | p50 345 ms, p95 358 ms, max 735 ms | `inference.jsonl`, 389 chunks, `inference_ms` |
| server time per chunk | p50 359 ms, p95 375 ms, max 767 ms | `inference.jsonl`, `total_ms` |
| round trip, observation sent to chunk received | p50 376 ms, p95 398 ms, max 737 ms | `serve_client_check --episodes=12`, 329 chunks |
| bytes per observation | 1,843,771 | `len(pickle.dumps(obs))`, 2 x 640x480x3 raw |
| bytes per chunk of 50 | 76,005 | `len(pickle.dumps(chunk))` |
| bandwidth while sending every tick | 442.5 Mbit/s | bytes x 30 fps, threshold 0.5 |
| bandwidth averaged over a run | 170.9 Mbit/s | 11.59 observations/s measured, same run |
| GPU memory, loaded | 8.72 GiB allocated, 9.12 GiB reserved, 9.63 GiB on the device while serving | `session.json`; `torch.cuda.mem_get_info` |
| load and warmup | about 3.5 minutes | wall clock, one earlier pre-commit run |

Over 12 episodes the queue ran empty 10 times. Those stalls had a p50 of 367 ms and
a max of 628 ms. One of them was over the guardrail below.

**Fake servers**: `serve_client_check --episodes=2` on the GPU host, sessions
`verify_fake_hold_20261104-135839` and `verify_fake_replay_20261104-135934`. Both
had a round trip p50 of 13 ms and no stalls, so a fake server does not show the
timing a real one will. Replay's chunks matched the dataset exactly (mean MAE 0.0).

## 8. What this means for the robot side

- **The link has to carry the peak or the control loop slows.** Below the threshold
  `robot_client` sends an observation on every tick until a chunk arrives, not once
  per chunk, and each send blocks the control loop. At 30 fps that needs 442.5 Mbit/s
  for about the length of a round trip, every cycle. On a slower link a send takes
  longer than a 33 ms tick (at 100 Mbit/s, about 147 ms), so the tick rate drops
  during that window. Actions are still executed in order, but more slowly than they
  were recorded.
- **Expect to meet `queue_starved_ms` (500 ms).** LeRobot's server drops observations
  whose joints moved less than 1 unit in norm unless the queue is already empty, so
  an arm at rest runs its queue dry before the next chunk is computed. On loopback
  that stall is about 370 ms, once it reached 628 ms, and the 500 ms guardrail in
  `robot/config.py` has to absorb it plus the time to send 1.84 MB through the
  tunnel. The robot-side rollout client records gaps from the last executed action
  (or connection acceptance before the first action). At 500 ms it records a stall
  while leaving the last commanded target and torque unchanged; it resumes when
  actions return. This threshold does not itself abort the trial. Connection errors
  still abort it, and the trial time limit remains in force. Stall start, recovery
  and an unresolved stall at stop are recorded in the trial events. These paths
  have fake-client coverage only; physical holding and real-link timing still need
  validation.
  Lowering the camera resolution is not an option, since it would not match the
  checkpoint.
- **Aggregation is LeRobot's.** Chunks overlap by half at threshold 0.5 and the client
  blends the overlap with its `aggregate_fn_name` (default `weighted_average`). The
  first action of each chunk is labelled with the observation's timestep, which the
  client has already executed, so it is dropped.

## 9. Robot-side rollout development

`robot/rollout_client.py` provides the control helpers; `robot/rollout_worker.py`
owns them in a dedicated process. It is not web-connected yet. Its recorded runner writes:

| file, relative to the trial directory | content |
|---|---|
| `session.json` | immutable resolved client config, effective instruction, time limit, robot id, LeRobot version, Git commit and dirty state |
| `status.json` | current state, action/chunk counts, queue/latency/stall metrics, stop state and optional preview availability/errors |
| `events.jsonl` | append-only connection, acceptance, stall and terminal events |
| `frames/<camera>.jpg` | latest preview JPEG from existing observations, using the names in `robot/config.py` |
| `frames/frames.json` | observation capture timestamps and image dimensions |
| `commands/<id>.json` | immutable Stop/Release request with the current trial token |
| `command_results/<id>.json` | acknowledgement: accepted, completed, rejected or failed |

Optional `TrialPreview` reuses the recording frame publisher. It copies frames
before observation transport and encodes them outside the control thread, with
one pending frame set and a 5 Hz limit. It opens no cameras. Capture timestamps,
not JPEG modification times, determine freshness; frames older than 2 seconds
are dropped and refused by the reader. Preview errors are reported independently
of policy errors. Trial metadata and past events are never overwritten.

The process entry point is `python -m robot.rollout_worker`, with the same
required flags as the client parser: `--checkpoint`, `--task`, `--trial-dir`
(a new directory), and `--max-relative-target`. Optional flags include `--server`
and `--time-limit-s`. Starting it opens real hardware; do this only during a
separately prepared hardware test. `--help` does not import RobotClient or open devices.
The exact checkpoint string must match the selected server. Log files created by
LeRobot's import are placed under the trial's `logs/`, not the repository root.
Missing calibration is refused before connecting; native calibration mismatch
is refused without interactive prompts or automatic recalibration.

From another process, submit commands with:

```bash
python -m robot.rollout_commands stop --trial-dir=<trial-directory>
python -m robot.rollout_commands release --trial-dir=<trial-directory>
```

The command prints a queued command id. Read `command_results/<id>.json` to
confirm execution; queued or accepted is not completion. Status includes a
per-trial token, PID, process start time and `process_alive`. The submitter checks
process identity; the worker rejects malformed/stale requests and consumes each
id once. These files are local IPC, not an authenticated network API.

Stop cancels pending transport calls, ends policy workers and sends the existing
hold command. Its acknowledgement completes after cleanup; a cleanup failure
is reported as failed. Stop, the time limit and policy errors all retain device
ownership and keep the process alive for Release. Release while policy cleanup
is still active is rejected rather than queued for later execution. After
cleanup, Release disables torque, disconnects devices and frees the hardware
lock. A Release failure is recorded and can be retried with a new command id.
Successful Release exits 0 for a normal trial, 1 if policy execution failed;
initialization failure records the error and exits 1 without waiting for Release.
The trial's stopped/failed state is retained after Release, with `arm_released`
and `process_alive` updated.

SIGINT/SIGTERM request Stop and leave the owner process waiting for Release.
Do not use Runner's terminate/kill stop path for this worker: its escalation to
SIGKILL cannot perform Release. Signal/Stop handling is not a hardware emergency
stop and serial I/O or device initialization can delay it.

Synthetic-image/fake-client tests cover this lifecycle, command acknowledgements,
retry and a separate owner process receiving SIGTERM.

### Hold transport integration without hardware

`tests/test_rollout_hold_integration.py` starts the existing
`research.serve_policy --fake=hold` server and the installed LeRobot 0.6.1
RobotClient in separate processes over loopback gRPC. Only hardware construction
is replaced by `tests/rollout_fake_worker.py`; observation serialization, chunk
reception, action queue/control loop, rollout recording and command IPC are real.
The helper is test-only and is not a fake mode for the production Run page.

It creates temporary configuration files without model weights or recorded data.
Synthetic observations have the joint order, camera names and dimensions from
`robot/config.py`. Serial and VideoCapture opening are explicitly forbidden;
hardware ownership is tested with an isolated lock rather than the actual arm
lock. Outputs and subprocess logs remain under pytest's temporary directory.

Run in the robot-side environment with LeRobot async support and pytest:

```bash
python -B -m pytest tests/test_rollout_hold_integration.py -q -p no:cacheprovider
```

All five scenarios passed on macOS with LeRobot 0.6.1:

| scenario | verified behavior |
|---|---|
| explicit Stop | observations produce chunks/actions matching observed joints; preview appears; Stop retains ownership until Release |
| time limit | policy stops at its limit and the owner waits for Release |
| checkpoint mismatch | server refusal reaches failed status without policy actions |
| camera shape mismatch | server refusal names shape mismatch without policy actions |
| server disconnect | communication failure aborts the trial and retains ownership until Release |

Every scenario checks worker cleanup, the final hold target, Release acknowledgement,
torque-disable/disconnect calls on the fake devices, lock removal and process exit.
Failed trials retain their failed/aborted status and exit 1 after successful Release.
Local socket binding must be permitted to run these tests. This verifies actual
protocol/control plumbing with synthetic hardware; it does not establish physical
holding, emergency-stop timing, GPU inference or SSH tunnel behavior.
The additional Lab API integration test launches the same actual hold server and
fake-hardware client through the job executor, requests Stop, restarts the backend,
recovers the owner from disk, and requests Release. It also checks that Jobs Stop
does not terminate the owner and that the worker uses the Lab's shared hardware
lock path. The test substitutes only the worker's hardware factory/launch command.
Web wiring and actual arm + local hold validation are still pending.

### Lab rollout API

The backend exposes this file-backed contract; no browser state is required to
recover the selected job after a backend restart:

| endpoint | behavior |
|---|---|
| `POST /api/rollouts/check` | read-only configuration, server/tunnel and ownership checks |
| `POST /api/rollouts/start` | recheck and submit a local robot-owner job; returns 201 with its snapshot |
| `GET /api/rollouts` | newest-first rollout job snapshots |
| `GET /api/rollouts/<job_id>` | job, effective state, worker status, session metadata, owner liveness and read error |
| `POST /api/rollouts/<job_id>/stop` | submit Stop; returns 202 and command id |
| `POST /api/rollouts/<job_id>/release` | submit Release after a terminal policy state; returns 202 and command id |
| `GET /api/rollouts/<job_id>/commands/<command_id>` | queued, accepted, completed, rejected or failed acknowledgement |
| `GET /api/rollouts/<job_id>/logs?tail_lines=100` | existing job stdout/stderr log, bounded to 1–1000 lines |

Check/Start accept exactly `server_job_id`, nonblank `instruction`, positive finite
`max_relative_target` and `time_limit_s`. They do not accept arbitrary commands,
checkpoint paths or network addresses. The checkpoint comes from the selected
server job; local hold uses its recorded local port, SSH uses the existing running
tunnel at the robot-side loopback endpoint. The HTTP handler delegates to
`lab/core/rollout.py`, which reuses `serving.session_status` and Runner.

Start stops standalone preview after passing preflight, then launches
`python -m robot.rollout_worker` on this machine, even when inference is on SSH.
The trial gets a unique directory under `<data_dir>/rollouts/robot/`. The data
root is passed to the child so its hardware lock matches collection/calibration/
teleoperation. LocalExecutor uses the stored argv without a shell for these jobs;
instruction text cannot become shell syntax. The display command is still saved
for provenance. Fake hardware mode refuses real rollout Start.

Start is refused while collection or another rollout owns the devices. An active
rollout job continues to block new rollout, collection and standalone preview
after Stop and until Release/process exit. This check also covers startup before
the child acquires its hardware lock. The child lock remains the final authority
against other CLI/hardware processes.

`POST /api/jobs/<job_id>/stop` routes rollout jobs through the same Stop channel;
Runner never applies its terminate/kill path to them. A channel that is not ready
returns 409, not a forced stop. Release before a terminal policy state returns 409.
The worker still checks real worker-thread liveness before releasing devices;
failed cleanup can be retried after a delayed thread finishes. A 202 response
only means submission: clients must inspect the acknowledgement and status.

Owner liveness checks PID and process start time, independently of the job ledger.
Stale active status with a missing/reused owner is exposed as `ended_unknown`,
never as verified Stop/Release. Raw worker status remains available for diagnosis.
Backend shutdown does not automatically release a surviving owner. Job exit status
may be unknown after restart because LocalExecutor loses Popen handles; the trial
files still retain the stopped/failed state and Release result.

Web Start/Stop/Release, rollout preview HTTP access and physical validation remain
separate next steps.
