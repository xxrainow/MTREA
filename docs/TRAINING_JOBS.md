# Training jobs: the contract between lab and `research/train.py`

lab's Train page drives training through two channels only: the CLI
(`scripts/train.sh`, which execs `python -m research.train`) and files in the
run directory. lab never imports `research/` (`tests/test_boundary.py`), so
everything the page needs is in this document. Verified against lerobot
v0.6.1 (git tag `v0.6.1`); see the module docstring of `research/train.py` for
the exact source locations.

## 1. Describe: what to render

```bash
scripts/train.sh --describe
```

Needs no other flag and touches nothing. Prints one JSON object:

| key | meaning |
|---|---|
| `schema_version` | `1`. Bumped on an incompatible change to this output. |
| `entrypoint` | `"scripts/train.sh"` |
| `lerobot_version_tested` | `"0.6.1"` |
| `methods[]` | `name`, `label`, `description`, `flags` (lerobot flags), `max_batch_size_tested`, `oom_batch_size` (or null), `seconds_per_step` (keys are batch sizes as strings), `params` (method-specific inputs; empty for now) |
| `params[]` | one per CLI option: `name`, `flag`, `type` (`string`, `int`, `float`, `bool`, `flag`, `int_list`), `required`, `default`, `choices`, `min`, `label`, `help`, `group` |
| `fixed` | lerobot flags every run gets, and the scheduler |
| `run_name_template` | `"{method}_{dataset_name}_s{seed}_{yyyymmdd-HHMM}"` |
| `run_files` | one line per file in the run directory |

Groups: `basic` (method, dataset, dataset_root, seed, output, steps,
batch_size, save_freq) belongs on the form; `advanced` (lr, warmup_steps,
log_freq, keep_checkpoints, prune_checkpoints, wandb, wandb_project) behind a
disclosure; `fixed` (base) shown, not edited; `fake` (fake,
fake_seconds_per_step, fake_fail) only in fake mode.

Value syntax: `bool` is `--flag=true|false`; `flag` is the bare flag;
`int_list` is comma-separated (`--keep-checkpoints=5000,10000`).

Batch-size limits are hints: show a warning above `max_batch_size_tested`;
`--check` reports the same. Nothing is enforced.

`dataset_name` in the run name is the part of the repo_id after the last `/`.
The page builds `--output` as `<runs root>/<run name>`; see section 4.

## 2. Check: before Start

```bash
scripts/train.sh <the same flags as the run> --check
```

Writes nothing. Prints one JSON object and exits 0 if no check failed, 2 if
any did:

```json
{"ok": true,
 "checks": [{"id": "output", "status": "ok", "message": "..."}, ...],
 "estimated_seconds": 39200,
 "command": "scripts/train.sh --method=h0 --dataset=... --seed=0 ..."}
```

| id | fails when | warns when |
|---|---|---|
| `output` | the run directory exists and is not empty | |
| `warmup` | `--warmup-steps` >= `--steps` | |
| `stats` | `meta/stats.json` is unreadable or lacks q01/q99 for state/action (message says how to recompute) | same problem in fake mode |
| `lerobot` | `lerobot-train` is not on PATH | installed lerobot is not 0.6.1 |
| `batch_size` | | above `max_batch_size_tested` (mentions the measured OOM size) |
| `keep_checkpoints` | | a kept step is never saved (not a multiple of save_freq, not the final step) |

`estimated_seconds` is `seconds_per_step[batch_size] * steps` from
docs/COMPUTE.md, training steps only (startup and checkpoint saves are
extra), or `null` when that batch size was never measured. In fake mode it is
`fake_seconds_per_step * steps`.

`command` is the run's command line with every option written out. Submitting
it is the simplest correct JobSpec.

## 3. Start: building `JobSpec.cmd`

`JobSpec(kind=TRAIN, cmd=..., executor=...)`. Rules:

- Pass every `basic` value explicitly, `--seed` included. Defaults can change
  between commits; an explicit command line reproduces the run (AGENTS.md).
  Taking `command` from `--check` satisfies this.
- `--output` must be new for every run. A non-empty directory is refused
  (exit 2) and never overwritten.
- No `--check`, `--dry-run` or `--describe` in a submitted job.
- The command runs from the repo root on the executing machine.

```bash
scripts/train.sh --method=h0 --dataset=lab/pick_place --dataset-root=<dataset dir on the GPU host> \
  --seed=0 --output=/data/mtrea/runs/h0_pick_place_s0_20261031-1200 \
  --steps=10000 --batch-size=16 --save-freq=1000
```

### Fake vs real

Switching the page from fake to real removes `--fake` (and the `fake` group
options) and changes where things live. Nothing else changes: same flags, same
files, same stdout lines.

| | fake | real |
|---|---|---|
| flag | `--fake` | none |
| runs on | lab machine, LocalExecutor | GPU host, SSH executor |
| run directory | `data/fake/runs/<run name>` | `/data/mtrea/runs/<run name>` |
| datasets | `data/fake/collection/raw/<name>` | real recordings |
| missing quantile stats | warning on stderr, run continues | refused, exit 2 |
| lerobot | not needed (`research/fake_lerobot_train.py`) | `lerobot-train` v0.6.1 |
| losses, weights | synthetic; placeholder bytes in checkpoint files | real |
| `config.json` | `"fake": true`, `"lerobot_version": "fake"` | `"fake": false`, installed version |
| W&B url | `https://wandb.invalid/...` | real W&B run |
| failure drill | `--fake-fail=oom` (MIG NVML assert at step 2) or `error` (right after the halfway step) | |

`--fake-seconds-per-step` (default 0.01) sets the speed of a fake run.

## 4. Run directory

Every file is written atomically (temp file + rename) except the two
append-only logs. A reader may see the previous version of a JSON file, never
a partial one. In `metrics.jsonl`, ignore a last line without a newline.

| file | written | content |
|---|---|---|
| `config.json` | once, before launch | every resolved option, `fake`, the exact lerobot argv, git commit and dirty flag, lerobot/python versions, host, start time |
| `status.json` | at start, every log step, every checkpoint, on stop, at the end | see below |
| `metrics.jsonl` | one line per lerobot log step (every `--log-freq` steps) | `step`, `step_exact`, `step_logged`, `loss`, `lr`, `epoch`, `grad_norm`, `update_s`, `data_s`, `samples_per_s`, `mem_gb` (only on a GPU), `time` |
| `checkpoints.json` | whenever the set of checkpoints changes | `{"checkpoints": [...], "updated_at"}` |
| `result.json` | on stop (immediately) and at the end | `status`, `exit_code`, `end_time`, `last_step`, `last_loss`, `last_checkpoint_step`, `fake`, and `failure` when failed |
| `train.log` | streamed | lerobot's stdout and stderr, unmodified |
| `lerobot/` | by lerobot | its output_dir; checkpoints in `lerobot/checkpoints/<step>/` |

`status.json`:

| field | meaning |
|---|---|
| `status` | `running`, `done`, `failed`, `stopped` |
| `fake`, `method`, `seed`, `total_steps`, `started_at` | from the command line |
| `step`, `loss`, `lr` | from the latest log line (window averages, as lerobot logs them) |
| `seconds_per_step` | `updt_s + data_s` of the latest log line |
| `eta_s` | `(total_steps - step) * seconds_per_step`; excludes checkpoint saves |
| `last_checkpoint_step` | the latest complete checkpoint, or null |
| `wandb_url` | null unless W&B is on |
| `failure` | `oom` or `error` when failed, else null |
| `updated_at` | time of this write |

Each `checkpoints.json` entry: `step`, `path` (relative to the run directory),
`complete` (step <= the one `last` points to), `has_training_state`, `last`,
`saved_at`.

`status.json` cannot change after the process is SIGKILLed. If the job is no
longer active but `status.json` still says `running`, show the run as ended
with unknown status.

## 5. stdout protocol

Captured in the job log. One line per event, nothing else on stdout. Warnings
and errors go to stderr.

```
STATUS started
PROGRESS step=<n>/<total> loss=<x> lr=<y>
CHECKPOINT step=<n>
WANDB url=<url>
STATUS done|failed|stopped exit=<code>
```

- `CHECKPOINT step=n`: checkpoint n is complete. Every saved checkpoint is
  announced once, in order. It is reported at the next log line or checkpoint
  line after the save, so it lags by up to one `--log-freq` interval. An
  announced checkpoint can be pruned later (section 6). Use
  `checkpoints.json` for what is on disk now.
- `WANDB url=...`: emitted once, ANSI colour codes stripped.
- The files carry the same information. The lines exist for a page that reads
  the job log.

## 6. Checkpoints

lerobot v0.6.1 saves every `--save-freq` steps and always after the final
step, into `checkpoints/<step zero-padded to max(6, digits of --steps)>/`
with `pretrained_model/` (weights, configs, processors) and `training_state/`
(optimizer, scheduler, RNG, step). After each save it re-points
`checkpoints/last` by unlink + symlink.

Pruning (`--prune-checkpoints=true`, the default):

- Runs when lerobot logs "Checkpoint policy after step N" (a save starts) and
  once after the process exits.
- Deletes every checkpoint older than the one `last` points to.
- Steps listed in `--keep-checkpoints` keep `pretrained_model/` and lose only
  `training_state/`.
- Never touches `last` or anything newer, because a save may be in progress.
- If `last` cannot be read (two or more step dirs and the symlink is missing
  through a brief retry), pruning is skipped that time. It never guesses.

So a finished run holds the final checkpoint with its training state, plus
the weights of each kept step. With `--prune-checkpoints=false` every
checkpoint stays.

Only `complete` checkpoints may be evaluated or served. An incomplete one is
still being written, or was cut off by a stop or crash.

Evaluation rule (research/collect/protocol.md): evaluate the last checkpoint
of every method. Kept intermediate checkpoints are for analysis only.

## 7. Exit codes

| code | meaning | status.json / result.json |
|---|---|---|
| 0 | training finished | `done` |
| 1 | lerobot failed (its non-zero code is passed through; killed by signal N gives 128+N) | `failed`, `failure` = `oom` or `error` |
| 2 | refused before launch: bad option, non-empty output, warmup >= steps, missing stats (real mode); also `--check` with a failed check | not written (run dir not created) |
| 127 | `lerobot-train` could not be started | `failed`, `failure` = `error` |
| 143 | stopped by SIGTERM (130 for SIGINT) | `stopped` |

On SIGTERM, `result.json` and `status.json` say `stopped` immediately, the
signal is forwarded to lerobot, and the final prune and files follow when it
exits. lab's runner records a stopped job as STOPPED itself. A job that exits
143 for another reason shows as FAILED in the ledger, so read `status.json`.
On Windows the child is terminated rather than signalled, and the exit code
is lerobot's (usually 1); `status.json` still says `stopped`.

## 8. Not covered here

- The SSH executor: running the command on the GPU host and reading the run
  directory back.
- Dataset sync is a separate job: the Datasets page's GPU server panel sends
  a training-ready dataset (`lab/core/dataset_sync.py`, a JobKind.SYNC job)
  and `GET /api/datasets/real/<name>/remote` returns its `remote_path`. Use
  that as `--dataset-root` only when its `state` is `synced`.
- Resume: lerobot's `--resume` is not wired, and a stopped run's directory
  cannot be reused. A new run starts from `--base`.
- An eval split: lerobot's held-out-frame `eval_loss` is not configured, and
  held-out *tasks* are measured by rollouts, not by training loss.
- Multi-GPU: one MIG slice, one process.
