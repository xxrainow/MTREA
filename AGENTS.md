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

## Tech stack

- Python 3.10+ · PyTorch
- LeRobot (pinned to a release tag, not PyPI latest) for hardware I/O, dataset format, training
- openpi as a git submodule under `third_party/` for π0.5
- TODO: pin exact versions and package manager (uv vs pip) once the env is built

## Commands

```bash
pip install -e .                      # editable install
pytest research/ -v                   # research-side tests (no GPU/robot needed)
pytest tests/test_boundary.py         # enforces lab -> research direction
```

GPU/robot commands live in `scripts/*.sh` and need the A100 SSH host or the arm
plugged in. They will not run in a plain checkout — check before invoking.

## Structure

```
lab/core/    robot control, recording, job runner, state — knows nothing about HTTP
lab/server/  FastAPI (:8001) + ws.py — receives requests and delegates to core
lab/web/     React + Vite (:8080) — buttons and status only, no Python
research/    methods, task definitions, eval, analysis
scripts/     thin shell wrappers over `python -m research.*`
robot/configs/ so101_follower/leader.yaml, cameras.yaml, remote.yaml (gitignored)
data/        raw, datasets, norm_stats, checkpoints, rollouts, jobs, logs — gitignored, never committed
```

`lab/` is three layers that do not mix: `core/` never imports FastAPI, `server/`
API handlers never contain robot code, and `core/runner.py` never imports
`research/` — it only runs the shell wrappers as subprocesses (local or over SSH).
`lab/` and `research/` share only the `data/` layout and `robot/configs/`.

`research/` must never import from `lab/`. The dependency runs one way so
`research/` can be lifted into a standalone paper repo. `tests/test_boundary.py`
enforces this; a violation is a failing test, not a style opinion.

## Experiment conventions

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
- Robot constants (FPS, joint order, camera names) live only in `lab/config.py`.
  Nothing else defines them; everything else imports them.
- `POLICY_SERVER_HOST/PORT` in `lab/config.py` (127.0.0.1:8765) is the robot-side
  end of an SSH tunnel, never a remote address. `serve_policy` on the GPU host binds
  its own localhost only; the GPU-side port lives in `robot/configs/remote.yaml`.
- Hardware is assumed to be SO-101 throughout. Supporting another arm means
  changing `lab/config.py`, `lab/core/robot.py`, `robot/configs/`, and
  `research/transforms/` — not a config flag today.
- Analysis code must run on synthetic inputs, so correctness is testable with
  no GPU and no robot.

## Boundaries

- Never commit anything under `data/`, `.env`, or calibration files.
- Never edit files under `third_party/` — it is a submodule.
- File ownership is in `.github/CODEOWNERS`; respect it rather than duplicating
  the list here.

## Keeping this file current

This file is the project's memory across sessions. When a decision is made or
reversed update this file in the same change, or the next session will repeat the mistake.
