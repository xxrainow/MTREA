"""Training entrypoint: a thin bookkeeping wrapper around LeRobot's `lerobot-train`.

    python -m research.train --method=h0 --dataset=<repo_id> --output=data/runs/<run_id> --seed=0
    python -m research.train --describe               # JSON: methods, options, run files
    python -m research.train <run flags> --check      # JSON preflight; writes nothing
    python -m research.train <run flags> --fake       # fake_lerobot_train.py stands in for lerobot

Training logic lives in lerobot. This module only makes a run reproducible and
observable: it refuses to overwrite a past run or to start on a dataset without
the quantile stats pi05 needs, writes config.json (every resolved value, the
exact lerobot argv, git commit, versions) before launching, streams the child's
output to train.log, keeps status.json, metrics.jsonl and checkpoints.json
current while it runs, prunes superseded checkpoints, and records how the run
ended in result.json. docs/TRAINING_JOBS.md is the contract lab builds on.

stdout carries only parseable lines for lab's runner:
    STATUS started
    PROGRESS step=<n>/<total> loss=<x> lr=<y>
    CHECKPOINT step=<n>        checkpoint n is complete (lerobot's `last` points to it)
    WANDB url=<url>
    STATUS done|failed|stopped exit=<code>

Read from the lerobot v0.6.1 source (git tag v0.6.1), not from memory:
- flags: policy.type, policy.pretrained_path, policy.train_expert_only,
  policy.freeze_vision_encoder, policy.gradient_checkpointing, policy.dtype,
  policy.push_to_hub, policy.optimizer_lr, policy.scheduler_*, dataset.repo_id,
  dataset.root, output_dir, job_name, seed, steps, batch_size, save_freq,
  log_freq, and WandBConfig's enable, project, disable_artifact.
- MetricsTracker.__str__ (utils/logging_utils.py) and the meters registered in
  scripts/lerobot_train.py: step/smpl/ep via format_big_number, epch, loss,
  grdn, lr, updt_s, data_s, smp/s, and mem_gb only when CUDA is available.
- scripts/lerobot_train.py logs "Checkpoint policy after step N" before
  save_checkpoint() and then update_last_checkpoint() (common/train_utils.py)
  re-points checkpoints/last by unlink + symlink_to(<step dir name>).
- Step dirs are zero-padded to max(6, len(str(steps))) digits, each holding
  pretrained_model/ and training_state/.
- W&B starts only if wandb.enable and wandb.project; WandBLogger logs
  "Track this run --> <url>", the url wrapped in termcolor codes.
TODO(GPU host): re-check the above if the installed lerobot is not v0.6.1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_LEROBOT = Path(__file__).resolve().with_name("fake_lerobot_train.py")

LEROBOT_VERSION_TESTED = "0.6.1"
DESCRIBE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Method:
    label: str
    description: str
    flags: dict[str, str]
    # Batch-size limits on one A100 MIG 3g.40gb slice (docs/COMPUTE.md). Hints for
    # the UI and --check, never enforced.
    max_batch_size_tested: int
    oom_batch_size: Optional[int]
    # Measured updt_s by batch size, same source.
    seconds_per_step: dict[int, float]
    # Method-specific inputs. Empty for now; reserved for H1+.
    params: tuple = field(default=())


# H1+ are added here. An oracle run is the h0 recipe on the held-out task's own
# dataset, not a separate method.
METHODS: dict[str, Method] = {
    "h0": Method(
        label="H0 · full fine-tuning",
        description="Naive single-task fine-tuning: every parameter trains. "
                    "The baseline every other method is compared against.",
        # lerobot's defaults, stated explicitly.
        flags={
            "policy.train_expert_only": "false",
            "policy.freeze_vision_encoder": "false",
        },
        max_batch_size_tested=16,
        oom_batch_size=32,
        seconds_per_step={1: 0.66, 8: 2.17, 16: 3.92},
    ),
    "expert_only": Method(
        label="Expert only",
        description="Only the action expert trains; all of PaliGemma, vision tower included, stays frozen.",
        flags={
            # train_expert_only freezes the vision tower too, so freeze_vision_encoder
            # is redundant; false matches the smoke test.
            "policy.train_expert_only": "true",
            "policy.freeze_vision_encoder": "false",
        },
        max_batch_size_tested=8,
        oom_batch_size=None,
        seconds_per_step={1: 0.31, 2: 0.48, 4: 0.83, 8: 1.55},
    ),
}

# Same for every method; not exposed as options.
FIXED_POLICY_FLAGS = {
    "policy.type": "pi05",
    "policy.gradient_checkpointing": "true",
    "policy.dtype": "bfloat16",  # pi05 defaults to float32
    "policy.push_to_hub": "false",  # defaults to true
}
# Without disable_artifact, lerobot uploads every checkpoint to W&B.
FIXED_WANDB_FLAGS = {"wandb.disable_artifact": "true"}

# PI05Config defaults in v0.6.1: optimizer_lr is the cosine schedule's peak,
# scheduler_decay_lr its floor (a tenth of the peak), scheduler_warmup_steps 1000.
PI05_OPTIMIZER_LR = 2.5e-5
PI05_DECAY_LR_DIVISOR = 10
PI05_WARMUP_STEPS = 1000

# On a MIG slice an OOM surfaces as the NVML assert, not as "CUDA out of memory"
# (see docs/COMPUTE.md).
OOM_MARKERS = ("CUDA out of memory", "NVML_SUCCESS == r INTERNAL ASSERT FAILED")

# pi05 normalizes these with QUANTILES; lerobot fails on the first batch without q01/q99.
QUANTILE_FEATURES = ("observation.state", "action")

# One lerobot training log line, from MetricsTracker.__str__ via logging.info:
#   INFO 2026-10-04 12:00:00 ot_train.py:641 step:2K smpl:2K ep:3 epch:0.13 loss:0.512 grdn:3.210 lr:2.5e-05 updt_s:0.812 ...
# search(), not match(): tqdm's \r-redrawn bar can share a line with the log record.
METRICS_LINE = re.compile(
    r"\bstep:(?P<step>\d+[KMBTQ]?)\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+"
    r"loss:(?P<loss>\S+)\s+grdn:\S+\s+lr:(?P<lr>\S+)"
)
METRICS_TOKEN = re.compile(r"([^\s:]+):(\S+)")
# Log name -> metrics.jsonl key, for the optional values of a metrics line.
EXTRA_METRICS = {"epch": "epoch", "grdn": "grad_norm", "updt_s": "update_s", "data_s": "data_s",
                 "smp/s": "samples_per_s", "mem_gb": "mem_gb"}
CHECKPOINT_LINE = re.compile(r"Checkpoint policy after step (\d+)")
WANDB_LINE = re.compile(r"Track this run --> (\S+)")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
TRAINING_STATE_DIR = "training_state"

RUN_NAME_TEMPLATE = "{method}_{dataset_name}_s{seed}_{yyyymmdd-HHMM}"
RUN_FILES = {
    "config.json": "every resolved option, the exact lerobot argv, git commit and versions; written before launch",
    "status.json": "live state: status, step, loss, lr, seconds_per_step, eta_s, last_checkpoint_step, wandb_url",
    "metrics.jsonl": "one line per lerobot log step: step, loss, lr, grad_norm, update_s, data_s, ...",
    "checkpoints.json": "checkpoints on disk: step, path, complete, has_training_state, last, saved_at",
    "result.json": "how the run ended: status, exit_code, failure, last_step, last_checkpoint_step",
    "train.log": "lerobot's full stdout and stderr",
    "lerobot/": "lerobot's output_dir; checkpoints under lerobot/checkpoints/<step>/",
}

_SUFFIXES = ["", "K", "M", "B", "T", "Q"]


def classify_failure(lines) -> str:
    """"oom" if any of the lines shows an out-of-memory error, else "error"."""
    return "oom" if any(marker in line for line in lines for marker in OOM_MARKERS) else "error"


def decay_lr(peak_lr: float) -> float:
    return float(f"{peak_lr / PI05_DECAY_LR_DIVISOR:.12g}")  # 1e-6, not 1.0000000000000002e-06


def scheduler_flags(args: argparse.Namespace) -> dict[str, str]:
    """Cosine schedule decaying over exactly --steps.

    v0.6.1's CosineDecayWithWarmupSchedulerConfig.build() rescales warmup and
    decay only when steps < scheduler_decay_steps (30000); with more steps the
    lr would sit at the floor after step 30000. Passing decay_steps=steps
    makes the schedule follow --steps either way, and the warmup is ours, not
    a rescaled one.
    """
    return {
        "policy.scheduler_warmup_steps": str(args.warmup_steps),
        "policy.scheduler_decay_steps": str(args.steps),
        "policy.scheduler_decay_lr": repr(decay_lr(args.lr)),
    }


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def parse_metrics_line(line: str) -> Optional[dict]:
    """Return {"step": token, "loss": float, "lr": float, ...} for a training log line, else None.

    `step` stays the raw token ("200", "2K"); see resolve_step. epoch, grad_norm,
    update_s, data_s, samples_per_s and mem_gb are added when the line has them.
    """
    m = METRICS_LINE.search(line)
    if m is None:
        return None
    try:
        out = {"step": m["step"], "loss": float(m["loss"]), "lr": float(m["lr"])}
    except ValueError:
        return None
    record = line[m.start():].split("\r", 1)[0]
    for name, value in METRICS_TOKEN.findall(record):
        if name in EXTRA_METRICS:
            try:
                out[EXTRA_METRICS[name]] = float(value)
            except ValueError:
                pass
    return out


def _format_big_number(num: float) -> str:
    # Mirrors lerobot.utils.utils.format_big_number(num, precision=0).
    for suffix in _SUFFIXES:
        if abs(num) < 1000:
            return f"{num:.0f}{suffix}"
        num /= 1000.0
    return str(num)


def resolve_step(token: str, prev_step: int, log_freq: int) -> tuple[int, bool]:
    """Recover the exact step from lerobot's rounded token. Returns (step, exact).

    lerobot logs every `log_freq` steps but prints the step rounded to a suffix
    (1500 -> "2K"), so past 1000 the exact value is prev_step + log_freq, checked
    against the rounded token. If they disagree, fall back to the token's value.
    """
    if token.isdigit():
        return int(token), True
    expected = prev_step + log_freq
    if _format_big_number(expected) == token:
        return expected, True
    return int(float(token[:-1]) * 1000 ** _SUFFIXES.index(token[-1])), False


# --- CLI ---

def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _nonneg_float(value: str) -> float:
    x = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {x}")
    return x


def _int_list(value: str) -> tuple[int, ...]:
    try:
        steps = {_positive_int(v) for v in value.split(",") if v.strip()}
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated steps, got {value!r}") from e
    return tuple(sorted(steps))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.train", description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=sorted(METHODS), help="fine-tuning method")
    p.add_argument("--dataset", required=True, help="LeRobot dataset repo_id")
    p.add_argument("--dataset-root", default=None, help="local dataset path, if not on the Hub")
    p.add_argument("--seed", required=True, type=int, help="explicit; never defaulted")
    p.add_argument("--output", required=True,
                   help="run directory; must not exist or be empty, e.g. data/runs/h0_pickplace_s0_20261031-1200")
    p.add_argument("--steps", type=_positive_int, default=10000, help="optimizer steps")
    # 16 is the throughput optimum for full fine-tuning on one MIG 3g.40gb slice (docs/COMPUTE.md).
    p.add_argument("--batch-size", type=_positive_int, default=16,
                   help="samples per step; lerobot v0.6.1 has no gradient accumulation")
    p.add_argument("--save-freq", type=_positive_int, default=1000,
                   help="save a checkpoint every N steps; the final step is always saved")
    p.add_argument("--base", default="lerobot/pi05_base", help="pretrained policy to start from")
    p.add_argument("--lr", type=float, default=PI05_OPTIMIZER_LR, help="peak of the cosine schedule")
    p.add_argument("--warmup-steps", type=_positive_int, default=PI05_WARMUP_STEPS,
                   help="linear warmup steps; must be below --steps")
    p.add_argument("--log-freq", type=_positive_int, default=200,
                   help="passed to lerobot; also needed to recover exact steps from its log")
    p.add_argument("--keep-checkpoints", type=_int_list, default=(),
                   help="comma-separated steps whose weights survive pruning (training state is still removed)")
    p.add_argument("--prune-checkpoints", type=_bool, default=True,
                   help="delete checkpoints older than the latest complete one")
    p.add_argument("--wandb", type=_bool, default=False, help="also log to Weights & Biases")
    p.add_argument("--wandb-project", default="mtrea", help="W&B project, used when --wandb=true")
    p.add_argument("--fake", action="store_true",
                   help="run research/fake_lerobot_train.py instead of lerobot-train (no GPU, synthetic losses)")
    p.add_argument("--fake-seconds-per-step", type=_nonneg_float, default=0.01,
                   help="fake mode: wall time per step")
    p.add_argument("--fake-fail", choices=["none", "oom", "error"], default="none",
                   help="fake mode: fail with the MIG OOM assert at step 2, or with an error halfway")
    p.add_argument("--describe", action="store_true",
                   help="print methods and options as JSON and exit; needs no other flag")
    p.add_argument("--check", action="store_true",
                   help="print preflight checks as JSON, write nothing; exit 0 if none failed, else 2")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and lerobot command, write nothing")
    p.add_argument("--lerobot-bin", default="lerobot-train",
                   help="command to run; may include arguments, e.g. a python + stub path")
    return p


def split_command(cmd: str) -> list[str]:
    """Split --lerobot-bin into argv. Windows paths keep their backslashes."""
    if os.name != "nt":
        return shlex.split(cmd)
    return [part[1:-1] if len(part) > 1 and part[0] == part[-1] == '"' else part
            for part in shlex.split(cmd, posix=False)]


def build_lerobot_argv(args: argparse.Namespace, output: Path) -> list[str]:
    flags = {
        **FIXED_POLICY_FLAGS,
        "policy.pretrained_path": args.base,
        **METHODS[args.method].flags,
        "policy.optimizer_lr": repr(args.lr),  # the policy preset overrides --optimizer.lr
        **scheduler_flags(args),
        "dataset.repo_id": args.dataset,
        **({"dataset.root": args.dataset_root} if args.dataset_root else {}),
        # lerobot refuses an existing output_dir; our run dir exists by now.
        "output_dir": (output / "lerobot").as_posix(),
        "job_name": output.name,
        "seed": str(args.seed),
        "steps": str(args.steps),
        "batch_size": str(args.batch_size),
        "save_freq": str(args.save_freq),
        "log_freq": str(args.log_freq),
        "wandb.enable": "true" if args.wandb else "false",
        # lerobot's own default project is "lerobot"; ours keeps runs together.
        "wandb.project": args.wandb_project,
        **FIXED_WANDB_FLAGS,
    }
    command = [sys.executable, str(FAKE_LEROBOT)] if args.fake else split_command(args.lerobot_bin)
    return [*command, *(f"--{k}={v}" for k, v in flags.items())]


# --- --describe ---

# Options that are not training inputs; they carry no UI metadata.
INTERNAL_OPTIONS = ("help", "describe", "check", "dry_run", "lerobot_bin")

# dest -> (group, label). Name, flag, type, default, choices, min and help come
# from the parser itself, so the UI cannot drift from what the CLI accepts.
PARAM_UI: dict[str, tuple[str, str]] = {
    "method": ("basic", "Method"),
    "dataset": ("basic", "Dataset"),
    "dataset_root": ("basic", "Dataset path"),
    "seed": ("basic", "Seed"),
    "output": ("basic", "Run directory"),
    "steps": ("basic", "Steps"),
    "batch_size": ("basic", "Batch size"),
    "save_freq": ("basic", "Checkpoint every"),
    "lr": ("advanced", "Peak learning rate"),
    "warmup_steps": ("advanced", "Warmup steps"),
    "log_freq": ("advanced", "Log every"),
    "keep_checkpoints": ("advanced", "Keep checkpoints"),
    "prune_checkpoints": ("advanced", "Prune checkpoints"),
    "wandb": ("advanced", "Weights & Biases"),
    "wandb_project": ("advanced", "W&B project"),
    "base": ("fixed", "Base model"),
    "fake": ("fake", "Fake backend"),
    "fake_seconds_per_step": ("fake", "Fake seconds per step"),
    "fake_fail": ("fake", "Fake failure"),
}

_UI_TYPES = {int: "int", _positive_int: "int", float: "float", _nonneg_float: "float",
             _bool: "bool", _int_list: "int_list"}
_UI_MIN = {_positive_int: 1, _nonneg_float: 0}


def _options(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in parser._actions if a.option_strings]


def _param(action: argparse.Action) -> dict:
    group, label = PARAM_UI[action.dest]
    flag = isinstance(action, argparse._StoreTrueAction)
    default = action.default
    return {
        "name": action.dest,
        "flag": action.option_strings[0],
        "type": "flag" if flag else _UI_TYPES.get(action.type, "string"),
        "required": action.required,
        "default": list(default) if isinstance(default, tuple) else default,
        "choices": list(action.choices) if action.choices else None,
        "min": _UI_MIN.get(action.type),
        "label": label,
        "help": action.help,
        "group": group,
    }


def describe() -> dict:
    parser = build_parser()
    return {
        "schema_version": DESCRIBE_SCHEMA_VERSION,
        "entrypoint": "scripts/train.sh",
        "lerobot_version_tested": LEROBOT_VERSION_TESTED,
        "methods": [
            {"name": name, "label": m.label, "description": m.description, "flags": m.flags,
             "max_batch_size_tested": m.max_batch_size_tested, "oom_batch_size": m.oom_batch_size,
             "seconds_per_step": {str(bs): s for bs, s in m.seconds_per_step.items()},
             "params": list(m.params)}
            for name, m in METHODS.items()
        ],
        "params": [_param(a) for a in _options(parser) if a.dest not in INTERNAL_OPTIONS],
        "fixed": {
            "lerobot_flags": {**FIXED_POLICY_FLAGS, **FIXED_WANDB_FLAGS},
            "scheduler": "cosine_decay_with_warmup; decays over --steps to --lr / "
                         f"{PI05_DECAY_LR_DIVISOR}",
        },
        "run_name_template": RUN_NAME_TEMPLATE,
        "run_files": RUN_FILES,
    }


# --- dataset checks ---

class StatsError(RuntimeError):
    """The dataset's meta/stats.json is unreadable or lacks what pi05 needs."""


def missing_quantile_stats(stats: dict) -> list[str]:
    """Features in QUANTILE_FEATURES whose stats lack q01 or q99."""
    return [f for f in QUANTILE_FEATURES
            if not isinstance(stats.get(f), dict) or not {"q01", "q99"} <= stats[f].keys()]


def _read_hub_stats(repo_id: str) -> dict:
    # Lazy import: only needed for Hub datasets, and absent on a laptop without lerobot.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise StatsError("huggingface_hub is not installed; pass --dataset-root") from e
    try:
        path = hf_hub_download(repo_id, "meta/stats.json", repo_type="dataset")
    except Exception as e:  # network, auth, missing file: all mean "cannot verify"
        raise StatsError(f"cannot download meta/stats.json for {repo_id} ({e.__class__.__name__}: {e})") from e
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stats(dataset: str, dataset_root: Optional[str]) -> dict:
    if dataset_root is None:
        return _read_hub_stats(dataset)
    path = Path(dataset_root) / "meta" / "stats.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StatsError(f"cannot read {path} ({e.__class__.__name__}: {e})") from e


def recompute_hint(dataset: str, dataset_root: Optional[str], one_line: bool = False) -> str:
    # recompute_stats copies its input from a local root, so a Hub dataset is fetched first
    # (as experiments/gpu_smoke_test.sh does).
    root = dataset_root or "<src_dir>"
    edit = (f"lerobot-edit-dataset --repo_id={dataset} --root={root} --new_root=<copy_dir> "
            "--operation.type=recompute_stats")
    fetch = ("" if dataset_root else
             f"python -c \"from lerobot.datasets import LeRobotDataset; LeRobotDataset('{dataset}', root='<src_dir>')\"")
    if one_line:
        steps = f"{fetch} && {edit}" if fetch else edit
        return f"{steps}, then train with --dataset-root=<copy_dir>"
    return ("Recompute them into a copy, then train on that copy with --dataset-root=<copy_dir>:\n"
            + (f"  {fetch}\n" if fetch else "") + f"  {edit}")


def _stats_problem(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    """("ok" | "unchecked" | "missing", message, missing features) for the dataset's quantile stats."""
    try:
        missing = missing_quantile_stats(load_stats(args.dataset, args.dataset_root))
    except StatsError as e:
        return "unchecked", str(e), []
    if missing:
        return "missing", (f"{args.dataset} meta/stats.json lacks q01/q99 for {missing}; pi05 normalizes "
                           "state/action with QUANTILES and would fail on the first batch"), missing
    return "ok", f"{args.dataset} meta/stats.json has q01/q99 for {list(QUANTILE_FEATURES)}", []


def _check(id_: str, status: str, message: str) -> dict:
    return {"id": id_, "status": status, "message": message}


def saved_steps(steps: int, save_freq: int) -> set[int]:
    """Steps lerobot saves a checkpoint at (should_save_checkpoint in v0.6.1)."""
    return set(range(save_freq, steps + 1, save_freq)) | {steps}


def blocking_checks(args: argparse.Namespace, include_stats: bool = True) -> list[dict]:
    """Checks a real run refuses to start without. Fake mode only warns about stats."""
    output = Path(args.output)
    checks = []
    if output.exists() and (output.is_file() or any(output.iterdir())):
        checks.append(_check("output", "fail", f"--output {output} already exists and is not empty; "
                                               "past runs are never overwritten, pick a new run directory"))
    else:
        checks.append(_check("output", "ok", f"--output {output} is free"))
    if args.warmup_steps >= args.steps:
        checks.append(_check("warmup", "fail",
                             f"--warmup-steps ({args.warmup_steps}) must be below --steps ({args.steps})"))
    else:
        checks.append(_check("warmup", "ok", f"warmup {args.warmup_steps} of {args.steps} steps"))
    if not include_stats:
        return checks
    kind, message, _ = _stats_problem(args)
    if kind == "missing":
        message += ". " + recompute_hint(args.dataset, args.dataset_root, one_line=True)
    if kind == "ok":
        checks.append(_check("stats", "ok", message))
    elif args.fake:
        checks.append(_check("stats", "warn", f"{message} (ignored in fake mode)"))
    else:
        checks.append(_check("stats", "fail", message))
    return checks


def advisory_checks(args: argparse.Namespace) -> list[dict]:
    """Warnings only; a run starts regardless."""
    method = METHODS[args.method]
    checks = []
    if args.batch_size > method.max_batch_size_tested:
        note = (f"; {method.oom_batch_size} ran out of memory"
                if method.oom_batch_size and args.batch_size >= method.oom_batch_size else "")
        checks.append(_check("batch_size", "warn",
                             f"batch size {args.batch_size} is above the largest tested for {args.method} "
                             f"({method.max_batch_size_tested}) on one MIG 3g.40gb slice{note}"))
    else:
        checks.append(_check("batch_size", "ok", f"batch size {args.batch_size} is within the tested range "
                                                 f"for {args.method} (<= {method.max_batch_size_tested})"))
    never = [s for s in args.keep_checkpoints if s not in saved_steps(args.steps, args.save_freq)]
    if never:
        checks.append(_check("keep_checkpoints", "warn",
                             f"--keep-checkpoints {never} are never saved (every {args.save_freq} steps "
                             f"and at step {args.steps}), so nothing is kept for them"))
    else:
        checks.append(_check("keep_checkpoints", "ok", f"keep {list(args.keep_checkpoints) or 'none'}"))
    return checks


def lerobot_check(args: argparse.Namespace) -> dict:
    if args.fake:
        return _check("lerobot", "ok", f"fake backend {FAKE_LEROBOT.name}; lerobot is not needed")
    binary = split_command(args.lerobot_bin)[0]
    if shutil.which(binary) is None:
        return _check("lerobot", "fail", f"{binary!r} is not on PATH")
    version = _lerobot_version()
    if version != LEROBOT_VERSION_TESTED:
        return _check("lerobot", "warn", f"lerobot {version} is installed; flags and log formats were "
                                         f"verified against {LEROBOT_VERSION_TESTED}")
    return _check("lerobot", "ok", f"{binary} found, lerobot {version}")


def estimated_seconds(args: argparse.Namespace) -> Optional[float]:
    """Training-step time only; startup, data loading warm-up and checkpoint saves are extra."""
    if args.fake:
        return round(args.fake_seconds_per_step * args.steps, 1)
    per_step = METHODS[args.method].seconds_per_step.get(args.batch_size)
    return None if per_step is None else round(per_step * args.steps)


def job_command(args: argparse.Namespace) -> str:
    """The scripts/train.sh command line for these args, every option explicit."""
    parts = ["scripts/train.sh"]
    for action in _options(build_parser()):
        dest, value = action.dest, getattr(args, action.dest, None)
        if dest in INTERNAL_OPTIONS and not (dest == "lerobot_bin" and value != action.default):
            continue
        if PARAM_UI.get(dest, ("",))[0] == "fake" and not args.fake:
            continue
        if value is None or value == ():
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, tuple):
            value = ",".join(map(str, value))
        elif isinstance(value, float):
            value = repr(value)
        parts.append(f"{flag}={value}")
    return shlex.join(parts)


def run_check(args: argparse.Namespace) -> int:
    checks = blocking_checks(args) + [lerobot_check(args)] + advisory_checks(args)
    ok = not any(c["status"] == "fail" for c in checks)
    print(json.dumps({"ok": ok, "checks": checks, "estimated_seconds": estimated_seconds(args),
                      "command": job_command(args)}, indent=2))
    return 0 if ok else 2


# --- provenance ---

def _git(*cmd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _lerobot_version() -> str:
    try:
        return metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        return "unknown"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_config(args: argparse.Namespace, argv: list[str]) -> dict:
    status = _git("status", "--porcelain")
    return {
        "args": {k: v for k, v in vars(args).items() if k not in ("dry_run", "check", "describe")},
        "fake": args.fake,
        "method_flags": METHODS[args.method].flags,
        "scheduler": {"type": "cosine_decay_with_warmup", "peak_lr": args.lr,
                      "decay_lr": decay_lr(args.lr),
                      "warmup_steps": args.warmup_steps, "decay_steps": args.steps},
        "lerobot_argv": argv,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "lerobot_version": "fake" if args.fake else _lerobot_version(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "start_time": _now(),
    }


_tmp_names = itertools.count()


def _write_json(path: Path, obj: dict) -> None:
    # A unique temp name: the signal handler can write the same file while the
    # main loop is between write and replace.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{next(_tmp_names)}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)  # never leave a half-written file behind
            return
        except PermissionError:  # Windows: a reader has the target open
            if attempt == 39:
                raise
            time.sleep(0.025)


def _say(line: str) -> None:
    print(line, flush=True)


# --- checkpoints ---

def step_dirs(ckpt_dir: Path) -> dict[int, Path]:
    """Step number -> checkpoint directory. `last` and anything else non-numeric is skipped."""
    try:
        entries = list(ckpt_dir.iterdir())
    except OSError:
        return {}
    return {int(p.name): p for p in entries
            if p.name.isdigit() and p.is_dir() and not p.is_symlink()}


def read_last_step(ckpt_dir: Path, retries: int = 10, delay_s: float = 0.05) -> Optional[int]:
    """The step `last` points to, or None if no checkpoint is known to be complete.

    lerobot re-points `last` by unlink + symlink, so a read can land in between.
    With fewer than two step dirs a missing `last` just means the first save is
    still running; with two or more it must be that gap, so retry briefly.
    """
    link = ckpt_dir / LAST_CHECKPOINT_LINK
    for attempt in range(retries + 1):
        try:
            name = Path(os.readlink(link)).name
        except OSError:  # missing, or not a symlink
            if len(step_dirs(ckpt_dir)) < 2 or attempt == retries:
                return None
            time.sleep(delay_s)
            continue
        return int(name) if name.isdigit() else None
    return None


def prune_checkpoints(ckpt_dir: Path, keep: Iterable[int] = (), last: Optional[int] = None) -> Optional[int]:
    """Delete checkpoints older than the one `last` points to; return that step.

    Steps in `keep` lose only their training state. `last` and anything newer
    (a save may be in progress) are never touched. `last` is read from disk
    unless the caller just read it. If it is unknown nothing is deleted and
    None is returned.
    """
    if last is None:
        last = read_last_step(ckpt_dir)
    if last is None:
        return None
    keep = set(keep)
    for step, path in sorted(step_dirs(ckpt_dir).items()):
        if step >= last:
            continue
        target = path / TRAINING_STATE_DIR if step in keep else path
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
        except OSError as e:
            print(f"warning: cannot prune {target}: {e}", file=sys.stderr)
    return last


def list_checkpoints(run_dir: Path, last: Optional[int]) -> list[dict]:
    out = []
    for step, path in sorted(step_dirs(run_dir / "lerobot" / CHECKPOINTS_DIR).items()):
        # pretrained_model's mtime is its last written file, and pruning a
        # sibling training_state does not change it.
        model = path / "pretrained_model"
        try:
            mtime = (model if model.exists() else path).stat().st_mtime
        except OSError:  # pruned between listing and stat
            continue
        out.append({
            "step": step,
            "path": path.relative_to(run_dir).as_posix(),
            "complete": last is not None and step <= last,
            "has_training_state": (path / TRAINING_STATE_DIR).is_dir(),
            "last": step == last,
            "saved_at": datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds"),
        })
    return out


# --- run ---

def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--describe" in argv:
        print(json.dumps(describe(), indent=2))
        return 0
    args = build_parser().parse_args(argv)
    output = Path(args.output)

    if args.check:
        return run_check(args)

    # Stats are fatal in a real run; in a dry run only reported, so a preview works offline.
    failed = False
    for check in blocking_checks(args, include_stats=not args.dry_run):
        if check["status"] == "fail":
            print(f"error: {check['message']}", file=sys.stderr)
            failed = True
        elif check["status"] == "warn":
            print(f"warning: {check['message']}", file=sys.stderr)
    if failed:
        return 2
    if args.dry_run:
        kind, message, missing = _stats_problem(args)
        if kind == "unchecked":
            print(f"STATS unchecked: {message}")
        elif kind == "missing":
            print(f"STATS missing q01/q99 for {missing} — run "
                  + recompute_hint(args.dataset, args.dataset_root, one_line=True))
        else:
            print("STATS ok")

    lerobot_argv = build_lerobot_argv(args, output)
    config = build_config(args, lerobot_argv)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        print("COMMAND " + shlex.join(lerobot_argv))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    env = {**os.environ,
           # A piped Python child block-buffers stdout; without this, train.log
           # lags and a crash can lose its last lines.
           "PYTHONUNBUFFERED": "1"}
    if args.fake:
        env.update(MTREA_FAKE_SECONDS_PER_STEP=repr(args.fake_seconds_per_step),
                   MTREA_FAKE_FAIL=args.fake_fail)
    return _Run(args, output).run(lerobot_argv, env)


class _Run:
    """One lerobot child process and the run-directory files that follow it."""

    def __init__(self, args: argparse.Namespace, output: Path):
        self.output = output
        self.ckpt_dir = output / "lerobot" / CHECKPOINTS_DIR
        self.total_steps = args.steps
        self.log_freq = args.log_freq
        self.fake = args.fake
        self.prune = args.prune_checkpoints
        self.keep = args.keep_checkpoints
        self.proc: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self.last_loss: Optional[float] = None
        self.checkpoints: Optional[list[dict]] = None
        self.status = {
            "status": "running", "fake": args.fake, "method": args.method, "seed": args.seed,
            "step": 0, "total_steps": args.steps, "loss": None, "lr": None,
            "seconds_per_step": None, "eta_s": None, "last_checkpoint_step": None,
            "wandb_url": None, "failure": None, "started_at": _now(), "updated_at": None,
        }

    def write_status(self, **changes) -> None:
        self.status.update(changes, updated_at=_now())
        _write_json(self.output / "status.json", self.status)

    def result(self, status: str, exit_code: Optional[int], failure: Optional[str] = None) -> dict:
        out = {"status": status, "exit_code": exit_code, "end_time": _now(),
               "last_step": self.status["step"] or None, "last_loss": self.last_loss,
               "last_checkpoint_step": self.status["last_checkpoint_step"], "fake": self.fake}
        if status == "failed":
            out["failure"] = failure or "error"
        return out

    def refresh_checkpoints(self, prune: bool) -> None:
        """Announce newly completed checkpoints, prune if asked, rewrite checkpoints.json."""
        last = read_last_step(self.ckpt_dir)
        if last is not None:
            # `last` can move several saves between two reads when we lag behind
            # lerobot; every step dir up to it is complete, so announce each,
            # before pruning can remove it.
            announced = self.status["last_checkpoint_step"] or 0
            for step in sorted(s for s in step_dirs(self.ckpt_dir) if announced < s <= last):
                _say(f"CHECKPOINT step={step}")
            self.status["last_checkpoint_step"] = last
            if prune and self.prune:
                prune_checkpoints(self.ckpt_dir, self.keep, last=last)
        checkpoints = list_checkpoints(self.output, last)
        if checkpoints != self.checkpoints:
            self.checkpoints = checkpoints
            _write_json(self.output / "checkpoints.json", {"checkpoints": checkpoints, "updated_at": _now()})

    def on_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        # Record the stop now: lab's runner SIGKILLs us a few seconds after SIGTERM.
        self.status["status"] = "stopped"
        _write_json(self.output / "result.json", self.result("stopped", None))
        self.write_status()
        if self.proc is not None and self.proc.poll() is None:
            if os.name == "nt":
                self.proc.terminate()  # Windows Popen cannot deliver SIGINT/SIGTERM
            else:
                self.proc.send_signal(signum)

    def on_metrics(self, parsed: dict, prev_step: int, metrics) -> int:
        step, exact = resolve_step(parsed["step"], prev_step, self.log_freq)
        self.last_loss = parsed["loss"]
        extras = {k: v for k, v in parsed.items() if k not in ("step", "loss", "lr")}
        record = {"step": step, "step_exact": exact, "step_logged": parsed["step"],
                  "loss": parsed["loss"], "lr": parsed["lr"], **extras, "time": _now()}
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        _say(f"PROGRESS step={step}/{self.total_steps} loss={parsed['loss']} lr={parsed['lr']}")
        per_step = None
        if "update_s" in extras and "data_s" in extras:
            per_step = round(extras["update_s"] + extras["data_s"], 4)
        self.refresh_checkpoints(prune=False)
        self.write_status(step=step, loss=parsed["loss"], lr=parsed["lr"], seconds_per_step=per_step,
                          eta_s=None if per_step is None else round((self.total_steps - step) * per_step))
        return step

    def on_line(self, text: str, prev_step: int, metrics) -> int:
        if self.status["wandb_url"] is None and (m := WANDB_LINE.search(text)):
            _say(f"WANDB url={m[1]}")
            self.write_status(wandb_url=m[1])
        if CHECKPOINT_LINE.search(text):
            self.refresh_checkpoints(prune=True)
            self.write_status()
        parsed = parse_metrics_line(text)
        return prev_step if parsed is None else self.on_metrics(parsed, prev_step, metrics)

    def run(self, lerobot_argv: list[str], env: dict) -> int:
        tail: deque[str] = deque(maxlen=30)  # printed to stderr if the run fails
        handled = [signal.SIGINT, signal.SIGTERM]
        previous = {s: signal.signal(s, self.on_signal) for s in handled}
        try:
            self.write_status()
            _say("STATUS started")
            try:
                self.proc = subprocess.Popen(lerobot_argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, env=env)
            except OSError as e:
                print(f"error: cannot start {lerobot_argv[0]!r}: {e}", file=sys.stderr)
                _write_json(self.output / "result.json", self.result("failed", 127))
                self.write_status(status="failed", failure="error")
                _say("STATUS failed exit=127")
                return 127

            with open(self.output / "train.log", "wb") as log, \
                    open(self.output / "metrics.jsonl", "a", encoding="utf-8") as metrics:
                prev_step = 0
                for raw in self.proc.stdout:
                    log.write(raw)
                    log.flush()
                    text = strip_ansi(raw.decode("utf-8", errors="replace"))
                    # Keep what a terminal would show: the text after tqdm's last \r redraw.
                    tail.append(text.rstrip("\r\n").rsplit("\r", 1)[-1])
                    prev_step = self.on_line(text, prev_step, metrics)
            rc = self.proc.wait()
        finally:
            for s, h in previous.items():
                signal.signal(s, h)

        exit_code = 128 - rc if rc < 0 else rc  # killed by signal N -> 128+N, as a shell reports it
        status = "stopped" if self.stop_requested else ("done" if rc == 0 else "failed")
        failure = None
        if status == "failed":
            for line in tail:
                print(f"| {line}", file=sys.stderr)
            sys.stderr.flush()
            failure = classify_failure(tail)
        # The child is gone, so nothing is mid-save: the final prune is safe.
        self.refresh_checkpoints(prune=True)
        _write_json(self.output / "result.json", self.result(status, exit_code, failure))
        self.write_status(status=status, failure=failure)
        _say(f"STATUS {status} exit={exit_code}")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
