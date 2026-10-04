"""Training entrypoint: a thin bookkeeping wrapper around LeRobot's `lerobot-train`.

    python -m research.train --method=h0 --dataset=<repo_id> --output=data/runs/<run_id> --seed=0

Training logic lives in lerobot. This module only makes a run reproducible:
it refuses to overwrite a past run or to start on a dataset without the
quantile stats pi05 needs, writes config.json (every resolved value,
the exact lerobot argv, git commit, versions) before launching, streams the
child's output to train.log, parses metrics into metrics.jsonl, and records
how the run ended in result.json.

stdout carries only parseable lines for lab's runner:
    STATUS started
    PROGRESS step=<n>/<total> loss=<x> lr=<y>
    STATUS done|failed|stopped exit=<code>

lerobot flag names are the ones verified against lerobot v0.6.1 in
experiments/gpu_smoke_test.sh, plus policy.optimizer_lr, log_freq and the
MetricsTracker log-line format read from the same v0.6.1 source.
TODO(GPU host): re-check against the installed lerobot if it is not v0.6.1:
policy.type, policy.pretrained_path, policy.train_expert_only,
policy.freeze_vision_encoder, policy.gradient_checkpointing, policy.dtype,
policy.push_to_hub, policy.optimizer_lr, dataset.repo_id, dataset.root,
output_dir, job_name, seed, steps, batch_size, save_freq, log_freq,
wandb.enable, and the MetricsTracker log-line format.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Method name -> lerobot policy flags. H1+ are added here. An oracle run is the
# h0 recipe on the held-out task's own dataset, not a separate method.
METHODS: dict[str, dict[str, str]] = {
    "h0": {
        # train_expert_only freezes all of PaliGemma, vision tower included, so
        # freeze_vision_encoder is redundant; false matches the smoke test.
        "policy.train_expert_only": "true",
        "policy.freeze_vision_encoder": "false",
    },
}

# pi05 normalizes these with QUANTILES; lerobot fails on the first batch without q01/q99.
QUANTILE_FEATURES = ("observation.state", "action")

# One lerobot training log line, from MetricsTracker.__str__ via logging.info:
#   INFO 2026-10-04 12:00:00 ot_train.py:641 step:2K smpl:2K ep:3 epch:0.13 loss:0.512 grdn:3.210 lr:2.5e-05 updt_s:0.812 ...
# search(), not match(): tqdm's \r-redrawn bar can share a line with the log record.
METRICS_LINE = re.compile(
    r"\bstep:(?P<step>\d+[KMBTQ]?)\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+"
    r"loss:(?P<loss>\S+)\s+grdn:\S+\s+lr:(?P<lr>\S+)"
)

_SUFFIXES = ["", "K", "M", "B", "T", "Q"]


def parse_metrics_line(line: str) -> Optional[dict]:
    """Return {"step": token, "loss": float, "lr": float} for a training log line, else None.

    `step` stays the raw token ("200", "2K"); see resolve_step.
    """
    m = METRICS_LINE.search(line)
    if m is None:
        return None
    try:
        return {"step": m["step"], "loss": float(m["loss"]), "lr": float(m["lr"])}
    except ValueError:
        return None


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.train", description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--dataset", required=True, help="LeRobot dataset repo_id")
    p.add_argument("--output", required=True, help="run directory, e.g. data/runs/h0_pickplace_seed0")
    p.add_argument("--seed", required=True, type=int, help="explicit; never defaulted")
    p.add_argument("--dataset-root", default=None, help="local dataset path, if not on the Hub")
    p.add_argument("--base", default="lerobot/pi05_base")
    p.add_argument("--steps", type=_positive_int, default=30000)
    p.add_argument("--batch-size", type=_positive_int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save-freq", type=_positive_int, default=5000)
    p.add_argument("--log-freq", type=_positive_int, default=200,
                   help="passed to lerobot; also needed to recover exact steps from its log")
    p.add_argument("--wandb", type=_bool, default=False)
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
        "policy.type": "pi05",
        "policy.pretrained_path": args.base,
        **METHODS[args.method],
        "policy.gradient_checkpointing": "true",
        "policy.dtype": "bfloat16",  # pi05 defaults to float32
        "policy.optimizer_lr": repr(args.lr),  # the policy preset overrides --optimizer.lr
        "policy.push_to_hub": "false",  # defaults to true
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
    }
    return [*split_command(args.lerobot_bin), *(f"--{k}={v}" for k, v in flags.items())]


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
        "args": {k: v for k, v in vars(args).items() if k != "dry_run"},
        "method_flags": METHODS[args.method],
        "lerobot_argv": argv,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "lerobot_version": _lerobot_version(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "start_time": _now(),
    }


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # never leave a half-written result behind


def _say(line: str) -> None:
    print(line, flush=True)


# --- run ---

def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)

    if output.exists() and (output.is_file() or any(output.iterdir())):
        print(f"error: --output {output} already exists and is not empty; "
              "past runs are never overwritten, pick a new run directory", file=sys.stderr)
        return 2

    # Fatal in a real run; in a dry run only reported, so a preview works offline.
    try:
        missing = missing_quantile_stats(load_stats(args.dataset, args.dataset_root))
    except StatsError as e:
        if not args.dry_run:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"STATS unchecked: {e}")
    else:
        if missing and not args.dry_run:
            print(f"error: {args.dataset} meta/stats.json lacks q01/q99 for {missing}; pi05 normalizes "
                  f"state/action with QUANTILES and would fail on the first batch. "
                  + recompute_hint(args.dataset, args.dataset_root), file=sys.stderr)
            return 2
        if args.dry_run and missing:
            print(f"STATS missing q01/q99 for {missing} — run "
                  + recompute_hint(args.dataset, args.dataset_root, one_line=True))
        elif args.dry_run:
            print("STATS ok")

    lerobot_argv = build_lerobot_argv(args, output)
    config = build_config(args, lerobot_argv)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        print("COMMAND " + shlex.join(lerobot_argv))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    return _run(lerobot_argv, output, args.steps, args.log_freq)


def _run(lerobot_argv: list[str], output: Path, total_steps: int, log_freq: int) -> int:
    proc: Optional[subprocess.Popen] = None
    stop_requested = False
    last: dict = {"step": None, "loss": None}

    def result(status: str, exit_code: Optional[int]) -> dict:
        return {"status": status, "exit_code": exit_code, "end_time": _now(),
                "last_step": last["step"], "last_loss": last["loss"]}

    def on_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        # Record the stop now: lab's runner SIGKILLs us a few seconds after SIGTERM.
        _write_json(output / "result.json", result("stopped", None))
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                proc.terminate()  # Windows Popen cannot deliver SIGINT/SIGTERM
            else:
                proc.send_signal(signum)

    handled = [signal.SIGINT, signal.SIGTERM]
    previous = {s: signal.signal(s, on_signal) for s in handled}
    try:
        _say("STATUS started")
        try:
            proc = subprocess.Popen(lerobot_argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        except OSError as e:
            print(f"error: cannot start {lerobot_argv[0]!r}: {e}", file=sys.stderr)
            _write_json(output / "result.json", result("failed", 127))
            _say("STATUS failed exit=127")
            return 127

        with open(output / "train.log", "wb") as log, open(output / "metrics.jsonl", "a", encoding="utf-8") as metrics:
            prev_step = 0
            for raw in proc.stdout:
                log.write(raw)
                log.flush()
                parsed = parse_metrics_line(raw.decode("utf-8", errors="replace"))
                if parsed is None:
                    continue
                step, exact = resolve_step(parsed["step"], prev_step, log_freq)
                prev_step = step
                last.update(step=step, loss=parsed["loss"])
                record = {"step": step, "step_exact": exact, "step_logged": parsed["step"],
                          "loss": parsed["loss"], "lr": parsed["lr"], "time": _now()}
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                _say(f"PROGRESS step={step}/{total_steps} loss={parsed['loss']} lr={parsed['lr']}")
        rc = proc.wait()
    finally:
        for s, h in previous.items():
            signal.signal(s, h)

    exit_code = 128 - rc if rc < 0 else rc  # killed by signal N -> 128+N, as a shell reports it
    status = "stopped" if stop_requested else ("done" if rc == 0 else "failed")
    _write_json(output / "result.json", result(status, exit_code))
    _say(f"STATUS {status} exit={exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
