"""Stand-in for lerobot v0.6.1's `lerobot-train`: no GPU, no lerobot, synthetic losses.

    python research/fake_lerobot_train.py --output_dir=<dir> --steps=<n> ... (lerobot's --key=value flags)

research.train runs this under --fake, with the same flags it gives lerobot,
so lab's Train page and the run-directory bookkeeping can be exercised end to
end without the GPU host. What it writes follows the v0.6.1 source (git tag
v0.6.1), not memory:

- MetricsTracker log lines on stderr (utils/logging_utils.py, meters from
  scripts/lerobot_train.py), steps/samples/episodes via format_big_number.
- "Checkpoint policy after step N" before each save, at every save_freq and
  at the final step (common/train_utils.py should_save_checkpoint).
- checkpoints/<step zero-padded to max(6, len(str(steps)))>/{pretrained_model,
  training_state}, then `last` re-pointed by unlink + symlink_to(<step name>),
  as update_last_checkpoint does. Weight files hold placeholder bytes.
- FileExistsError for an existing output_dir (configs/train.py validate).
- lr follows CosineDecayWithWarmupSchedulerConfig, read after scheduler.step().
- With wandb.enable and wandb.project: WandBLogger's "Track this run --> <url>"
  in termcolor codes. The url is under the reserved .invalid domain.

Environment: MTREA_FAKE_SECONDS_PER_STEP (default 0.01) is the wall time per
step; MTREA_FAKE_FAIL=oom fails at step 2 with the MIG NVML assert (as bs 32
did, docs/COMPUTE.md), =error fails right after the halfway step.

Standalone on purpose: it imports nothing from research/ or lerobot.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

NUM_FRAMES = 20_000
NUM_EPISODES = 50
MEM_GB = {"true": 13.33, "false": 37.13}  # by policy.train_expert_only, docs/COMPUTE.md

OOM_TRACE = [
    "Traceback (most recent call last):",
    '  File "/lerobot/src/lerobot/scripts/lerobot_train.py", line 609, in train',
    "    train_tracker, _ = update_policy(",
    "RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "
    '"/pytorch/c10/cuda/CUDACachingAllocator.cpp":1131, please report a bug to PyTorch.',
]
ERROR_TRACE = [
    "Traceback (most recent call last):",
    '  File "/lerobot/src/lerobot/scripts/lerobot_train.py", line 609, in train',
    "RuntimeError: fake lerobot-train failure (MTREA_FAKE_FAIL=error)",
]


def format_big_number(num, precision=0):
    # lerobot.utils.utils.format_big_number, verbatim.
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0
    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor
    return num


def log(message: str, source: str = "lerobot_train.py:641") -> None:
    # init_logging's format: "%(levelname)s %(asctime)s %(lerobot_location)15s %(message)s",
    # where lerobot_location is the last 15 characters of "<path>:<lineno>".
    print(f"INFO {datetime.now():%Y-%m-%d %H:%M:%S} {source[-15:]:>15} {message}", file=sys.stderr, flush=True)


def colored_bold_yellow(text: str) -> str:
    return f"\x1b[1m\x1b[33m{text}\x1b[0m"


def parse_flags(argv: list[str]) -> dict[str, str]:
    flags = {}
    for arg in argv:
        if not arg.startswith("--") or "=" not in arg:
            raise SystemExit(f"fake lerobot-train: expected --key=value, got {arg!r}")
        key, value = arg[2:].split("=", 1)
        flags[key] = value
    return flags


def lr_lambda(step: int, warmup: int, decay_steps: int, total: int, peak: float, floor: float) -> float:
    # CosineDecayWithWarmupSchedulerConfig.build(), including its auto-scaling.
    if total < decay_steps:
        warmup = int(warmup * total / decay_steps)
        decay_steps = total
    if step < warmup:
        if step <= 0:
            return 1 / (warmup + 1)
        frac = 1 - step / warmup
        return (1 / (warmup + 1) - 1) * frac + 1
    s = min(step, decay_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * s / decay_steps))
    alpha = floor / peak
    return (1 - alpha) * cosine + alpha


class Meter:
    """AverageMeter: logs the window average as "{name}:{avg<fmt>}"."""

    def __init__(self, name: str, fmt: str):
        self.name, self.fmt = name, fmt
        self.sum = self.count = 0.0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def reset(self) -> None:
        self.sum = self.count = 0.0

    def __str__(self) -> str:
        return f"{self.name}:{format(self.avg, self.fmt)}"


def save_checkpoint(checkpoint_dir: Path, step: int, flags: dict, batch_size: int) -> None:
    pretrained = checkpoint_dir / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text(
        json.dumps({"type": flags.get("policy.type", "pi05"), "fake": True}, indent=2), encoding="utf-8")
    (pretrained / "model.safetensors").write_bytes(b"fake weights\n")
    (pretrained / "train_config.json").write_text(json.dumps({**flags, "fake": True}, indent=2), encoding="utf-8")
    state = checkpoint_dir / "training_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "training_step.json").write_text(
        json.dumps({"step": step, "num_processes": 1, "batch_size": batch_size}), encoding="utf-8")
    (state / "rng_state.safetensors").write_bytes(b"fake rng state\n")
    (state / "optimizer_state.safetensors").write_bytes(b"fake optimizer state\n")
    (state / "optimizer_param_groups.json").write_text("[]", encoding="utf-8")
    (state / "scheduler_state.json").write_text(json.dumps({"last_epoch": step}), encoding="utf-8")


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    # common/train_utils.py update_last_checkpoint, verbatim.
    last_checkpoint_dir = checkpoint_dir.parent / "last"
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def main(argv: list[str] | None = None) -> int:
    flags = parse_flags(sys.argv[1:] if argv is None else argv)
    output_dir = Path(flags["output_dir"])
    if output_dir.is_dir():
        raise FileExistsError(
            f"Output directory {output_dir} already exists and resume is False. "
            f"Please change your output directory so that {output_dir} is not overwritten.")

    steps = int(flags["steps"])
    batch_size = int(flags.get("batch_size", "8"))
    save_freq = int(flags.get("save_freq", "20000"))
    log_freq = int(flags.get("log_freq", "200"))
    seed = int(flags.get("seed", "1000"))
    peak = float(flags.get("policy.optimizer_lr", "2.5e-5"))
    floor = float(flags.get("policy.scheduler_decay_lr", "2.5e-6"))
    warmup = int(flags.get("policy.scheduler_warmup_steps", "1000"))
    decay_steps = int(flags.get("policy.scheduler_decay_steps", "30000"))
    seconds_per_step = float(os.environ.get("MTREA_FAKE_SECONDS_PER_STEP", "0.01"))
    fail = os.environ.get("MTREA_FAKE_FAIL", "none")
    fail_step = {"oom": 2, "error": steps // 2 + 1}.get(fail)

    output_dir.mkdir(parents=True)
    log(f"Output dir: {output_dir}", "lerobot_train.py:434")
    if flags.get("wandb.enable") == "true" and flags.get("wandb.project"):
        url = f"https://wandb.invalid/fake/{flags['wandb.project']}/runs/fake-{seed}"
        log("Logs will be synced with wandb.", "wandb_utils.py:121")
        log(f"Track this run --> {colored_bold_yellow(url)}", "wandb_utils.py:122")
    else:
        log("Logs will be saved locally.", "lerobot_train.py:280")
    log(f"cfg.steps={steps} ({format_big_number(steps)})", "lerobot_train.py:441")

    rng = random.Random(seed)
    meters = {
        "loss": Meter("loss", ".3f"), "grad_norm": Meter("grdn", ".3f"), "lr": Meter("lr", "0.1e"),
        "update_s": Meter("updt_s", ".3f"), "dataloading_s": Meter("data_s", ".3f"),
        "samples_per_s": Meter("smp/s", ".0f"),
        "gpu_mem_gb": Meter("mem_gb", ".2f"),
    }
    mem_gb = MEM_GB.get(flags.get("policy.train_expert_only", "false"), 37.13)
    samples = 0
    for step in range(1, steps + 1):
        start = time.perf_counter()
        meters["dataloading_s"].update(time.perf_counter() - start)
        start = time.perf_counter()
        time.sleep(seconds_per_step)
        if step == fail_step:
            for line in OOM_TRACE if fail == "oom" else ERROR_TRACE:
                print(line, file=sys.stderr, flush=True)
            return 1
        progress = step / steps
        meters["loss"].update(max(0.01, 0.06 + 0.9 * math.exp(-5 * progress) + rng.gauss(0, 0.02)))
        meters["grad_norm"].update(abs(1.0 + 4 * math.exp(-5 * progress) + rng.gauss(0, 0.2)))
        meters["lr"].update(peak * lr_lambda(step, warmup, decay_steps, steps, peak, floor))
        meters["update_s"].update(time.perf_counter() - start)
        meters["gpu_mem_gb"].update(mem_gb)
        samples += batch_size

        if log_freq > 0 and step % log_freq == 0:
            step_time = meters["update_s"].avg + meters["dataloading_s"].avg
            if step_time > 0:
                meters["samples_per_s"].update(batch_size / step_time)
            line = " ".join([
                f"step:{format_big_number(step)}",
                f"smpl:{format_big_number(samples)}",
                f"ep:{format_big_number(samples / (NUM_FRAMES / NUM_EPISODES))}",
                f"epch:{samples / NUM_FRAMES:.2f}",
                *[str(m) for m in meters.values()],
            ])
            log(line)
            for m in meters.values():
                m.reset()

        if (save_freq > 0 and step % save_freq == 0) or step == steps:
            log(f"Checkpoint policy after step {step}", "lerobot_train.py:687")
            checkpoint_dir = output_dir / "checkpoints" / f"{step:0{max(6, len(str(steps)))}d}"
            save_checkpoint(checkpoint_dir, step, flags, batch_size)
            update_last_checkpoint(checkpoint_dir)

    log("End of training", "lerobot_train.py:770")
    return 0


if __name__ == "__main__":
    sys.exit(main())
