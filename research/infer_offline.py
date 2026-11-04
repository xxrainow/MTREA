"""Offline check: does a trained checkpoint reproduce its own demos?

    python -m research.infer_offline --checkpoint=<pretrained_model dir or hub id> \\
        --dataset-root=<dir> --samples=50 --seed=0 --out=<report.json> [--compare=lerobot/pi05_base]

Each sampled frame goes through the same code the policy server runs
(serve_backend.PolicyBackend._predict_action_chunk, starting from a raw robot
observation), and the chunk is compared with the dataset's recorded actions.

--compare loads other weights behind the checkpoint's own config and processors.
That is the only comparison that means anything for lerobot/pi05_base: it has
no SO-101 features or stats of its own, and this is exactly the state training
started from. Noise is reseeded per sample, so both models see the same noise.

Exit codes: 0 done (and, with --compare, the checkpoint is clearly better),
1 failed or not clearly better, 2 refused.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from research.serve_common import (DEFAULT_CHUNK_SIZE, UNITS_NOTE, CheckpointError, ChunkErrors, DatasetError,
                                   DemoDataset, checkpoint_provenance, gpu_busy_reason, latency_summary,
                                   lerobot_log_dir, resolve_checkpoint, run_provenance, served_policy_from_config,
                                   validate_checkpoint_dir)
from research.train import _write_json

WORST_SAMPLES = 10
# The verdict for --compare: the checkpoint's mean MAE must be under this share of
# the reference's at every horizon. These are absolute errors, reported in full; the
# factor only decides the exit code.
CLEARLY_BETTER_FACTOR = 0.5
SEED_NOTE = ("Flow matching samples noise. torch is reseeded per sample from --seed, so a run is repeatable on "
             "one machine; results vary slightly across seeds.")


def sample_frames(dataset: DemoDataset, samples: int, seed: int, horizon: int) -> list[tuple[int, int]]:
    """(episode, frame) pairs spread across episodes, with `horizon` recorded actions ahead of each."""
    import numpy as np

    eligible = dataset.eligible_frames(horizon)
    if not eligible:
        raise DatasetError(f"no episode of {dataset.root} has {horizon} frames")
    rng = np.random.default_rng(seed)
    episodes = list(eligible)
    rng.shuffle(episodes)
    picked: set[tuple[int, int]] = set()
    available = sum(len(r) for r in eligible.values())
    i = 0
    while len(picked) < min(samples, available):
        episode = episodes[i % len(episodes)]
        picked.add((int(episode), int(rng.integers(len(eligible[episode])))))
        i += 1
    return sorted(picked)


def evaluate(backend, dataset: DemoDataset, frames: list[tuple[int, int]], images: dict, seed: int,
             chunk_size: int) -> dict:
    """One model on all the samples: errors, the worst samples, timings and peak GPU memory."""
    import torch

    from lerobot.async_inference.helpers import TimedObservation

    errors = ChunkErrors(dataset.joint_names)
    per_sample = []
    timings: dict[str, list[float]] = {"prepare": [], "preprocess": [], "inference": [], "postprocess": [], "total": []}
    for i, (episode, frame) in enumerate(frames):
        start, _ = dataset.episode_range(episode)
        task = dataset.task_of(start + frame)
        raw = dataset.raw_observation(episode, frame, images[episode, frame], task)
        torch.manual_seed(seed + i)
        chunk = backend._predict_action_chunk(TimedObservation(time.time(), frame, raw))
        predicted = [a.get_action().tolist() for a in chunk]
        recorded = dataset.actions_from(episode, frame, chunk_size).tolist()
        errors.add(predicted, recorded)
        one = ChunkErrors(dataset.joint_names)
        one.add(predicted, recorded)
        summary = one.summary()
        worst_joint = max(summary["first_50"]["mae"], key=summary["first_50"]["mae"].get)
        per_sample.append({"episode": episode, "frame": frame, "task": task,
                           "mae_first_1": summary["first_1"]["mean_mae"], "mae_first_10": summary["first_10"]["mean_mae"],
                           "mae_first_50": summary["first_50"]["mean_mae"], "worst_joint": worst_joint,
                           "worst_joint_mae": summary["first_50"]["mae"][worst_joint]})
        for name, values in timings.items():
            values.append(backend.last_timing[f"{name}_ms"])
    per_sample.sort(key=lambda s: s["mae_first_50"], reverse=True)
    out = {"errors": errors.summary(), "worst_samples": per_sample[:WORST_SAMPLES],
           "timing_ms": {name: latency_summary(values) for name, values in timings.items()}}
    if torch.cuda.is_available():
        out["peak_gpu_gib"] = {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                               "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    return out


def run_model(backend_module, spec, dataset, frames, images, args, weights: Optional[str]) -> dict:
    import gc

    import torch

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    backend = backend_module.build_backend(spec, Path(spec.checkpoint), args.device, args.fps, weights=weights)
    backend.warmup()  # excluded from the timings below
    result = evaluate(backend, dataset, frames, images, args.seed, spec.chunk_size)
    del backend
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def clearly_better(checkpoint: dict, reference: dict) -> bool:
    for horizon, errors in checkpoint.items():
        if errors["mean_mae"] >= CLEARLY_BETTER_FACTOR * reference[horizon]["mean_mae"]:
            return False
    return True


def print_summary(report: dict) -> None:
    def line(name: str, result: dict) -> None:
        errors = result["errors"]
        print(f"{name}: mean MAE first_1={errors['first_1']['mean_mae']} first_10={errors['first_10']['mean_mae']} "
              f"first_50={errors['first_50']['mean_mae']} ({UNITS_NOTE})")
        print(f"  inference ms {result['timing_ms']['inference']}; peak GPU GiB {result.get('peak_gpu_gib')}")

    print(f"checkpoint step={report['provenance']['step']} run={report['provenance']['run_id']} "
          f"samples={len(report['samples'])} seed={report['seed']}")
    line("checkpoint", report["checkpoint_result"])
    if "compare_result" in report:
        line(report["compare"], report["compare_result"])
        print(f"clearly better: {report['clearly_better']}")
    print(f"report: {report['out']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.infer_offline", description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="pretrained_model directory or Hub id")
    p.add_argument("--dataset-root", required=True, help="the dataset directory to compare with")
    p.add_argument("--samples", type=int, default=50, help="frames to sample across episodes")
    p.add_argument("--seed", type=int, required=True, help="explicit; never defaulted")
    p.add_argument("--out", required=True, help="report JSON, written atomically")
    p.add_argument("--compare", default=None, help="other weights to run behind the same config, e.g. lerobot/pi05_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=None, help="default: the dataset's fps")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint_dir = resolve_checkpoint(args.checkpoint)
        config = validate_checkpoint_dir(checkpoint_dir)
        dataset = DemoDataset(Path(args.dataset_root))
    except (CheckpointError, DatasetError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.fps = args.fps or dataset.fps
    chunk_size = int(config.get("chunk_size", DEFAULT_CHUNK_SIZE))
    frames = sample_frames(dataset, args.samples, args.seed, chunk_size)
    # Decoded once so every model sees identical pixels.
    images = {(e, f): dataset.frame_images(e, f) for e, f in frames}

    import torch

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("error: CUDA is not available; use --device=cpu", file=sys.stderr)
            return 2
        reason = gpu_busy_reason(*torch.cuda.mem_get_info())
        if reason:
            print(f"error: {reason}", file=sys.stderr)
            return 2

    with lerobot_log_dir():
        from research import serve_backend

    spec = served_policy_from_config(str(checkpoint_dir), config)
    report = {
        "checkpoint": args.checkpoint, "dataset_root": args.dataset_root, "seed": args.seed, "device": args.device,
        "seed_note": SEED_NOTE, "units": UNITS_NOTE, "joint_names": dataset.joint_names,
        "provenance": checkpoint_provenance(checkpoint_dir), "run": run_provenance(), "command": sys.argv,
        "samples": [{"episode": e, "frame": f} for e, f in frames],
        "checkpoint_result": run_model(serve_backend, spec, dataset, frames, images, args, weights=None),
        "out": args.out,
    }
    if args.compare:
        report["compare"] = args.compare
        report["compare_note"] = (f"{args.compare} weights behind this checkpoint's config and processors: "
                                  "the state fine-tuning started from")
        report["compare_result"] = run_model(serve_backend, spec, dataset, frames, images, args, weights=args.compare)
        report["clearly_better"] = clearly_better(report["checkpoint_result"]["errors"],
                                                  report["compare_result"]["errors"])
        report["clearly_better_rule"] = (f"mean MAE below {CLEARLY_BETTER_FACTOR} x the reference's "
                                         "at first_1, first_10 and first_50")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, report)
    print_summary(report)
    return 1 if args.compare and not report["clearly_better"] else 0


if __name__ == "__main__":
    sys.exit(main())
