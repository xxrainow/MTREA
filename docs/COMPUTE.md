# Compute: what fits on the GPU

Measured 2026-10-22 on the project's server with `experiments/gpu_smoke_test.sh`
at commit `052b3c3`. The full fine-tuning runs at bs 8/16/32 and the 300-step
run used the same lerobot flags but are outside that commit's run matrix; the
script now includes full fine-tuning at bs 1/8/16.

- GPU: one A100-80GB MIG slice, 3g.40gb
- lerobot v0.6.1, Python 3.12
- `lerobot/pi05_base` on `lerobot/svla_so101_pickplace`
- bf16 (`--policy.dtype=bfloat16`), `--policy.gradient_checkpointing=true`

`mem_gb` is lerobot's `torch.cuda.max_memory_allocated()` in GiB; `updt_s` is
seconds per optimizer step.

| config | flags | bs | mem_gb | updt_s | samples/s |
|---|---|---|---|---|---|
| expert-only | `train_expert_only=true` | 1 | 12.86 | 0.31 | 3.2 |
| | | 2 | 12.87 | 0.48 | 4.2 |
| | | 4 | 12.89 | 0.83 | 4.8 |
| | | 8 | 13.33 | 1.55 | 5.2 |
| vision frozen | `freeze_vision_encoder=true` | 1 | 30.80 | 0.53 | 1.9 |
| full fine-tuning | both `false` (lerobot defaults) | 1 | 36.96 | 0.66 | 1.5 |
| | | 8 | 37.06 | 2.17 | 3.7 |
| | | 16 | 37.13 | 3.92 | 4.1 |
| | | 32 | OOM at step 2 | | |

Full fine-tuning at bs 16 for 300 steps stayed at 37.09 GiB in every log line.

Other facts from the same runs:

- On MIG, an OOM does not print "CUDA out of memory". It surfaces as
  `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp`,
  because NVML queries are not permitted in the container.
- `nvidia-smi` memory queries return `[InsufficientPermissions]`.
- lerobot v0.6.1 has no gradient accumulation, so the batch size per step is
  the effective batch size.

What this implies:

- Full fine-tuning fits at bs 16 with about 1 GB of headroom, and bs 32 does not.
- bs 16 is the throughput optimum. The slice is compute-bound: full fine-tuning
  saturates around 4 samples/s and expert-only around 5.
- At large batch, expert-only is about 20% faster than full fine-tuning
  (5.2 vs 4.1 samples/s). It is not the memory saver it is on a full GPU.
- Measure memory with `mem_gb` (torch_peak_gib in the smoke script), not
  `nvidia-smi`, on MIG. Classify OOM by either error string.
