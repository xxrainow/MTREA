#!/usr/bin/env bash
# GPU memory smoke test for pi05 fine-tuning — which configs fit on the A100
# (40GB), at what batch size, and how fast. Run by hand on the GPU host before
# committing to H0. Nothing produced here is a result: the dataset only exercises
# the pipeline and no checkpoint is written.
#
#     bash experiments/gpu_smoke_test.sh [--dataset=lerobot/svla_so101_pickplace] [--steps=20] [--out=data/runs/smoke]
#
# Written against lerobot v0.6.1 (src/lerobot/policies/pi05/configuration_pi05.py,
# configs/train.py, scripts/lerobot_train.py, scripts/lerobot_edit_dataset.py).
# Flags checked there, not taken on trust:
#   --policy.dtype                 defaults to float32, so bfloat16 is passed explicitly
#   --policy.train_expert_only     freezes all of PaliGemma, vision tower included
#   --policy.freeze_vision_encoder freezes only the SigLIP vision tower
#   --policy.gradient_checkpointing, --policy.push_to_hub (default true), --wandb.enable
#   --save_checkpoint=false        required: lerobot always saves at the final step,
#                                  so save_freq > steps alone still writes a checkpoint
#   --output_dir                   must not exist (FileExistsError), hence one dir per run
# Normalization: pi05 uses QUANTILES for state/action and needs q01/q99 in
# meta/stats.json. If the dataset lacks them, they are recomputed with
# `lerobot-edit-dataset --operation.type=recompute_stats` (the fix documented in
# docs/source/pi05.mdx) into a copy under --out. Normalization mode is not changed.
#
# Output: one `SMOKE key=value ...` line per run on stdout, rows appended to
# --out/results.csv, a final table. Exit 0 iff expert_only at bs=1 passed.

set -uo pipefail   # not -e: an OOM must not abort the remaining runs

cd "$(dirname "$0")/.."   # repo root, wherever this is run from

EXPECTED_LEROBOT=0.6.1
BASE_MODEL=lerobot/pi05_base
TOKENIZER_REPO=google/paligemma-3b-pt-224   # gated; see processor_pi05.py
SEED=1000
MIN_FREE_GIB=10

DATASET=lerobot/svla_so101_pickplace
STEPS=20
OUT=data/runs/smoke

for arg in "$@"; do
  case "$arg" in
    --dataset=*) DATASET="${arg#*=}" ;;
    --steps=*)   STEPS="${arg#*=}" ;;
    --out=*)     OUT="${arg#*=}" ;;
    -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (expected --dataset=, --steps=, --out=)" >&2; exit 2 ;;
  esac
done

die() { echo "FAIL: $*" >&2; exit 1; }

[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -ge 1 ]] || die "--steps must be a positive integer, got '$STEPS'"
case "$OUT" in ""|"/"|"."|"..") die "refusing --out='$OUT'" ;; esac

mkdir -p "$OUT" || die "cannot create $OUT"
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$OUT/$STAMP"
mkdir -p "$RUN_DIR"
RESULTS="$OUT/results.csv"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD 2>/dev/null || GIT_COMMIT="${GIT_COMMIT}-dirty"

# ---------------------------------------------------------------- preflight

echo "preflight: run dir $RUN_DIR (commit $GIT_COMMIT)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this the GPU host?"
# Sample the first visible GPU; lerobot-train uses the same one.
GPU="${CUDA_VISIBLE_DEVICES:-0}"; GPU="${GPU%%,*}"
GPU_INFO=$(nvidia-smi -i "$GPU" --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits) \
  || die "nvidia-smi failed for GPU $GPU"
IFS=',' read -r GPU_NAME GPU_TOTAL_MIB GPU_USED_MIB <<<"$GPU_INFO"
GPU_NAME="$(echo "$GPU_NAME" | xargs)"; GPU_TOTAL_MIB=${GPU_TOTAL_MIB// /}; GPU_USED_MIB=${GPU_USED_MIB// /}
echo "preflight: GPU $GPU = $GPU_NAME, total ${GPU_TOTAL_MIB} MiB, already used ${GPU_USED_MIB} MiB"
[ "$GPU_USED_MIB" -gt 1024 ] && echo "preflight: WARNING another process holds ${GPU_USED_MIB} MiB; peaks below include it (see baseline_mib)"

LEROBOT_VERSION=$(python -c 'import lerobot; print(lerobot.__version__)' 2>&1) \
  || die "import lerobot failed: $LEROBOT_VERSION"
echo "preflight: lerobot $LEROBOT_VERSION"
[ "$LEROBOT_VERSION" = "$EXPECTED_LEROBOT" ] \
  || echo "preflight: WARNING written against lerobot $EXPECTED_LEROBOT; re-check flags in the header against $LEROBOT_VERSION"
command -v lerobot-train >/dev/null || die "lerobot-train not on PATH"
command -v lerobot-edit-dataset >/dev/null || die "lerobot-edit-dataset not on PATH"

python - "$BASE_MODEL" "$TOKENIZER_REPO" <<'EOF' || exit 1
import sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

base, tok = sys.argv[1], sys.argv[2]
try:
    user = HfApi().whoami()["name"]
except Exception as e:
    sys.exit(f"FAIL: Hugging Face login missing or invalid ({e.__class__.__name__}). Run: hf auth login")
print(f"preflight: HF user {user}")
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(tok)
except (GatedRepoError, RepositoryNotFoundError, OSError) as e:
    sys.exit(
        f"FAIL: cannot download the {tok} tokenizer ({e.__class__.__name__}).\n"
        f"  pi05 uses this gated tokenizer. Logged in as '{user}', open\n"
        f"    https://huggingface.co/{tok}\n"
        f"  accept the license there, wait for access to be granted, and re-run."
    )
print(f"preflight: tokenizer {tok} OK")
try:
    path = snapshot_download(base)
except Exception as e:
    sys.exit(f"FAIL: cannot download {base} ({e.__class__.__name__}: {e})")
print(f"preflight: base model {base} OK at {path}")
EOF

FREE_MIB=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
echo "preflight: ${FREE_MIB} MiB free under $OUT"
[ "${FREE_MIB:-0}" -ge $((MIN_FREE_GIB * 1024)) ] || die "less than ${MIN_FREE_GIB} GiB free under $OUT"

# ---------------------------------------------------------------- dataset + quantile stats

# Kept under --out across invocations; the source copy is downloaded, never edited in place.
DS_SRC="$OUT/datasets/$(echo "$DATASET" | tr '/' '_')"
DS_Q="${DS_SRC}_q"

has_quantiles() {
  python - "$1/meta/stats.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except OSError:
    sys.exit(1)
ok = all({"q01", "q99"} <= set(s.get(k, {})) for k in ("action", "observation.state"))
sys.exit(0 if ok else 1)
EOF
}

echo "preflight: fetching $DATASET into $DS_SRC"
python -c 'import sys; from lerobot.datasets import LeRobotDataset; LeRobotDataset(sys.argv[1], root=sys.argv[2])' \
  "$DATASET" "$DS_SRC" >"$RUN_DIR/dataset_fetch.log" 2>&1 \
  || { tail -30 "$RUN_DIR/dataset_fetch.log" >&2; die "could not download $DATASET"; }

if has_quantiles "$DS_SRC"; then
  DS_ROOT="$DS_SRC"
  echo "preflight: dataset already has q01/q99 stats"
elif has_quantiles "$DS_Q"; then
  DS_ROOT="$DS_Q"
  echo "preflight: reusing recomputed quantile stats at $DS_Q"
else
  echo "preflight: dataset lacks q01/q99 (pi05 QUANTILES norm); recomputing into $DS_Q"
  lerobot-edit-dataset \
    --repo_id="$DATASET" \
    --root="$DS_SRC" \
    --new_root="$DS_Q" \
    --operation.type=recompute_stats \
    >"$RUN_DIR/recompute_stats.log" 2>&1 \
    || { tail -30 "$RUN_DIR/recompute_stats.log" >&2; die "recompute_stats failed"; }
  has_quantiles "$DS_Q" || die "recompute_stats ran but $DS_Q still lacks q01/q99"
  DS_ROOT="$DS_Q"
fi

# ---------------------------------------------------------------- runs

[ -f "$RESULTS" ] || echo "stamp,git_commit,lerobot,gpu,config,bs,status,peak_mib,baseline_mib,torch_peak_gib,s_per_step,s_per_step_src,wall_s,steps,dataset,log" >"$RESULTS"

SAMPLER_PID=""
stop_sampler() { [ -z "$SAMPLER_PID" ] || kill "$SAMPLER_PID" 2>/dev/null; }
trap stop_sampler EXIT
trap 'stop_sampler; echo "interrupted" >&2; exit 130' INT TERM

LAST_STATUS=""
EXPERT_BS1=""

# run_one NAME BS [extra lerobot-train flags...]
run_one() {
  local name=$1 bs=$2; shift 2
  local tag="${name}_bs${bs}"
  local log="$RUN_DIR/$tag.log" mem="$RUN_DIR/$tag.mem"

  local baseline
  baseline=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$mem" 2>/dev/null &
  SAMPLER_PID=$!

  local t0 t1 rc
  t0=$(date +%s)
  lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DS_ROOT" \
    --policy.type=pi05 \
    --policy.pretrained_path="$BASE_MODEL" \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --save_checkpoint=false \
    --save_freq=$((STEPS + 1)) \
    --log_freq=1 \
    --steps="$STEPS" \
    --batch_size="$bs" \
    --seed="$SEED" \
    --output_dir="$RUN_DIR/$tag" \
    --job_name="smoke_$tag" \
    "$@" \
    >"$log" 2>&1
  rc=$?
  t1=$(date +%s)

  kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""

  local status peak torch_peak sps src wall=$((t1 - t0))
  if grep -q "CUDA out of memory" "$log"; then
    status=OOM
  elif [ "$rc" -eq 0 ]; then
    status=OK
  else
    status=ERROR
    tail -30 "$log" >"$RUN_DIR/$tag.tail"
  fi

  peak=$(tr -d ' ' <"$mem" | grep -E '^[0-9]+$' | sort -n | tail -1)
  peak=${peak:-NA}

  # Per-step log lines: "step:N ... updt_s:X data_s:Y ... mem_gb:Z". Skip the first
  # 3 steps (CUDA warmup, allocator growth) when there are enough, take the median.
  sps=$(awk -v skip=$([ "$STEPS" -gt 5 ] && echo 3 || echo 0) '
    match($0, /updt_s:[0-9.]+/) {
      u = substr($0, RSTART + 7, RLENGTH - 7)
      d = 0; if (match($0, /data_s:[0-9.]+/)) d = substr($0, RSTART + 7, RLENGTH - 7)
      if (++n > skip) v[++m] = u + d
    }
    END {
      if (m == 0) exit
      for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
      printf "%.3f", (m % 2) ? v[(m + 1) / 2] : (v[m / 2] + v[m / 2 + 1]) / 2
    }' "$log")
  if [ -n "$sps" ]; then
    src=log
  elif [ "$status" = OK ]; then
    sps=$(awk -v w="$wall" -v s="$STEPS" 'BEGIN { printf "%.3f", w / s }')
    src=wall   # includes model load — an overestimate
  else
    sps=NA; src=NA
  fi

  torch_peak=$(grep -oE 'mem_gb:[0-9.]+' "$log" | cut -d: -f2 | sort -n | tail -1)
  torch_peak=${torch_peak:-NA}

  echo "SMOKE config=$name bs=$bs status=$status peak_mib=$peak baseline_mib=$baseline torch_peak_gib=$torch_peak s_per_step=$sps s_per_step_src=$src wall_s=$wall log=$log"
  [ "$status" = ERROR ] && sed 's/^/    | /' "$RUN_DIR/$tag.tail" >&2
  echo "$STAMP,$GIT_COMMIT,$LEROBOT_VERSION,$GPU_NAME,$name,$bs,$status,$peak,$baseline,$torch_peak,$sps,$src,$wall,$STEPS,$DATASET,$log" >>"$RESULTS"

  LAST_STATUS=$status
}

# 1. H0's config: action expert only. Increase batch size until the first OOM.
for bs in 1 2 4 8; do
  run_one expert_only "$bs" --policy.train_expert_only=true --policy.freeze_vision_encoder=false
  [ "$bs" = 1 ] && EXPERT_BS1=$LAST_STATUS
  [ "$LAST_STATUS" = OOM ] && break
done

# 2. VLM language model + expert trained, vision tower frozen.
run_one freeze_vision 1 --policy.train_expert_only=false --policy.freeze_vision_encoder=true

# 3. Full fine-tuning.
run_one full 1 --policy.train_expert_only=false --policy.freeze_vision_encoder=false

# ---------------------------------------------------------------- summary

echo
echo "=== smoke results ($STAMP, $GPU_NAME, ${GPU_TOTAL_MIB} MiB, $STEPS steps) ==="
{
  echo "config bs status peak_mib torch_peak_gib s_per_step src"
  awk -F, -v s="$STAMP" '$1 == s { print $5, $6, $7, $8, $10, $11, $12 }' "$RESULTS"
} | column -t
echo "full rows: $RESULTS"

if [ "$EXPERT_BS1" = OK ]; then
  exit 0
fi
echo "FAIL: expert_only at bs=1 did not pass ($EXPERT_BS1); H0 does not fit as configured" >&2
exit 1
