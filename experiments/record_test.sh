#!/usr/bin/env bash
# Terminal recording test for SO-101 — one short episode into data/raw/.
#
# Fill in the four values below, then
#     bash experiments/record_test.sh
# 
# Find ports:    lerobot-find-port        (unplug/replug one arm at a time)
# Find cameras:  lerobot-find-cameras opencv
# Calibration ids must match what LeLab used:
#     ls ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/
#     ls ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/

set -euo pipefail

# ---- need to fill ------------------------------------------------
FOLLOWER_PORT="/dev/tty.usbmodemXXXX"
LEADER_PORT="/dev/tty.usbmodemYYYY"
WRIST_CAM=0
SCENE_CAM=1
# ------------------------------------------------------------------------------

FOLLOWER_ID="so-arm1"
LEADER_ID="so-arm1"
TASK="pick up the red block and place it in the blue cup"
OUT="data/raw/pick_place_cli_test"

cd "$(dirname "$0")/.."   # repo root, wherever this is run from

lerobot-record \
  --robot.type so101_follower \
  --robot.port "$FOLLOWER_PORT" \
  --robot.id "$FOLLOWER_ID" \
  --robot.cameras "{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": $WRIST_CAM, \"width\": 640, \"height\": 480, \"fps\": 30}, \"scene\": {\"type\": \"opencv\", \"index_or_path\": $SCENE_CAM, \"width\": 640, \"height\": 480, \"fps\": 30}}" \
  --teleop.type so101_leader \
  --teleop.port "$LEADER_PORT" \
  --teleop.id "$LEADER_ID" \
  --dataset.repo_id local/pick_place_cli_test \
  --dataset.root "$OUT" \
  --dataset.single_task "$TASK" \
  --dataset.num_episodes 1 \
  --dataset.episode_time_s 20 \
  --dataset.reset_time_s 5 \
  --dataset.push_to_hub false

echo
echo "recorded to $OUT — now run:"
echo "  python experiments/inspect_dataset.py $OUT"
