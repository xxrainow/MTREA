#!/usr/bin/env bash
# Terminal recording test for SO-101 — one short episode into data/raw/.
#
#     bash experiments/record_test.sh
#
# Ports, calibration ids and camera indices come from robot/configs/; edit
# so101_follower.yaml, so101_leader.yaml and cameras.yaml, not this script.
# Find ports:    lerobot-find-port        (unplug/replug one arm at a time)
# Find cameras:  lerobot-find-cameras opencv
# Set the `id` in each yaml to whatever calibration already exists under:
#     ls ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/
#     ls ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, wherever this is run from

# cfg FILE KEY [KEY...] — print a nested value from a yaml file.
cfg() {
  python -c 'import sys, functools, yaml; print(functools.reduce(lambda d, k: d[k], sys.argv[2:], yaml.safe_load(open(sys.argv[1]))))' "$@"
}

FOLLOWER_PORT=$(cfg robot/configs/so101_follower.yaml port)
FOLLOWER_ID=$(cfg robot/configs/so101_follower.yaml id)
LEADER_PORT=$(cfg robot/configs/so101_leader.yaml port)
LEADER_ID=$(cfg robot/configs/so101_leader.yaml id)
WRIST_CAM=$(cfg robot/configs/cameras.yaml wrist index_or_path)
SCENE_CAM=$(cfg robot/configs/cameras.yaml scene index_or_path)

TASK="pick up the red block and place it in the blue cup"
OUT="data/raw/pick_place_cli_test"

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
