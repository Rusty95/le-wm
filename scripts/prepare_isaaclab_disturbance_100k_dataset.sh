#!/usr/bin/env bash
# Prepare a continuously collected PPO recovery dataset with random pole impulses.

set -euo pipefail

TASK="${TASK:-RLLab-Cartpole-SwingUp-Direct-v0}"
TARGET_FRAMES="${TARGET_FRAMES:-100000}"
EPISODE_LEN="${EPISODE_LEN:-600}"
DISTURBANCE_MIN="${DISTURBANCE_MIN:-2.4}"
DISTURBANCE_MAX="${DISTURBANCE_MAX:-6.0}"
NPZ_DIR="${NPZ_DIR:-/home/hall/code/.stable-wm/isaaclab_policy_disturbance_npz_100k}"
H5_PATH="${H5_PATH:-/home/hall/code/.stable-wm/datasets/isaaclab_policy_disturbance_100k.h5}"
VIS_DIR="${VIS_DIR:-/home/hall/code/.stable-wm/visualizations}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole_swingup_centered/2026-06-30_01-24-53_reward_v2/model_200.pt}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

print_config() {
  cat <<EOF
[INFO] IsaacLab continuous-disturbance dataset
  task:              $TASK
  target_frames:     $TARGET_FRAMES
  max_chunk_len:     $EPISODE_LEN
  disturbance_range: [$DISTURBANCE_MIN, $DISTURBANCE_MAX] rad/s
  policy:            $POLICY_CHECKPOINT
  npz_dir:           $NPZ_DIR
  h5_path:           $H5_PATH
EOF
}

print_collect_command() {
  cat <<EOF
python "$REPO_ROOT/scripts/collect_isaaclab_policy_npz.py" \\
  --task "$TASK" \\
  --policy-checkpoint "$POLICY_CHECKPOINT" \\
  --target-frames "$TARGET_FRAMES" \\
  --episode-len "$EPISODE_LEN" \\
  --output-dir "$NPZ_DIR" \\
  --continuous-disturbance \\
  --disturbance-min "$DISTURBANCE_MIN" \\
  --disturbance-max "$DISTURBANCE_MAX" \\
  --disturbance-stable-steps 60 \\
  --disturbance-cooldown-steps 600 \\
  --use-render \\
  --image-size 224 \\
  --high-contrast-scene \\
  --disable_fabric \\
  --headless \\
  --enable_cameras
EOF
}

collect() {
  mkdir -p "$NPZ_DIR"
  python "$REPO_ROOT/scripts/collect_isaaclab_policy_npz.py" \
    --task "$TASK" \
    --policy-checkpoint "$POLICY_CHECKPOINT" \
    --target-frames "$TARGET_FRAMES" \
    --episode-len "$EPISODE_LEN" \
    --output-dir "$NPZ_DIR" \
    --continuous-disturbance \
    --disturbance-min "$DISTURBANCE_MIN" \
    --disturbance-max "$DISTURBANCE_MAX" \
    --disturbance-stable-steps 60 \
    --disturbance-cooldown-steps 600 \
    --use-render \
    --image-size 224 \
    --high-contrast-scene \
    --disable_fabric \
    --headless \
    --enable_cameras
}

convert() {
  mkdir -p "$(dirname "$H5_PATH")"
  python "$REPO_ROOT/scripts/convert_isaaclab_npz_to_h5.py" \
    "$NPZ_DIR" \
    "$H5_PATH" \
    --keys pixels action reward done policy_obs disturbance stable recovery_phase prediction_valid
}

inspect() {
  python "$REPO_ROOT/scripts/inspect_isaaclab_dataset.py" "$H5_PATH"
}

visualize() {
  mkdir -p "$VIS_DIR"
  python "$REPO_ROOT/scripts/visualize_pixels.py" \
    "$H5_PATH" \
    --episode 0 \
    --count 24 \
    --stride 8 \
    --out "$VIS_DIR/isaaclab_policy_disturbance_100k_sheet.png" \
    --gif "$VIS_DIR/isaaclab_policy_disturbance_100k_episode0.gif" \
    --fps 12
}

case "${1:-plan}" in
  plan)
    print_config
    echo
    print_collect_command
    ;;
  collect)
    print_config
    collect
    ;;
  convert)
    convert
    inspect
    ;;
  inspect)
    inspect
    ;;
  visualize)
    visualize
    ;;
  all)
    convert
    inspect
    visualize
    ;;
  *)
    echo "Usage: $(basename "$0") [plan|collect|convert|inspect|visualize|all]" >&2
    exit 2
    ;;
esac
