#!/usr/bin/env bash
# Collect and prepare full-angle Cartpole swing-up trajectories.

set -euo pipefail

TASK="${TASK:-RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0}"
TARGET_FRAMES="${TARGET_FRAMES:-120000}"
EPISODE_LEN="${EPISODE_LEN:-240}"
NPZ_DIR="${NPZ_DIR:-/home/hall/code/.stable-wm/isaaclab_full_angle_npz_120k}"
H5_PATH="${H5_PATH:-/home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5}"
VIS_DIR="${VIS_DIR:-/home/hall/code/.stable-wm/visualizations}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole_swingup_centered/2026-06-30_01-24-53_reward_v2/model_200.pt}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

collect() {
  mkdir -p "$NPZ_DIR"
  python "$REPO_ROOT/scripts/collect_isaaclab_policy_npz.py" \
    --task "$TASK" \
    --policy-checkpoint "$POLICY_CHECKPOINT" \
    --target-frames "$TARGET_FRAMES" \
    --episode-len "$EPISODE_LEN" \
    --output-dir "$NPZ_DIR" \
    --initial-pole-angle-range -3.141592653589793 3.141592653589793 \
    --action-noise-std 0.08 \
    --random-action-prob 0.12 \
    --random-action-scale 1.0 \
    --end-on-stable-steps 20 \
    --disturbance-angle-threshold 0.15 \
    --disturbance-cart-threshold 0.8 \
    --stable-pole-vel-threshold 0.8 \
    --stable-cart-vel-threshold 0.5 \
    --env-episode-length-s 30 \
    --high-contrast-scene \
    --headless \
    --enable_cameras
}

convert() {
  mkdir -p "$(dirname "$H5_PATH")"
  python "$REPO_ROOT/scripts/convert_isaaclab_npz_to_h5.py" \
    "$NPZ_DIR" \
    "$H5_PATH" \
    --keys pixels action reward done policy_obs
}

inspect() {
  python "$REPO_ROOT/scripts/inspect_isaaclab_dataset.py" "$H5_PATH"
  python "$REPO_ROOT/scripts/analyze_cartpole_coverage.py" "$H5_PATH"
}

visualize() {
  mkdir -p "$VIS_DIR"
  python "$REPO_ROOT/scripts/visualize_pixels.py" \
    "$H5_PATH" \
    --episode 0 \
    --count 20 \
    --stride 6 \
    --out "$VIS_DIR/isaaclab_full_angle_120k_sheet.png" \
    --gif "$VIS_DIR/isaaclab_full_angle_120k_episode0.gif" \
    --fps 12
}

case "${1:-plan}" in
  collect) collect ;;
  convert) convert ;;
  inspect) inspect ;;
  visualize) visualize ;;
  all) convert; inspect; visualize ;;
  plan)
    printf 'task=%s\ntarget_frames=%s\nepisode_len=%s\nnpz=%s\nh5=%s\n' \
      "$TASK" "$TARGET_FRAMES" "$EPISODE_LEN" "$NPZ_DIR" "$H5_PATH"
    ;;
  *)
    echo "Usage: $(basename "$0") [plan|collect|convert|inspect|visualize|all]" >&2
    exit 2
    ;;
esac
