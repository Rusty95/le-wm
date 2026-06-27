#!/usr/bin/env bash
# Prepare a reproducible IsaacLab random-policy dataset for LeWM.
#
# This script intentionally defaults to "plan" mode so it does not start a
# long IsaacLab collection job unless the user explicitly asks for "collect".

set -euo pipefail

TASK="${TASK:-Isaac-Cartpole-RGB-Camera-Direct-v0}"
TARGET_FRAMES="${TARGET_FRAMES:-100000}"
EPISODE_LEN="${EPISODE_LEN:-80}"
NPZ_DIR="${NPZ_DIR:-/home/hall/code/.stable-wm/isaaclab_npz_100k}"
H5_PATH="${H5_PATH:-/home/hall/code/.stable-wm/datasets/isaaclab_random_100k.h5}"
PIXEL_KEY="${PIXEL_KEY:-policy}"
VIS_DIR="${VIS_DIR:-/home/hall/code/.stable-wm/visualizations}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [plan|collect|inspect|convert|visualize|all]

Default environment overrides:
  TASK=$TASK
  TARGET_FRAMES=$TARGET_FRAMES
  EPISODE_LEN=$EPISODE_LEN
  NPZ_DIR=$NPZ_DIR
  H5_PATH=$H5_PATH
  PIXEL_KEY=$PIXEL_KEY
  VIS_DIR=$VIS_DIR

Examples:
  # Print the exact commands without running long collection.
  le-wm/scripts/prepare_isaaclab_100k_dataset.sh plan

  # Run collection from an activated IsaacLab environment.
  source /home/hall/code/activate_isaaclab.sh
  le-wm/scripts/prepare_isaaclab_100k_dataset.sh collect

  # Convert and inspect from the LeWM environment.
  source /home/hall/code/activate_lewm.sh
  le-wm/scripts/prepare_isaaclab_100k_dataset.sh all
EOF
}

print_config() {
  cat <<EOF
[INFO] IsaacLab -> LeWM dataset plan
  task:          $TASK
  target_frames: $TARGET_FRAMES
  episode_len:   $EPISODE_LEN
  npz_dir:       $NPZ_DIR
  h5_path:       $H5_PATH
  pixel_key:     $PIXEL_KEY
  visualizations:$VIS_DIR
EOF
}

print_collect_command() {
  cat <<EOF
python "$REPO_ROOT/scripts/collect_isaaclab_random_npz.py" \\
  --task "$TASK" \\
  --target-frames "$TARGET_FRAMES" \\
  --episode-len "$EPISODE_LEN" \\
  --output-dir "$NPZ_DIR" \\
  --pixel-key "$PIXEL_KEY" \\
  --headless \\
  --enable_cameras
EOF
}

collect() {
  mkdir -p "$NPZ_DIR"
  python "$REPO_ROOT/scripts/collect_isaaclab_random_npz.py" \
    --task "$TASK" \
    --target-frames "$TARGET_FRAMES" \
    --episode-len "$EPISODE_LEN" \
    --output-dir "$NPZ_DIR" \
    --pixel-key "$PIXEL_KEY" \
    --headless \
    --enable_cameras
}

inspect_npz() {
  python "$REPO_ROOT/scripts/inspect_isaaclab_dataset.py" "$NPZ_DIR"
}

convert_h5() {
  mkdir -p "$(dirname "$H5_PATH")"
  python "$REPO_ROOT/scripts/convert_isaaclab_npz_to_h5.py" \
    "$NPZ_DIR" \
    "$H5_PATH" \
    --keys pixels action reward done
}

inspect_h5() {
  python "$REPO_ROOT/scripts/inspect_isaaclab_dataset.py" "$H5_PATH"
}

visualize() {
  mkdir -p "$VIS_DIR"
  python "$REPO_ROOT/scripts/visualize_pixels.py" \
    "$H5_PATH" \
    --episode 0 \
    --count 16 \
    --stride 1 \
    --out "$VIS_DIR/isaaclab_random_100k_sheet.png" \
    --gif "$VIS_DIR/isaaclab_random_100k_episode0.gif" \
    --fps 8
}

plan() {
  print_config
  cat <<EOF

[INFO] Collection command. Run this only in the IsaacLab environment:
EOF
  print_collect_command
  cat <<EOF

[INFO] After collection, run:
  $(basename "$0") inspect
  $(basename "$0") convert
  $(basename "$0") visualize

[INFO] Train with:
  cd "$REPO_ROOT"
  python train.py data=isaaclab_h5_100k
EOF
}

cmd="${1:-plan}"
case "$cmd" in
  plan)
    plan
    ;;
  collect)
    print_config
    collect
    ;;
  inspect)
    inspect_npz
    if [[ -f "$H5_PATH" ]]; then
      inspect_h5
    fi
    ;;
  convert)
    convert_h5
    inspect_h5
    ;;
  visualize)
    visualize
    ;;
  all)
    inspect_npz
    convert_h5
    inspect_h5
    visualize
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
