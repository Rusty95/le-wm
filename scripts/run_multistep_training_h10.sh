#!/usr/bin/env bash
set -euo pipefail

source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm

exec python train.py \
  data=isaaclab_full_angle_120k \
  output_model_name=lewm_full_angle_multistep_h10 \
  subdir=lewm_full_angle_multistep_h10 \
  training_mode=autoregressive \
  num_preds=10 \
  loader.batch_size=64 \
  tensorboard.enabled=true \
  trainer.max_epochs=100
