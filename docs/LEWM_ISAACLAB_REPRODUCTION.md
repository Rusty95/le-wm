# LeWM + IsaacLab Reproduction Guide

本文档用于在其它机器上复现当前 LeWM + IsaacLab 的主要成果。只保留最终采用的策略、脚本、命令和关键结果；中途讨论和废弃路线不再展开。

## Current Recommended Path

当前最优部署路径已经从 MPC 切换为：

```text
LeWM encoder + latent policy head
```

运行时不加载 PPO。PPO/swing-up policy 只作为离线数据来源；在线控制时，
IsaacLab 相机图像先进入 LeWM encoder，最近 3 帧 latent 拼接后送入
`LatentPolicyHead`，直接输出 action。

关键产物：

```text
LeWM checkpoint:
  /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt

Latent policy head:
  /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt
```

核心脚本：

```text
scripts/run_multistep_training_h10.sh
scripts/train_cartpole_latent_policy.py
scripts/isaaclab_lewm_policy_cartpole.py
```

实测结果：

```text
near-upright deployment:
  survival_steps      300 / 300
  reward_sum          1231.06
  mean_abs_pole_angle 0.0647 rad

bottom deployment:
  survival_steps      300 / 300
  reward_sum          714.07
  mean_abs_pole_angle 1.094 rad

long disturbance deployment:
  survival_steps      1200 / 1200
  reward_sum          3647.48
  first disturbance recovered after 309 steps
```

探索过程和路线切换原因见：

```text
docs/LEWM_ISAACLAB_EXPLORATION_LOG.md
```

当前成果分为四层：

1. IsaacLab 采集 Cartpole RGB camera 数据。
2. LeWM 在 IsaacLab 数据上离线训练和评估。
3. LeWM checkpoint 在 IsaacLab 进程内在线推理。
4. LeWM latent policy head 脱离 PPO 控制 Cartpole。

## 1. 代码与环境

推荐目录结构：

```text
/home/hall/code
├── le-wm
├── stable-worldmodel
├── stable-pretraining
└── RL-Learning-BasedOn-IsaacLab
```

保留双环境隔离：

```text
IsaacLab 环境:
  Python 3.11 / Isaac Sim / IsaacLab
  负责仿真、相机观测、环境 step、数据采集、在线部署

LeWM 环境:
  推荐 Python 3.10
  负责 HDF5 数据读取、训练、离线评估、probe 训练
```

LeWM 环境：

```bash
cd /home/hall/code
bash le-wm/scripts/setup_lewm_env.sh
source /home/hall/code/activate_lewm.sh

export STABLEWM_HOME=/home/hall/code/.stable-wm
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export SPT_CACHE_DIR=/home/hall/code/.stable-pretraining
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints" "$SPT_CACHE_DIR"

python le-wm/scripts/smoke_test_lewm.py
```

IsaacLab 环境：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
```

注意：

- IsaacLab 侧默认使用 `gymnasium`，不要改回旧 `gym`。
- 本地 IsaacLab 版本依赖 `gymnasium==1.2.0`，任务通过 `gym.make(task, cfg=env_cfg)` 创建。
- IsaacLab 环境不强制安装 `loguru`；在线部署脚本采用轻依赖 direct `state_dict` loader。

## 2. 数据采集与转换

最终采用两个训练集：

```text
isaaclab_random_100k.h5
  100000 frames
  random action
  pixels/action/reward/done

isaaclab_policy_camera_50k.h5
  50000 frames
  PPO policy action
  pixels/action/reward/done/policy_obs
```

另有一个未参与训练的 policy test set：

```text
isaaclab_policy_camera_test_10k.h5
  10000 frames
  用于 held-out eval 和 state probe test
```

### 2.1 Random 100k

采集 random 100k 使用编排脚本：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

cd /home/hall/code
le-wm/scripts/prepare_isaaclab_100k_dataset.sh collect
```

采集完成后转换、检查和可视化：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code
le-wm/scripts/prepare_isaaclab_100k_dataset.sh all
```

目标文件：

```text
/home/hall/code/.stable-wm/datasets/isaaclab_random_100k.h5
```

期望检查结果：

```text
episodes: 1250
frames: 100000
pixels: shape=(100000, 100, 100, 3), dtype=float32
action: shape=(100000, 1), dtype=float32
```

### 2.2 PPO Policy Camera 50k

先确保 PPO checkpoint 存在：

```text
/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole/2026-06-07_21-41-11/model_149.pt
```

采集：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

python /home/hall/code/le-wm/scripts/collect_isaaclab_policy_npz.py \
  --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
  --target-frames 50000 \
  --episode-len 80 \
  --output-dir /home/hall/code/.stable-wm/isaaclab_policy_camera_npz_50k \
  --headless \
  --device cuda
```

转换：

```bash
source /home/hall/code/activate_lewm.sh

python /home/hall/code/le-wm/scripts/convert_isaaclab_npz_to_h5.py \
  /home/hall/code/.stable-wm/isaaclab_policy_camera_npz_50k \
  /home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_50k.h5 \
  --keys pixels action reward done policy_obs
```

`policy_obs` 很重要，后续训练 `latent -> state` probe 会用它。顺序固定为：

```text
[pole_pos, pole_vel, cart_pos, cart_vel]
```

### 2.3 Policy Camera Test 10k

单独采一个测试集，不参与训练：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

python /home/hall/code/le-wm/scripts/collect_isaaclab_policy_npz.py \
  --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
  --target-frames 10000 \
  --episode-len 80 \
  --output-dir /home/hall/code/.stable-wm/isaaclab_policy_camera_npz_test_10k \
  --headless \
  --device cuda
```

转换：

```bash
source /home/hall/code/activate_lewm.sh

python /home/hall/code/le-wm/scripts/convert_isaaclab_npz_to_h5.py \
  /home/hall/code/.stable-wm/isaaclab_policy_camera_npz_test_10k \
  /home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_test_10k.h5 \
  --keys pixels action reward done policy_obs
```

### 2.4 连续随机扰动数据 100k（当前推荐）

当前方案让 swing-up PPO 持续运行。杆和小车连续稳定 60 帧后，
向杆施加一次随机方向、随机强度的角速度脉冲，幅值范围为
`2.4-6.0 rad/s`。恢复稳定并经过 cooldown 后再次扰动。采集场景固定
使用黑色地面、青色小车和黄色杆。

采集过程不会在输出文件边界 reset。每个 NPZ 最多保存 600 帧，
仅作为安全分段；真实环境 `done` 会立即结束当前分段，禁止跨 reset
拼接轨迹。

查看完整配置和命令：

```bash
cd /home/hall/code
le-wm/scripts/prepare_isaaclab_disturbance_100k_dataset.sh plan
```

在 IsaacLab 环境采集：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

cd /home/hall/code
le-wm/scripts/prepare_isaaclab_disturbance_100k_dataset.sh collect
```

在 LeWM 环境转换、检查并生成可视化：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code
le-wm/scripts/prepare_isaaclab_disturbance_100k_dataset.sh all
```

HDF5 额外保存：

```text
disturbance       本次 transition 注入的杆角速度脉冲，0 表示无扰动
stable            transition 开始前是否处于稳定区域
recovery_phase    是否正在从上一次扰动中恢复
prediction_valid  是否计算该 transition 的 LeWM 预测损失
```

外部脉冲不是 policy action 的一部分，因此注入脉冲的 transition 设置
`prediction_valid=False`。LeWM 会屏蔽这一项不可预测的损失，但仍学习
后续完整的倾倒、救杆和恢复过程。

训练：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm
python train.py data=isaaclab_policy_disturbance_100k
```

## 3. LeWM 混合训练

最终采用 balanced interleave，而不是简单把两个数据集拼接：

```yaml
# le-wm/config/train/data/isaaclab_mixed_balanced.yaml
dataset:
  num_steps: ${eval:'${num_preds} + ${history_size}'}
  frameskip: 1
  names:
    - isaaclab_random_100k.h5
    - isaaclab_policy_camera_50k.h5
  balance: interleave
  keys_to_load:
    - pixels
    - action
  keys_to_cache:
    - action
```

训练：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm

python train.py \
  data=isaaclab_mixed_balanced \
  output_model_name=lewm_isaaclab_mixed_balanced
```

期望 checkpoint：

```text
/home/hall/code/.stable-wm/checkpoints/lewm_isaaclab_mixed_balanced/weights_epoch_100.pt
```

训练时可以开启 TensorBoard：

```bash
python train.py \
  data=isaaclab_mixed_balanced \
  tensorboard.enabled=true \
  tensorboard.config.name=lewm_isaaclab_mixed_balanced \
  output_model_name=lewm_isaaclab_mixed_balanced

tensorboard \
  --logdir /home/hall/code/.stable-pretraining/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

## 4. 离线评估

### 4.1 One-step eval

Random 100k：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm

python scripts/eval_isaaclab_h5.py \
  --checkpoint lewm_isaaclab_mixed_balanced/weights_epoch_100.pt \
  --data isaaclab_random_100k \
  --batch-size 32 \
  --device cuda
```

Policy camera 50k：

```bash
python scripts/eval_isaaclab_h5.py \
  --checkpoint lewm_isaaclab_mixed_balanced/weights_epoch_100.pt \
  --data isaaclab_policy_camera_50k \
  --batch-size 32 \
  --device cuda
```

当前参考结果：

```text
random_100k:
  pred_loss: 0.01347
  sigreg_loss: 7.4776
  loss: 0.6865

policy_camera_50k:
  pred_loss: 0.01452
  sigreg_loss: 8.7371
  loss: 0.8009
```

### 4.2 Multi-step rollout eval

```bash
python scripts/eval_multistep_rollout.py \
  --checkpoint lewm_isaaclab_mixed_balanced/weights_epoch_100.pt \
  --data isaaclab_policy_camera_test_10k \
  --batch-size 32 \
  --horizons 1 3 5 10 \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_mixed_balanced_policy_camera_test_10k_rollout_eval.json
```

当前 held-out policy test 参考结果：

```text
horizon_1_mse: 0.01954
horizon_3_mse: 0.04739
horizon_5_mse: 0.06781
horizon_10_mse: 0.10838
```

### 4.3 Rollout GIF

LeWM 预测 latent，不直接解码像素；GIF 使用 nearest-neighbor frame 展示 predicted latent 最接近的数据帧。

```bash
python scripts/make_rollout_comparison_gif.py \
  --checkpoint lewm_isaaclab_mixed_balanced/weights_epoch_100.pt \
  --data isaaclab_policy_camera_test_10k \
  --episode 0 \
  --horizons 1 3 5 10 \
  --out /home/hall/code/.stable-wm/visualizations/lewm_mixed_balanced_policy_camera_test_10k_rollout_compare.gif
```

## 5. 当前推荐：Latent Policy Head 部署

### 5.1 训练 H10 多步 LeWM

当前推荐使用 autoregressive H10 多步训练：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm

bash scripts/run_multistep_training_h10.sh
```

期望 checkpoint：

```text
/home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt
```

### 5.2 训练 Latent Policy Head

冻结 LeWM，只训练一个轻量 MLP，把最近 3 帧 latent 映射到 action：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm

python scripts/train_cartpole_latent_policy.py \
  --checkpoint /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt \
  --train-data /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5 \
  --test-data /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_test_10k_seed9317.h5 \
  --action-stats-h5 /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5 \
  --cache-dir /home/hall/code/.stable-wm \
  --encode-batch-size 128 \
  --train-batch-size 1024 \
  --epochs 80 \
  --history-size 3 \
  --hidden-dim 256 \
  --out /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt \
  --metrics-out /home/hall/code/.stable-wm/eval/lewm_full_angle_latent_policy_eval.json \
  --device auto
```

参考离线结果：

```text
test MAE          0.160
sign match        74.7%
target action std 0.330
pred action std   0.224
```

### 5.3 IsaacLab 在线部署

近竖直初始：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/hall/code/le-wm:/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning:${PYTHONPATH:-}
export LD_PRELOAD=${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1
cd /home/hall/code

python le-wm/scripts/isaaclab_lewm_policy_cartpole.py \
  --task RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0 \
  --checkpoint /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt \
  --policy-head /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt \
  --action-stats-h5 /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5 \
  --episodes 1 \
  --episode-len 300 \
  --episode-length-s 25 \
  --initial-pole-angle-range -0.25 0.25 \
  --high-contrast-scene \
  --save-gif \
  --gif-out /home/hall/code/.stable-wm/visualizations/lewm_latent_policy_near_upright.gif \
  --out /home/hall/code/.stable-wm/eval/lewm_latent_policy_near_upright.json \
  --seed 9317 \
  --device cuda:0
```

底部附近初始：

```bash
python le-wm/scripts/isaaclab_lewm_policy_cartpole.py \
  --task RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0 \
  --checkpoint /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt \
  --policy-head /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt \
  --action-stats-h5 /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5 \
  --episodes 1 \
  --episode-len 300 \
  --episode-length-s 25 \
  --initial-pole-angle-range 2.8 3.14 \
  --high-contrast-scene \
  --save-gif \
  --gif-out /home/hall/code/.stable-wm/visualizations/lewm_latent_policy_bottom.gif \
  --out /home/hall/code/.stable-wm/eval/lewm_latent_policy_bottom.json \
  --seed 9317 \
  --device cuda:0
```

稳定后扰动长测：

```bash
python le-wm/scripts/isaaclab_lewm_policy_cartpole.py \
  --task RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0 \
  --checkpoint /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt \
  --policy-head /home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt \
  --action-stats-h5 /home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5 \
  --episodes 1 \
  --episode-len 1200 \
  --episode-length-s 100 \
  --initial-pole-angle-range -0.25 0.25 \
  --disturbance-start-step 60 \
  --disturbance-interval 160 \
  --disturbance-count 5 \
  --disturbance-min 2.4 \
  --disturbance-max 6.0 \
  --disturbance-stable-steps 60 \
  --disturbance-angle-threshold 0.15 \
  --disturbance-pole-vel-threshold 0.8 \
  --disturbance-cart-threshold 0.8 \
  --disturbance-cart-vel-threshold 0.5 \
  --high-contrast-scene \
  --save-gif \
  --gif-out /home/hall/code/.stable-wm/visualizations/lewm_latent_policy_long_disturbance.gif \
  --out /home/hall/code/.stable-wm/eval/lewm_latent_policy_long_disturbance.json \
  --seed 9317 \
  --device cuda:0
```

## 6. Legacy: IsaacLab 内在线部署评估

在线评估脚本：

```text
le-wm/scripts/isaaclab_lewm_online_eval.py
```

它在 IsaacLab 进程里运行：

```text
IsaacLab env -> PPO policy action -> live pixels/action
             -> LeWM encode -> LeWM latent rollout -> future latent MSE
```

这里仍使用 PPO 产生动作，但 LeWM 已在 IsaacLab 进程内推理。

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

timeout 180s python /home/hall/code/le-wm/scripts/isaaclab_lewm_online_eval.py \
  --episodes 1 \
  --episode-len 16 \
  --horizons 1 3 \
  --headless \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_online_isaaclab_smoke.json
```

当前 smoke 参考结果：

```text
horizon_1_mse: 0.00797
horizon_3_mse: 0.03185
```

设计选择：

- 不默认安装 `loguru`。
- 不依赖 `stable_worldmodel` loader。
- 直接使用 `transformers.ViTModel + jepa.JEPA + module.py` 复原模型。
- `lewm_isaaclab_common.py` 会自适应不同 `transformers` 版本的 ViT key 命名。

## 7. Legacy: LeWM-only MPC 控制 Cartpole

最终实现了第一版脱离 PPO 的控制闭环：

```text
live camera history + action history
  -> LeWM encode
  -> sample action sequences
  -> LeWM latent rollout
  -> state probe predicts [pole_pos, pole_vel, cart_pos, cart_vel]
  -> CEM/MPC chooses first action
  -> IsaacLab step
```

### 6.1 训练 state probe

脚本：

```text
le-wm/scripts/train_cartpole_state_probe.py
```

训练：

```bash
source /home/hall/code/activate_lewm.sh

python /home/hall/code/le-wm/scripts/train_cartpole_state_probe.py \
  --epochs 80 \
  --encode-batch-size 128 \
  --train-batch-size 1024 \
  --device cuda \
  --out /home/hall/code/.stable-wm/checkpoints/lewm_cartpole_state_probe.pt \
  --metrics-out /home/hall/code/.stable-wm/eval/lewm_cartpole_state_probe_eval.json
```

当前 probe test 参考结果：

```text
overall mse: 0.00249
pole_pos mse: 0.00066
pole_vel mse: 0.00454
cart_pos mse: 0.00104
cart_vel mse: 0.00371
```

### 6.2 运行 LeWM-only MPC

脚本：

```text
le-wm/scripts/isaaclab_lewm_mpc_cartpole.py
```

smoke test：

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

timeout 180s python /home/hall/code/le-wm/scripts/isaaclab_lewm_mpc_cartpole.py \
  --episodes 1 \
  --episode-len 20 \
  --horizon 4 \
  --num-candidates 64 \
  --cem-iters 1 \
  --headless \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_mpc_cartpole_smoke.json
```

当前 smoke 参考结果：

```text
survival_steps: 20 / 20
done_count: 0
```

当前较稳短 horizon 参数：

```bash
timeout 300s python /home/hall/code/le-wm/scripts/isaaclab_lewm_mpc_cartpole.py \
  --episodes 1 \
  --episode-len 80 \
  --horizon 6 \
  --num-candidates 128 \
  --cem-iters 2 \
  --headless \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_mpc_cartpole_80step.json
```

参考结果：

```text
survival_steps: 72 / 80
mean_abs_pole_angle: 0.257
```

正式长 horizon 参数当前并不更好：

```bash
timeout 420s python /home/hall/code/le-wm/scripts/isaaclab_lewm_mpc_cartpole.py \
  --episodes 1 \
  --episode-len 300 \
  --horizon 12 \
  --num-candidates 512 \
  --cem-iters 3 \
  --headless \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_mpc_cartpole_default_1ep.json
```

参考结果：

```text
survival_steps: 30 / 300
mean_abs_pole_angle: 0.758
```

结论：LeWM-only 控制链路已经跑通，确实可以不加载 PPO 产生动作；但当前还不能称为稳定控制。短 horizon 比长 horizon 更好，说明模型误差累积和 CEM 利用模型偏差已经出现。

## 8. 易错点

### 8.1 双环境不要混装

不要把 LeWM 训练依赖全装进 IsaacLab 环境。IsaacLab 侧保持轻依赖更稳。

当前在线部署脚本不依赖：

```text
stable_worldmodel
stable_pretraining
loguru
```

它直接复原模型结构并加载 `state_dict`。

### 8.2 GPU 在 sandbox 中可能不可见

如果普通 shell 里 `torch.cuda.is_available()` 是 `False`，但 IsaacLab 或训练需要 GPU，应在真实终端或允许 GPU 的环境中运行。

GB10 上可能出现 warning：

```text
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

目前 smoke test 可以跑，但长时间训练/部署最好确认 PyTorch 与 CUDA capability 匹配。

### 8.3 IsaacLab 异常退出可能卡住

Isaac Sim 在 Python 异常路径里可能卡住。调试命令建议套 `timeout`：

```bash
timeout 180s python ...
```

必要时清理残留：

```bash
pkill -f isaaclab_lewm_mpc_cartpole.py
pkill -f isaaclab_lewm_online_eval.py
```

### 8.4 100x100 图像可以训练，但不是最终上限

当前采集图像是：

```text
pixels: (N, 100, 100, 3)
```

训练前 resize 到 `224x224` 进入 ViT。这个分辨率能跑通验证，但如果 pole 很细或视角不理想，模型鲁棒性会受影响。

### 8.5 `policy_obs` 只在 policy 数据里有

`isaaclab_policy_camera_50k.h5` 和 test 10k 有 `policy_obs`，random 100k 当前没有。训练 state probe 必须用带 `policy_obs` 的数据。

### 8.6 `action` 必须使用训练时一致的 z-score

在线部署和 MPC 使用 mixed training action stats：

```text
isaaclab_random_100k.h5
isaaclab_policy_camera_50k.h5
```

不要直接把 raw action 当成 LeWM action embedding 输入。

### 8.7 Gymnasium 仍然是 IsaacLab RL 入口

IsaacLab 当前仍大量使用：

```python
import gymnasium as gym
env = gym.make(task, cfg=env_cfg)
```

不要改成旧 `gym`。同时保持新 step API：

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated
```

## 9. 后续路线

优先级从高到低：

1. **补采 recovery 数据**  
   当前数据中稳定 policy 样本较多，失败恢复样本不足。应采集更大 pole angle、cart 偏移、随机扰动后的恢复动作。

2. **random 数据补 `policy_obs`**  
   修改 `collect_isaaclab_random_npz.py`，像 policy collector 一样保存 `[pole_pos, pole_vel, cart_pos, cart_vel]`。然后用 random + policy 一起训练 state probe。

3. **短 horizon MPC 调参**  
   当前 `horizon=6, candidates=128, cem_iters=2` 比长 horizon 更稳。优先围绕短 horizon 调 cost 权重、action smoothness 和 warm-start。

4. **加入 uncertainty / ensemble penalty**  
   长 horizon 容易利用模型偏差。可以训练多个 probe 或多个 world model，用预测分歧惩罚不可靠 action sequence。

5. **更强视觉数据**  
   尝试更高分辨率、更清晰视角、多视角或随机化背景，但要避免 train/test 背景分布不一致。

6. **从 LeWM-MPC 走向 policy distillation**  
   如果 MPC 在线计算太慢，可以用 LeWM-MPC 生成动作，再蒸馏成一个轻量视觉 policy。

7. **更长期目标：纯视觉目标代价**  
   目前状态 probe 仍依赖 state supervision。后续可以尝试目标图像 latent cost，但第一版不建议直接跳过去。
