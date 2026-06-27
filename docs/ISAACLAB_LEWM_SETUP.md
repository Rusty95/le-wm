# IsaacLab + LeWM 双环境搭建

本项目建议先使用双环境隔离：

- IsaacLab 环境：只负责仿真、相机观测、动作执行和数据采集。
- LeWM 环境：只负责数据读取、模型训练、checkpoint 生成和模型分析。

这样可以避免 IsaacLab 的 Python 3.11 / Isaac Sim 依赖和 LeWM 训练依赖互相覆盖。

## 1. 准备 LeWM 独立环境

在 `/home/hall/code` 下执行：

```bash
bash le-wm/scripts/setup_lewm_env.sh
source /home/hall/code/activate_lewm.sh
```

设置数据目录：

```bash
export STABLEWM_HOME=/home/hall/code/.stable-wm
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export SPT_CACHE_DIR=/home/hall/code/.stable-pretraining
mkdir -p "$LOCAL_DATASET_DIR/datasets"
mkdir -p "$SPT_CACHE_DIR"
```

运行不依赖真实数据的模型链路测试：

```bash
python le-wm/scripts/smoke_test_lewm.py
```

## 2. 用 IsaacLab 采集随机策略数据

切到 IsaacLab 环境：

```bash
source /home/hall/code/activate_isaaclab.sh
```

采集一个相机任务的数据。示例任务使用 IsaacLab 的 Cartpole RGB camera task；如果任务名不同，先用 IsaacLab 的 `list_envs.py` 确认。

```bash
python le-wm/scripts/collect_isaaclab_random_npz.py \
  --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
  --episodes 8 \
  --episode-len 80 \
  --output-dir /home/hall/code/.stable-wm/isaaclab_npz \
  --headless \
  --enable_cameras
```

默认脚本读取 `obs["policy"]` 作为 `pixels`。如果某个任务没有视觉 observation，可以加 `--use-render`，改用 `env.render()` 作为图像来源。

大规模采集建议使用可续采参数。PushT 文档里的 expert 数据集是 1000 episodes；如果先按 1000 条 episode 对齐，可以运行：

```bash
python le-wm/scripts/collect_isaaclab_random_npz.py \
  --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
  --episodes 1000 \
  --episode-len 80 \
  --output-dir /home/hall/code/.stable-wm/isaaclab_npz \
  --headless \
  --enable_cameras
```

如果要按帧数对齐，例如采到 2M frames：

```bash
python le-wm/scripts/collect_isaaclab_random_npz.py \
  --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
  --target-frames 2000000 \
  --episode-len 80 \
  --output-dir /home/hall/code/.stable-wm/isaaclab_npz \
  --headless \
  --enable_cameras
```

脚本会自动从已有最大 episode index 后继续写；中断后重新执行同一条命令即可续采。采集中途可以检查规模：

```bash
python le-wm/scripts/inspect_isaaclab_dataset.py \
  /home/hall/code/.stable-wm/isaaclab_npz
```

### 推荐：先准备 10 万帧验证数据

10 万帧适合作为第一版有效性验证数据：它比 smoke test 更接近真实训练规模，但还不会像百万帧数据那样占用太长采集时间。当前项目提供了一个编排脚本，默认只打印命令，不会自动启动长时间采集：

```bash
le-wm/scripts/prepare_isaaclab_100k_dataset.sh plan
```

默认配置：

```text
task: Isaac-Cartpole-RGB-Camera-Direct-v0
target_frames: 100000
episode_len: 80
npz_dir: /home/hall/code/.stable-wm/isaaclab_npz_100k
h5_path: /home/hall/code/.stable-wm/datasets/isaaclab_random_100k.h5
```

真正开始采集时，先进入 IsaacLab 环境，再显式运行 `collect`：

```bash
source /home/hall/code/activate_isaaclab.sh
le-wm/scripts/prepare_isaaclab_100k_dataset.sh collect
```

如果中断，重新运行同一条命令即可从已有最大 episode index 后续采。采集完成后回到 LeWM 环境，检查、转换并导出可视化：

```bash
source /home/hall/code/activate_lewm.sh
le-wm/scripts/prepare_isaaclab_100k_dataset.sh all
```

可以用环境变量覆盖默认参数，例如先采一个 160 帧 smoke test：

```bash
TARGET_FRAMES=160 EPISODE_LEN=80 \
  le-wm/scripts/prepare_isaaclab_100k_dataset.sh plan
```

## 3. 转换为 stable-worldmodel HDF5

回到 LeWM 环境：

```bash
source /home/hall/code/activate_lewm.sh
```

把 IsaacLab 采集的 episode `.npz` 转成 HDF5：

```bash
python le-wm/scripts/convert_isaaclab_npz_to_h5.py \
  /home/hall/code/.stable-wm/isaaclab_npz \
  /home/hall/code/.stable-wm/datasets/isaaclab_random.h5 \
  --keys pixels action reward done
```

LeWM 训练默认只使用 `pixels` 和 `action`，`reward/done` 会保留在文件里供后续分析。

## 4. 用 LeWM 读取 IsaacLab 数据训练

使用新增的数据配置：

```bash
cd /home/hall/code/le-wm
python train.py data=isaaclab_h5
```

如果使用 10 万帧数据配置：

```bash
cd /home/hall/code/le-wm
python train.py data=isaaclab_h5_100k
```

如果想用 TensorBoard 实时看训练曲线，先确保 LeWM 环境安装了 `tensorboard`，然后启用 logger：

```bash
source /home/hall/code/activate_lewm.sh
pip install tensorboard

cd /home/hall/code/le-wm
python train.py \
  data=isaaclab_h5_100k \
  tensorboard.enabled=true \
  tensorboard.config.name=lewm_isaaclab_100k \
  output_model_name=lewm_isaaclab_100k
```

训练曲线默认写到：

```text
/home/hall/code/.stable-pretraining/tensorboard
```

启动 TensorBoard：

```bash
tensorboard --logdir /home/hall/code/.stable-pretraining/tensorboard --host 0.0.0.0 --port 6006
```

配置文件：

```text
le-wm/config/train/data/isaaclab_h5.yaml
le-wm/config/train/data/isaaclab_h5_100k.yaml
```

默认数据名是：

```text
isaaclab_random.h5
isaaclab_random_100k.h5
```

会解析到：

```text
$STABLEWM_HOME/datasets/isaaclab_random.h5
```

## 5. 离线评估 IsaacLab HDF5

训练后可以先做离线 eval，检查世界模型在 IsaacLab 数据上的预测误差：

```bash
source /home/hall/code/activate_lewm.sh
cd /home/hall/code/le-wm
python scripts/eval_isaaclab_h5.py \
  --checkpoint lewm/weights_epoch_100.pt \
  --data isaaclab_random.h5 \
  --batch-size 16
```

快速 smoke test 可以只跑少量 batch：

```bash
python scripts/eval_isaaclab_h5.py \
  --checkpoint lewm/weights_epoch_100.pt \
  --data isaaclab_random \
  --batch-size 2 \
  --limit-batches 1 \
  --device cpu
```

输出会包含 `pred_loss`、`sigreg_loss`、`loss`，以及 `emb`、`act_emb`、`pred_emb`、`tgt_emb` 的 shape。

训练完成后还可以做多步 latent rollout 验证，检查自回归预测误差是否随 horizon 平滑增长：

```bash
python scripts/eval_multistep_rollout.py \
  --checkpoint lewm_isaaclab_100k/weights_epoch_100.pt \
  --data isaaclab_random_100k \
  --batch-size 32 \
  --horizons 1 3 5 10 \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_isaaclab_100k_rollout_eval.json
```

这个验证使用 HDF5 中的连续窗口；如果要做严格泛化验证，应该传入单独采集、未参与训练的 test HDF5。

## 6. 当前接口约定

IsaacLab `.npz` episode 至少需要：

```text
pixels: (T,H,W,C) 或 (T,C,H,W)
action: (T,A)
```

转换后的 HDF5 包含：

```text
ep_len
ep_offset
pixels
action
reward  # optional
done    # optional
```

LeWM DataLoader 读取后会得到：

```text
pixels: (B,T,C,H,W)
action: (B,T,frameskip * action_dim)
```

当前 `isaaclab_h5.yaml` 使用 `frameskip: 1`，所以动作维度保持为原始 `action_dim`。

## 7. 部署到 IsaacLab 内做在线式评估

第一版部署先不直接接管控制，而是在 IsaacLab 进程里完成以下闭环：

```text
IsaacLab env -> PPO policy action -> camera pixels/action buffer
             -> LeWM encode -> LeWM latent rollout -> future target latent MSE
```

这样可以先验证 LeWM checkpoint 能在 IsaacLab 环境内加载和推理，再继续做 MPC/规划控制。

当前脚本：

```text
le-wm/scripts/isaaclab_lewm_online_eval.py
```

它刻意不依赖 `stable_pretraining` / `stable_worldmodel`，而是直接用：

```text
transformers.ViTModel + le-wm/jepa.py + le-wm/module.py
```

恢复 `state_dict`，这样 IsaacLab 环境不用额外安装 LeWM 训练侧的一整套依赖。

短 smoke test：

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

正常验证：

```bash
python /home/hall/code/le-wm/scripts/isaaclab_lewm_online_eval.py \
  --episodes 4 \
  --episode-len 80 \
  --horizons 1 3 5 10 \
  --headless \
  --device cuda \
  --out /home/hall/code/.stable-wm/eval/lewm_online_isaaclab_eval.json
```

默认会使用：

```text
policy checkpoint:
/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole/2026-06-07_21-41-11/model_149.pt

LeWM checkpoint:
/home/hall/code/.stable-wm/checkpoints/lewm_isaaclab_mixed_balanced/weights_epoch_100.pt

action normalizer sources:
/home/hall/code/.stable-wm/datasets/isaaclab_random_100k.h5
/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_50k.h5
```

如果要换 checkpoint：

```bash
python /home/hall/code/le-wm/scripts/isaaclab_lewm_online_eval.py \
  --checkpoint your_run/weights_epoch_100.pt \
  --policy-checkpoint /path/to/model_149.pt \
  --episodes 4 \
  --episode-len 80 \
  --horizons 1 3 5 10 \
  --headless \
  --device cuda
```

当前脚本输出的是 latent MSE，不输出动作控制结果。下一步接 MPC 时，可以复用脚本里的这些函数：

```text
_preprocess_pixels
_normalize_actions
_load_lewm
model.encode(...)
model.rollout(...) 或 model.predict(...)
model.get_cost(...)
```

推荐下一阶段做法：

```text
1. 每步保存最近 history_size 帧 pixels 和 action。
2. 采样多条候选 action sequence。
3. 用 LeWM rollout 候选未来 latent。
4. 用目标 latent / 任务代价计算每条候选动作的 cost。
5. 执行 cost 最低序列的第一个 action。
```

## 8. LeWM-only MPC 控制 Cartpole

第一版脱离 PPO 的控制链路已经拆成三部分：

```text
scripts/lewm_isaaclab_common.py
scripts/train_cartpole_state_probe.py
scripts/isaaclab_lewm_mpc_cartpole.py
```

其中 `train_cartpole_state_probe.py` 会冻结 LeWM，把相机帧编码成 latent，并训练一个小 probe：

```text
latent emb -> [pole_pos, pole_vel, cart_pos, cart_vel]
```

训练 probe：

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

当前 probe 验证结果：

```text
test mse: 0.00249
test pole_pos mse: 0.00066
test pole_vel mse: 0.00454
test cart_pos mse: 0.00104
test cart_vel mse: 0.00371
```

运行 LeWM-only MPC smoke test：

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

当前较稳的短 horizon 参数：

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

当前观测到的能力边界：

```text
horizon=6, candidates=128, cem_iters=2:
  survival_steps: 72 / 80
  mean_abs_pole_angle: 0.257

horizon=12, candidates=512, cem_iters=3:
  survival_steps: 30 / 300
  mean_abs_pole_angle: 0.758
```

这说明 LeWM-only 控制链路已经跑通，但还不能称为稳定倒立摆控制。长 horizon 参数反而更差，符合 model-based control 中模型误差累积和 CEM 利用模型偏差的现象。下一步优先做：

```text
1. 继续使用短 horizon MPC。
2. 补采包含更大 pole angle / recovery 动作的数据。
3. 在 random 数据里也保存 policy_obs，用更丰富状态分布重新训练 probe。
4. 再考虑加入 action sequence warm-start 或 ensemble/uncertainty penalty。
```
