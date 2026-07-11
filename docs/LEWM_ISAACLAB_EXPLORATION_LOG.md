# LeWM + IsaacLab Exploration Log

本文记录从最初数据采集、LeWM 训练，到 IsaacLab 在线控制方案切换的主要过程。重点保留采取过的方法、观测到的问题、以及为什么切换到下一种方法。

## 最终结论

当前最优方案是：

```text
LeWM encoder + latent policy head
```

运行时不加载 PPO。PPO 或 swing-up policy 只作为离线数据来源。在线部署时流程为：

```text
IsaacLab RGB camera frames
  -> LeWM encoder
  -> 最近 3 帧 latent 拼接
  -> LatentPolicyHead
  -> IsaacLab action
```

最优结果：

```text
近竖直初始:
  survival_steps      300 / 300
  reward_sum          1231.06
  mean_abs_pole_angle 0.0647 rad
  mean_abs_cart_pos   0.554

底部附近初始:
  survival_steps      300 / 300
  reward_sum          714.07
  mean_abs_pole_angle 1.094 rad
  mean_abs_cart_pos   0.494

长时间稳定后扰动:
  survival_steps      1200 / 1200
  reward_sum          3647.48
  mean_abs_pole_angle 0.582 rad
  mean_abs_cart_pos   0.582
  first disturbance   recovered after 309 steps
```

## 1. 基础数据采集与 LeWM 训练

最开始使用 IsaacLab Cartpole RGB camera 数据训练 LeWM。先采集 random 数据，再采集 policy 数据，用 HDF5 作为稳定的数据接口。

早期 random 数据能验证链路，但策略分布太弱，不能覆盖稳定控制和强扰动恢复。后续切换到 PPO/swing-up policy 采集，并加入全角度与扰动数据。

关键原因：

```text
random 数据:
  优点: 容易采集，动作多样
  问题: 很多状态不接近有效控制轨迹，难以学到可部署行为

policy / swing-up 数据:
  优点: 状态-动作轨迹更接近控制目标
  问题: 可能覆盖不足，需要加入扰动和全角度初始化
```

## 2. 单步训练到多步训练

最初训练使用单步预测。离线指标能下降，但在线 rollout 和 MPC 需要多步预测，单步训练会在长 horizon 下累积误差。

后来在 `train.py` 中加入：

```yaml
training_mode: autoregressive
num_preds: 10
```

多步训练使用真实历史 latent 起步，然后把模型自己的预测 latent 递归喂回上下文。这样训练分布更接近在线 rollout。

对比结果：

```text
旧单步模型独立测试:
  H10 relative RMSE ≈ 37.19%

多步 H10 模型:
  H10 relative RMSE ≈ 15.32%
```

结论：多步 autoregressive 训练是后续控制部署的基础，应保留。

## 3. LeWM + MPC 尝试

第一版 LeWM-only 控制采用：

```text
LeWM rollout + state probe + CEM/random shooting MPC
```

动机是让 LeWM 作为世界模型，用候选动作 rollout 未来 latent，再通过 probe 解码状态，用手写 cost 选动作。

尝试过的改进包括：

```text
1. state-probe 目标
   预测 [sin(theta), cos(theta), cart_pos]

2. 边界保护
   防止 cart 持续向轨道边界滑动

3. history velocity probe
   用最近 3 帧 latent 预测 [sin, cos, cart_pos, pole_vel, cart_vel]

4. rollout-aware velocity probe
   用 LeWM 预测 latent 训练 probe，减少真实 latent 与 rollout latent 的分布差异
```

MPC 结果：

```text
旧 3D probe + edge protection:
  survival_steps      240 / 240
  reward_sum          36.15
  mean_abs_pole_angle 1.396

history velocity probe:
  survival_steps      240 / 240
  reward_sum          -115.58 ~ -236.11
  mean_abs_pole_angle 1.63 ~ 2.25

rollout-aware velocity probe:
  survival_steps      240 / 240
  reward_sum          -92.82
  mean_abs_pole_angle 1.571
```

切换原因：

```text
MPC 可以避免很快失败，但容易贴边振荡。
velocity probe 在离线测试中可用，但在 MPC 边界附近容易被 rollout 偏差放大。
手写 cost 难以同时处理 swing-up、回中、强救杆和动作平滑。
```

因此 MPC 不再作为当前推荐部署路线。

## 4. Latent Policy Head

最终采用轻量行为克隆头：

```text
输入: 最近 3 帧 LeWM latent, 3 * 192 = 576
输出: action, shape = (1,)
模型: LayerNorm + MLP + tanh action clamp
训练: 冻结 LeWM，只训练 policy head
```

训练目标是模仿离线数据中的 action。它不再通过手写 cost 做规划，而是直接从 LeWM latent 学一个控制映射。

离线测试：

```text
test MAE          0.160
sign match        74.7%
target action std 0.330
pred action std   0.224
```

选择原因：

```text
1. 在线速度快，不需要 CEM 候选采样。
2. 不加载 PPO，满足 LeWM latent 驱动部署目标。
3. 实测稳定性显著优于 MPC。
4. 代码路径简单，更适合作为当前仓库默认方案。
```

## 5. 当前保留策略

仓库中当前推荐保留并使用：

```text
train.py autoregressive training mode
scripts/run_multistep_training_h10.sh
scripts/train_cartpole_latent_policy.py
scripts/isaaclab_lewm_policy_cartpole.py
scripts/lewm_isaaclab_common.py 中的 LatentPolicyHead
```

MPC/probe 路线保留为历史参考，不再作为当前主线推荐。

## 6. 后续展望

下一步优先级：

```text
1. 用扰动恢复数据继续微调 latent policy head。
2. 做多 seed / 多初始角度批量评估，而不是只看单 seed。
3. 把 latent policy head 加入 TensorBoard 训练曲线。
4. 如果要进一步提升强扰动恢复，可尝试 DAgger:
   当前 latent policy 在线运行 -> 收集失败/恢复慢片段 -> 用 expert 或规则修正 action -> 继续训练 head。
5. 长远可以加入 value head，而不是回到纯手写 MPC cost。
```

