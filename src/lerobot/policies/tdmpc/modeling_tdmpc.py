#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""在真实世界中微调离线世界模型的实现。

此代码中的注释有时会引用以下参考文献：
    TD-MPC论文：模型预测控制的时序差分学习（https://huggingface.co/papers/2203.04955）
    FOWM论文：在真实世界中微调离线世界模型（https://huggingface.co/papers/2310.16029）
"""

# ruff: noqa: N806

from collections import deque
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.utils import get_device_from_parameters, get_output_shape, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_PREFIX, OBS_STATE, OBS_STR, REWARD


class TDMPCPolicy(PreTrainedPolicy):
    """TD-MPC学习和推理的实现。

    请注意此策略的几个警告：
        - 使用原始FOWM代码（https://github.com/fyhMer/fowm）创建的预训练权重的评估按预期工作。
          具体来说：我们使用FOWM代码针对xarm_lift_medium_replay数据集训练和评估了一个模型。
          我们将权重移植到LeRobot，并能够使用相同的成功指标进行评估。但是，我们必须使用进程间
          通信来使用来自FOWM的xarm环境。这是因为我们的xarm环境使用更新的依赖项，与FOWM中的
          环境不匹配。有关实现细节，请参阅https://github.com/huggingface/lerobot/pull/103。
        - 我们尚未检查在LeRobot上的训练是否能重现FOWM的结果。
        - 尽管如此，我们已经验证了我们可以为PushT训练TD-MPC。参见
          `lerobot/configs/policy/tdmpc_pusht_keypoints.yaml`。
        - 我们当前的xarm数据集是使用来自FOWM的环境生成的。因此它们与我们的xarm环境不匹配。
    """

    config_class = TDMPCConfig
    name = "tdmpc"

    def __init__(
        self,
        config: TDMPCConfig,
    ):
        """
        Args:
            config: 策略配置类实例，如果为None，则使用配置类的默认实例化。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = TDMPCTOLD(config)
        self.model_target = deepcopy(self.model)
        for param in self.model_target.parameters():
            param.requires_grad = False

        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """
        清空观测和动作队列。清除用于MPPI/CEM热启动的先前均值。应该在`env.reset()`时调用。
        """
        self._queues = {
            OBS_STATE: deque(maxlen=1),
            ACTION: deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGE] = deque(maxlen=1)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=1)
        # 从MPC期间使用的交叉熵方法(CEM)获得的先前均值。它用于为下一步热启动CEM。
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测预测一组动作。"""
        batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}

        # 删除时间维度，因为尚未处理。
        for key in batch:
            assert batch[key].shape[1] == 1
            batch[key] = batch[key][:, 0]

        # 注意：这里观测的顺序很重要。
        encode_keys = []
        if self.config.image_features:
            encode_keys.append(OBS_IMAGE)
        if self.config.env_state_feature:
            encode_keys.append(OBS_ENV_STATE)
        encode_keys.append(OBS_STATE)
        z = self.model.encode({k: batch[k] for k in encode_keys})
        if self.config.use_mpc:  # noqa: SIM108
            actions = self.plan(z)  # (horizon, batch, action_dim)
        else:
            # 仅使用策略(π)进行规划。这总是返回一个动作，因此需要unsqueeze来获得
            # 序列维度，就像MPC分支中那样。
            actions = self.model.pi(z).unsqueeze(0)

        actions = torch.clamp(actions, -1, +1)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测选择单个动作。"""
        # 注意：对于离线评估，批次中包含动作，因此我们需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以便添加键不会修改原始字典
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]
        # 注意：对于离线评估，批次中包含动作，因此我们需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        # 当动作队列耗尽时，通过查询策略再次填充它。
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues[ACTION].append(actions[0])
            else:
                # 动作队列是(n_action_steps, batch_size, action_dim)，因此我们转置动作。
                self._queues[ACTION].extend(actions[: self.config.n_action_steps])

        action = self._queues[ACTION].popleft()
        return action

    @torch.no_grad()
    def plan(self, z: Tensor) -> Tensor:
        """使用TD-MPC推理规划动作序列。

        Args:
            z: 初始状态的(batch, latent_dim,)张量。
        Returns:
            规划的动作轨迹的(horizon, batch, action_dim,)张量。
        """
        device = get_device_from_parameters(self)

        batch_size = z.shape[0]

        # 从策略中采样Nπ条轨迹。
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.action_feature.shape[0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # 注意：在推理过程中在这里添加少量噪声不会有害，甚至可能对CEM有帮助。
                pi_actions[t] = self.model.pi(_z, self.config.min_std)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # 在CEM循环中，我们需要这个来调用estimate_value，用于高斯采样的轨迹。
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # 模型预测路径积分(MPPI)，使用交叉熵方法(CEM)作为优化算法。
        # 交叉熵方法(CEM)的初始均值和标准差。
        mean = torch.zeros(
            self.config.horizon, batch_size, self.config.action_feature.shape[0], device=device
        )
        # 可能使用前一步的均值热启动CEM。
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # 从高斯分布中随机采样动作轨迹。
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.action_feature.shape[0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # 计算精英动作。
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value, self.config.n_elites, dim=0).indices  # (n_elites, batch)
            elite_value = value.take_along_dim(elite_idxs, dim=0)  # (n_elites, batch)
            # (horizon, n_elites, batch, action_dim)
            elite_actions = actions.take_along_dim(einops.rearrange(elite_idxs, "n b -> 1 n b 1"), dim=1)

            # 更新高斯PDF参数为精英样本的（加权）均值和标准差。
            max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
            # 权重是轨迹值的softmax。注意，这与TD-MPC论文中方程4中Ω的用法不同。
            # 相反，它是归一化版本：s = Ω/ΣΩ。这使得方程为：μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²)。
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(axis=0, keepdim=True)
            # (horizon, batch, action_dim)
            _mean = torch.sum(einops.rearrange(score, "n b -> n b 1") * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n b -> n b 1")
                    * (elite_actions - einops.rearrange(_mean, "h b d -> h 1 b d")) ** 2,
                    dim=1,
                )
            )
            # 使用指数移动平均更新均值，使用直接替换更新标准差。
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp_(self.config.min_std, self.config.max_std)

        # 跟踪均值以热启动后续步骤。
        self._prev_mean = mean

        # 使用最后一次迭代的softmax分数，从MPPI/CEM的最后一次迭代的精英动作中随机选择一个。
        actions = elite_actions[:, torch.multinomial(score.T, 1).squeeze(), torch.arange(batch_size)]

        return actions

    @torch.no_grad()
    def estimate_value(self, z: Tensor, actions: Tensor):
        """根据FOWM论文的方程4估计轨迹的价值。

        Args:
            z: 初始潜在状态的(batch, latent_dim)张量。
            actions: 动作轨迹的(horizon, batch, action_dim)张量。
        Returns:
            价值的(batch,)张量。
        """
        # 初始化回报和运行折扣因子。
        G, running_discount = 0, 1
        # 遍历轨迹中的动作，使用潜在动力学模型模拟轨迹。跟踪回报。
        for t in range(actions.shape[0]):
            # 我们稍后会计算奖励。首先计算FOWM论文方程4中的不确定性正则化器。
            if self.config.uncertainty_regularizer_coeff > 0:
                regularization = -(
                    self.config.uncertainty_regularizer_coeff * self.model.Qs(z, actions[t]).std(0)
                )
            else:
                regularization = 0
            # 估计下一个状态（潜在）和奖励。
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            # 更新回报和运行折扣。
            G += running_discount * (reward + regularization)
            running_discount *= self.config.discount
        # 添加最终状态的估计值（使用最小值进行保守估计）。
        # 通过预测下一个动作，然后在状态-动作价值估计器集合上取最小值来实现。
        # 注意：在推理时，这少量添加的噪声似乎有点帮助，如在xarm_lift_medium_replay的
        # 50个episode上观察到的成功指标所示。
        next_action = self.model.pi(z, self.config.min_std)  # (batch, action_dim)
        terminal_values = self.model.Qs(z, next_action)  # (ensemble, batch)
        # 随机选择2个Q函数进行终止值估计（如FOWM论文附录C所述）。
        if self.config.q_ensemble_size > 2:
            G += (
                running_discount
                * torch.min(terminal_values[torch.randint(0, self.config.q_ensemble_size, size=(2,))], dim=0)[
                    0
                ]
            )
        else:
            G += running_discount * torch.min(terminal_values, dim=0)[0]
        # 最后，也对终止值进行正则化。
        if self.config.uncertainty_regularizer_coeff > 0:
            G -= running_discount * self.config.uncertainty_regularizer_coeff * terminal_values.std(0)
        return G

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """通过模型运行批次并计算损失。

        返回一个字典，其中损失作为张量，其他信息作为原生浮点数。
        """
        device = get_device_from_parameters(self)

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以便添加键不会修改原始字典
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch[ACTION]  # (t, b, action_dim)
        reward = batch[REWARD]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}

        # 应用随机图像增强。
        if self.config.image_features and self.config.max_random_shift_ratio > 0:
            observations[OBS_IMAGE] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations[OBS_IMAGE],
            )

        # 获取用于预测轨迹的当前观测，以及用于潜在一致性损失和TD损失的所有未来观测。
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            OBS_IMAGE if self.config.image_features else OBS_ENV_STATE
        ].shape[:2]

        # 使用潜在动力学模型和策略模型运行潜在展开。
        # 注意这个形状是`horizon+1`，因为有`horizon`个动作和一个当前的`z`。
        # 每个动作给我们一个下一个`z`。
        batch_size = batch["index"].shape[0]
        z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        z_preds[0] = self.model.encode(current_observation)
        reward_preds = torch.empty_like(reward, device=device)
        for t in range(horizon):
            z_preds[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(z_preds[t], action[t])

        # 基于潜在展开计算Q和V价值预测。
        q_preds_ensemble = self.model.Qs(z_preds[:-1], action)  # (ensemble, horizon, batch)
        v_preds = self.model.V(z_preds[:-1])
        info.update({"Q": q_preds_ensemble.mean().item(), "V": v_preds.mean().item()})

        # 使用stopgrad计算各种目标。
        with torch.no_grad():
            # 潜在状态一致性目标。
            z_targets = self.model_target.encode(next_observations)
            # 状态-动作价值目标（或TD目标），如FOWM方程3所示。与TD-MPC使用学习的
            # 状态-动作价值函数结合学习的策略：Q(z, π(z))不同，FOWM使用学习的
            # 状态价值函数：V(z)。这意味着TD目标仅依赖于样本内动作（而不是由π估计的动作）。
            # 注意：这里我们不使用self.model_target，而是使用self.model。
            # 这是为了遵循原始代码和FOWM论文。
            q_targets = reward + self.config.discount * self.model.V(self.model.encode(next_observations))
            # 来自FOWM方程3。它们显示为Q(z, a)。这里我们称它们为v_targets，
            # 以强调我们使用它们来计算V的损失。
            v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_min=True)

        # 计算损失。
        # 相对于时间步长指数衰减损失权重。未来更远的步骤对损失的影响较小。
        # 注意：unsqueeze将让我们广播到(seq, batch)。
        temporal_loss_coeffs = torch.pow(
            self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        ).unsqueeze(-1)
        # 计算一致性损失，作为从展开预测的潜在变量与从（目标模型的）观测编码器
        # 预测的潜在变量之间的MSE损失。
        consistency_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1)
                # `z_preds`依赖于当前观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # `z_targets`依赖于下一个观测。
                * ~batch[f"{OBS_STR}.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # 计算奖励损失，作为从展开预测的奖励与数据集奖励之间的MSE损失。
        reward_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(reward_preds, reward, reduction="none")
                * ~batch["next.reward_is_pad"]
                # `reward_preds`依赖于当前观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )
        # 为集合中的所有Q函数计算状态-动作价值损失（TD损失）。
        q_value_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(
                    q_preds_ensemble,
                    einops.repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0]),
                    reduction="none",
                ).sum(0)  # 在集合上求和
                # `q_preds_ensemble`依赖于第一个观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # q_targets依赖于奖励和下一个观测。
                * ~batch["next.reward_is_pad"]
                * ~batch[f"{OBS_STR}.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # 根据FOWM方程3计算状态价值损失。
        diff = v_targets - v_preds
        # 期望分位数损失惩罚：
        #   - `v_preds <  v_targets`使用权重`expectile_weight`
        #   - `v_preds >= v_targets`使用权重`1 - expectile_weight`
        raw_v_value_loss = torch.where(
            diff > 0, self.config.expectile_weight, (1 - self.config.expectile_weight)
        ) * (diff**2)
        v_value_loss = (
            (
                temporal_loss_coeffs
                * raw_v_value_loss
                # `v_targets`依赖于第一个观测和动作，`v_preds`也是如此。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )

        # 计算FOWM 3.1中详细说明的π的优势加权回归损失。
        # 我们不再需要这些梯度，因此detach。
        z_preds = z_preds.detach()
        # 使用stopgrad进行优势计算。
        with torch.no_grad():
            advantage = self.model_target.Qs(z_preds[:-1], action, return_min=True) - self.model.V(
                z_preds[:-1]
            )
            info["advantage"] = advantage[0]
            # (t, b)
            exp_advantage = torch.clamp(torch.exp(advantage * self.config.advantage_scaling), max=100.0)
        action_preds = self.model.pi(z_preds[:-1])  # (t, b, a)
        # 计算动作和动作预测之间的MSE。
        # 注意：FOWM的原始代码计算对数概率（相对于单位标准差高斯分布），
        # 并在动作维度上求和。计算（负）对数概率相当于将MSE乘以0.5并添加
        # 常数偏移量（log(2*pi)/2项，乘以动作维度）。这里我们删除常数偏移量，
        # 因为它不会改变优化步骤，并且我们删除0.5，因为我们为其设置了一个
        # 配置参数（见下面计算总损失的地方）。
        mse = F.mse_loss(action_preds, action, reduction="none").sum(-1)  # (t, b)
        # 注意：原始实现不像其他损失那样在时间维度上求和。
        # TODO(alexander-soare): 在时间维度上求和，并检查训练是否仍然按预期工作。
        pi_loss = (
            exp_advantage
            * mse
            * temporal_loss_coeffs
            # `action_preds`依赖于第一个观测和动作。
            * ~batch[f"{OBS_STR}.state_is_pad"][0]
            * ~batch["action_is_pad"]
        ).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + self.config.value_coeff * v_value_loss
            + self.config.pi_coeff * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": q_value_loss.item(),
                "V_value_loss": v_value_loss.item(),
                "pi_loss": pi_loss.item(),
                "sum_loss": loss.item() * self.config.horizon,
            }
        )

        # 撤销(b, t) -> (t, b)。
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return loss, info

    def update(self):
        """使用EMA步骤更新目标模型的参数。"""
        # 注意与原始FOWM代码的微小差异。这里他们基于EMA更新频率参数进行此操作，
        # 该参数设置为2（每2步进行一次更新）。为了简化代码，我们每步更新并相应地
        # 调整衰减参数`alpha`（0.99 -> 0.995）
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)


class TDMPCTOLD(nn.Module):
    """TD-MPC中使用的任务导向潜在动力学(TOLD)模型。"""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config
        self._encoder = TDMPCObservationEncoder(config)
        self._dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Sigmoid(),
        )
        self._reward = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._pi = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.action_feature.shape[0]),
        )
        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
                    nn.LayerNorm(config.mlp_dim),
                    nn.Tanh(),
                    nn.Linear(config.mlp_dim, config.mlp_dim),
                    nn.ELU(),
                    nn.Linear(config.mlp_dim, 1),
                )
                for _ in range(config.q_ensemble_size)
            ]
        )
        self._V = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Tanh(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.ELU(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重。

        所有线性层和卷积层的权重使用正交初始化（除了奖励网络和Q网络的最后一层，
        它们使用零初始化）。
        所有线性层和卷积层的偏置使用零初始化。
        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight.data, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_apply_fn)
        for m in [self._reward, *self._Qs]:
            assert isinstance(m[-1], nn.Linear), (
                "完整性检查。最后一个线性层需要对权重进行零初始化。"
            )
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)  # 这已经完成了，但为了安全起见保留此行

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """将观测编码为其潜在表示。"""
        return self._encoder(obs)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """给定当前潜在状态和动作，预测下一个状态的潜在表示和奖励。

        Args:
            z: 当前状态潜在表示的(*, latent_dim)张量。
            a: 要应用的动作的(*, action_dim)张量。
        Returns:
            包含以下内容的元组：
                - 下一个状态潜在表示的(*, latent_dim)张量。
                - 估计奖励的(*,)张量。
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x).squeeze(-1)

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        """给定当前潜在状态和动作，预测下一个状态的潜在表示。

        Args:
            z: 当前状态潜在表示的(*, latent_dim)张量。
            a: 要应用的动作的(*, action_dim)张量。
        Returns:
            下一个状态潜在表示的(*, latent_dim)张量。
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def pi(self, z: Tensor, std: float = 0.0) -> Tensor:
        """从学习的策略中采样动作。

        策略还可以添加（截断的）高斯噪声，以在生成在线训练的展开时鼓励探索。

        Args:
            z: 当前状态潜在表示的(*, latent_dim)张量。
            std: 注入噪声的标准差。
        Returns:
            采样动作的(*, action_dim)张量。
        """
        action = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(action) * std
            action += torch.randn_like(action) * std
        return action

    def V(self, z: Tensor) -> Tensor:  # noqa: N802
        """预测状态价值(V)。

        Args:
            z: 当前状态潜在表示的(*, latent_dim)张量。
        Returns:
            估计状态价值的(*,)张量。
        """
        return self._V(z).squeeze(-1)

    def Qs(self, z: Tensor, a: Tensor, return_min: bool = False) -> Tensor:  # noqa: N802
        """预测所有学习的Q函数的状态-动作价值。

        Args:
            z: 当前状态潜在表示的(*, latent_dim)张量。
            a: 要应用的动作的(*, action_dim)张量。
            return_min: 设置为true以实现FOWM论文附录C中的细节：随机选择2个Q函数并返回最小值
        Returns:
            集合中每个学习的Q函数的价值预测的(q_ensemble, *)张量，或者
            如果return_min=True则为(*,)张量。
        """
        x = torch.cat([z, a], dim=-1)
        if not return_min:
            return torch.stack([q(x).squeeze(-1) for q in self._Qs], dim=0)
        else:
            if len(self._Qs) > 2:  # noqa: SIM108
                Qs = [self._Qs[i] for i in np.random.choice(len(self._Qs), size=2)]
            else:
                Qs = self._Qs
            return torch.stack([q(x).squeeze(-1) for q in Qs], dim=0).min(dim=0)[0]


class TDMPCObservationEncoder(nn.Module):
    """编码图像和/或状态向量观测。"""

    def __init__(self, config: TDMPCConfig):
        """
        为像素和/或状态模态创建编码器。
        TODO(alexander-soare): 原始工作允许通过沿通道维度连接多个图像。
            重新实现此功能。
        """
        super().__init__()
        self.config = config

        if config.image_features:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    next(iter(config.image_features.values())).shape[0],
                    config.image_encoder_hidden_dim,
                    7,
                    stride=2,
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
            )
            dummy_shape = (1, *next(iter(config.image_features.values())).shape)
            out_shape = get_output_shape(self.image_enc_layers, dummy_shape)[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Sigmoid(),
                )
            )

        if config.robot_state_feature:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(config.robot_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

        if config.env_state_feature:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(config.env_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """编码图像和/或状态向量。

        每个模态被编码为大小为(latent_dim,)的特征向量，然后在所有特征上取均匀平均值。
        """
        feat = []
        # 注意：这里观测的顺序很重要。
        if self.config.image_features:
            feat.append(
                flatten_forward_unflatten(
                    self.image_enc_layers, obs_dict[next(iter(self.config.image_features))]
                )
            )
        if self.config.env_state_feature:
            feat.append(self.env_state_enc_layers(obs_dict[OBS_ENV_STATE]))
        if self.config.robot_state_feature:
            feat.append(self.state_enc_layers(obs_dict[OBS_STATE]))
        return torch.stack(feat, dim=0).mean(0)


def random_shifts_aug(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """水平和垂直随机移动图像。

    改编自 https://github.com/facebookresearch/drqv2
    """
    b, _, h, w = x.size()
    assert h == w, "尚未处理非正方形图像"
    pad = int(round(max_random_shift_ratio * h))
    x = F.pad(x, tuple([pad] * 4), "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        h + 2 * pad,
        device=x.device,
        dtype=torch.float32,
    )[:h]
    arange = einops.repeat(arange, "w -> h w 1", h=h)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = einops.repeat(base_grid, "h w c -> b h w c", b=b)
    # 以像素为单位并在填充边界内的随机偏移。
    shift = torch.randint(
        0,
        2 * pad + 1,
        size=(b, 1, 1, 2),
        device=x.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """原地更新EMA参数，使用公式 ema_param <- alpha * ema_param + (1 - alpha) * param。"""
    for ema_module, module in zip(ema_net.modules(), net.modules(), strict=True):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False), strict=True
        ):
            assert n_p_ema == n_p, "EMA模型更新的参数名称不匹配"
            if isinstance(p, dict):
                raise RuntimeError("不支持字典参数")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # 直接复制BatchNorm参数和不可训练参数。
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            with torch.no_grad():
                p_ema.mul_(alpha)
                p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)


def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """辅助函数，用于临时展平图像张量开头的额外维度。

    Args:
        fn: 将传递图像张量的可调用对象。它应该接受(B, C, H, W)并返回
            (B, *)，其中*是任意数量的维度。
        image_tensor: 形状为(**, C, H, W)的图像张量，其中**是任意数量的维度，
            通常与*不同。
    Returns:
        从可调用对象返回的值，重塑为(**, *)。
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))
