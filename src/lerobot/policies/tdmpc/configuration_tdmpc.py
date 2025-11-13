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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("tdmpc")
@dataclass
class TDMPCConfig(PreTrainedConfig):
    """TDMPCPolicy的配置类。

    默认配置针对xarm_lift_medium_replay训练进行了调优，该训练提供本体感觉和单个相机观测。

    您最有可能需要更改的参数是那些依赖于环境/传感器的参数。
    它们是：`input_shapes`、`output_shapes`，可能还有`max_random_shift_ratio`。

    参数:
        n_action_repeats: 重复规划返回的动作的次数。（提示：在Q学习中搜索动作重复，
            或询问您喜欢的聊天机器人）
        horizon: 模型预测控制的规划时域。
        n_action_steps: 从模型预测控制给出的计划中采取的动作步数。这是使用动作重复的替代方法。
            如果此值设置为大于1，则要求`n_action_repeats == 1`、`use_mpc == True`且
            `n_action_steps <= horizon`。注意，这种使用计划中多个步骤的方法不在原始实现中。
        input_shapes: 定义策略输入数据形状的字典。键表示输入数据名称，值是指示对应数据维度的列表。
            例如，"observation.image"指来自相机的输入，维度为[3, 96, 96]，表示它有三个颜色通道
            和96x96分辨率。重要的是，`input_shapes`不包括批次维度或时间维度。
        output_shapes: 定义策略输出数据形状的字典。键表示输出数据名称，值是指示对应数据维度的列表。
            例如，"action"指输出形状为[14]，表示14维动作。重要的是，`output_shapes`不包括批次维度
            或时间维度。
        input_normalization_modes: 字典，键表示模态（例如"observation.state"），值指定要应用的
            归一化模式。两种可用模式是"mean_std"（减去均值并除以标准差）和"min_max"（重新缩放到
            [-1, 1]范围）。注意，这里默认为None，表示输入不进行归一化。这是为了与原始实现匹配。
        output_normalization_modes: 与`normalize_input_modes`类似的字典，但用于反归一化到原始比例。
            注意，这也用于归一化训练目标。注意：在MPPI/CEM期间使用裁剪到[-1, +1]。因此，建议您
            坚持使用"min_max"归一化模式。
        image_encoder_hidden_dim: 用于编码图像的卷积层的通道数。
        state_encoder_hidden_dim: 用于编码状态向量的MLP的隐藏维度。
        latent_dim: 观测的潜在嵌入维度。
        q_ensemble_size: 用于估计不确定性的集成中Q函数估计器的数量。
        mlp_dim: 用于建模动力学编码器、奖励函数、策略(π)、Q集成和V的MLP的隐藏维度。
        discount: 强化学习形式的折扣因子(γ)。
        use_mpc: 是否使用模型预测控制。替代方法是为每一步仅采样策略模型(π)。
        cem_iterations: MPC中MPPI/CEM循环的迭代次数。
        max_std: CEM中从高斯PDF采样动作的最大标准差。
        min_std: 应用于从策略模型(π)采样的动作的噪声的最小标准差。也用作CEM中从高斯PDF
            采样动作的最小标准差。
        n_gaussian_samples: 每次CEM迭代从高斯分布中抽取的样本数。必须非零。
        n_pi_samples: 每次CEM迭代从策略/世界模型展开中抽取的样本数。可以为零。
        uncertainty_regularizer_coeff: 估计轨迹价值时使用的不确定性正则化系数（FOWM中方程4的λ系数）。
        n_elites: 每次CEM迭代用于更新高斯参数的精英样本数量。
        elite_weighting_temperature: 更新CEM的高斯参数时用于softmax加权（按轨迹价值）精英的温度。
        gaussian_mean_momentum: 用于CEM中优化的高斯参数的均值参数μ的EMA更新的动量(α)。
            更新计算为μ⁻ ← αμ⁻ + (1-α)μ。
        max_random_shift_ratio: 训练时应用于图像增强的最大随机偏移（作为图像大小的比例）（以像素为单位）。
            如果设置为0，则不应用此类增强。注意，假定输入图像为正方形以进行此增强。
        reward_coeff: 奖励回归损失的损失加权系数。
        expectile_weight: 用于状态价值函数(V)的期望分位数回归的权重(τ)。v_pred < v_target的权重为τ，
            v_pred >= v_target的权重为(1-τ)。τ预期在[0, 1]中。将τ设置为更接近1会导致更"乐观"的V。
            这样做是合理的，因为v_target是通过评估学习的状态-动作价值函数(Q)与样本内动作获得的，
            这些动作可能并不总是最优的。
        value_coeff: 状态-动作价值(Q) TD损失和状态价值(V)期望分位数回归损失的损失加权系数。
        consistency_coeff: 一致性损失的损失加权系数。
        advantage_scaling: 在对策略(π)估计器的参数进行优势加权回归之前，优势被缩放的因子。
            注意，指数化的优势被裁剪到100.0。
        pi_coeff: 动作回归损失的损失加权系数。
        temporal_decay_coeff: 用于指数衰减未来时间步损失系数的系数。提示：每个损失都是用从当前时间步
            开始的`horizon`步数的动作计算的。
        target_model_momentum: 用于目标模型的EMA更新的动量(α)。更新计算为ϕ ← αϕ + (1-α)θ，
            其中ϕ是目标模型的参数，θ是正在训练的模型的参数。
    """

    # 输入/输出结构。
    n_obs_steps: int = 1
    n_action_repeats: int = 2
    horizon: int = 5
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 架构/建模。
    # 神经网络。
    image_encoder_hidden_dim: int = 32
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 50
    q_ensemble_size: int = 5
    mlp_dim: int = 512
    # 强化学习。
    discount: float = 0.9

    # 推理。
    use_mpc: bool = True
    cem_iterations: int = 6
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 51
    uncertainty_regularizer_coeff: float = 1.0
    n_elites: int = 50
    elite_weighting_temperature: float = 0.5
    gaussian_mean_momentum: float = 0.1

    # 训练和损失计算。
    max_random_shift_ratio: float = 0.0476
    # 损失系数。
    reward_coeff: float = 0.5
    expectile_weight: float = 0.9
    value_coeff: float = 0.1
    consistency_coeff: float = 20.0
    advantage_scaling: float = 3.0
    pi_coeff: float = 0.5
    temporal_decay_coeff: float = 0.5
    # 目标模型。
    target_model_momentum: float = 0.995

    # 训练预设
    optimizer_lr: float = 3e-4

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽）。"""
        if self.n_gaussian_samples <= 0:
            raise ValueError(
                f"The number of gaussian samples for CEM should be non-zero. Got `{self.n_gaussian_samples=}`"
            )
        if self.normalization_mapping["ACTION"] is not NormalizationMode.MIN_MAX:
            raise ValueError(
                "TD-MPC assumes the action space dimensions to all be in [-1, 1]. Therefore it is strongly "
                f"advised that you stick with the default. See {self.__class__.__name__} docstring for more "
                "information."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.n_action_steps > 1:
            if self.n_action_repeats != 1:
                raise ValueError(
                    "If `n_action_steps > 1`, `n_action_repeats` must be left to its default value of 1."
                )
            if not self.use_mpc:
                raise ValueError("If `n_action_steps > 1`, `use_mpc` must be set to `True`.")
            if self.n_action_steps > self.horizon:
                raise ValueError("`n_action_steps` must be less than or equal to `horizon`.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # 目前应该最多只有一个图像键。
        if len(self.image_features) > 1:
            raise ValueError(
                f"{self.__class__.__name__} handles at most one image for now. Got image keys {self.image_features}."
            )

        if len(self.image_features) > 0:
            image_ft = next(iter(self.image_features.values()))
            if image_ft.shape[-2] != image_ft.shape[-1]:
                # TODO(alexander-soare): 此限制仅是因为随机偏移增强中的代码。应该可以移除。
                raise ValueError(f"Only square images are handled now. Got image shape {image_ft.shape}.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return list(range(self.horizon))
