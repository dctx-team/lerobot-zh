#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """动作分块Transformer策略的配置类。

    默认配置用于在双臂Aloha任务（如"insertion"或"transfer"）上进行训练。

    您最有可能需要更改的参数是依赖于环境/传感器的参数。
    这些参数包括：`input_shapes` 和 'output_shapes`。

    关于输入和输出的说明：
        - 以下两种情况至少需要满足一种：
            - 至少需要一个以"observation.image"开头的键作为输入。
              和/或
            - 需要键"observation.environment_state"作为输入。
        - 如果有多个以"observation.images."开头的键，它们被视为多个相机视角。
          目前我们仅支持所有图像具有相同的形状。
        - 可以选择不使用"observation.state"键作为本体感觉机器人状态。
        - "action"是必需的输出键。

    参数:
        n_obs_steps: 传递给策略的观察的环境步数（取当前步和之前的额外步）。
        chunk_size: 以环境步为单位的动作预测"块"的大小。
        n_action_steps: 策略一次调用在环境中运行的动作步数。
            这应该不大于块大小。例如，如果块大小为100，您可以将其设置为50。
            这意味着模型预测100步的动作，在环境中运行50步，并丢弃其他50步。
        input_shapes: 定义策略输入数据形状的字典。键表示输入数据名称，值是指示
            相应数据维度的列表。例如，"observation.image"指来自相机的输入，尺寸为
            [3, 96, 96]，表示它有三个颜色通道和96x96分辨率。重要的是，`input_shapes`
            不包括批次维度或时间维度。
        output_shapes: 定义策略输出数据形状的字典。键表示输出数据名称，值是指示
            相应数据维度的列表。例如，"action"指输出形状为[14]，表示14维动作。
            重要的是，`output_shapes`不包括批次维度或时间维度。
        input_normalization_modes: 字典，键表示模态（例如"observation.state"），
            值指定要应用的归一化模式。两种可用模式是"mean_std"（减去均值并除以标准差）
            和"min_max"（在[-1, 1]范围内重新缩放）。
        output_normalization_modes: 与 `normalize_input_modes` 类似的字典，但用于
            反归一化到原始尺度。注意，这也用于归一化训练目标。
        vision_backbone: 用于编码图像的torchvision resnet骨干网络的名称。
        pretrained_backbone_weights: 来自torchvision的预训练权重，用于初始化骨干网络。
            `None`表示没有预训练权重。
        replace_final_stride_with_dilation: 是否用扩张卷积替换ResNet的最终2x2步幅。
        pre_norm: 是否在transformer块中使用"pre-norm"。
        dim_model: transformer块的主要隐藏维度。
        n_heads: transformer块的多头注意力中使用的头数。
        dim_feedforward: transformer的前馈层中扩展隐藏维度的维度。
        feedforward_activation: transformer块的前馈层中使用的激活函数。
        n_encoder_layers: transformer编码器使用的transformer层数。
        n_decoder_layers: transformer解码器使用的transformer层数。
        use_vae: 是否在训练期间使用变分目标。这引入了另一个transformer，用作VAE的编码器
            （不要与transformer编码器混淆 - 请参阅策略类中的文档）。
        latent_dim: VAE的潜在维度。
        n_vae_encoder_layers: VAE编码器使用的transformer层数。
        temporal_ensemble_coeff: 应用于时序集成的指数加权方案的系数。默认为None，
            表示不使用时序集成。使用此功能时 `n_action_steps` 必须为1，因为推理需要在
            每一步进行以形成集成。有关集成如何工作的更多信息，请参阅 `ACTTemporalEnsembler`。
        dropout: transformer层中使用的Dropout（详见代码）。
        kl_weight: 如果启用变分目标，用于损失的KL散度分量的权重。
            损失计算为：`reconstruction_loss + kl_weight * kld_loss`。
    """

    # 输入/输出结构。
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 架构。
    # 视觉骨干网络。
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer层。
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # 注意：虽然原始ACT实现对 `n_decoder_layers` 使用7，但代码中有一个bug，
    # 意味着只使用第一层。这里我们通过将其设置为1来匹配原始实现。
    # 请参阅此问题 https://github.com/tonyzhaozh/act/issues/25#issue-2258740521。
    n_decoder_layers: int = 1
    # VAE。
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # 推理。
    # 注意：启用时序集成时ACT使用的值为0.01。
    temporal_ensemble_coeff: float | None = None

    # 训练和损失计算。
    dropout: float = 0.1
    kl_weight: float = 10.0

    # 训练预设
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽性）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` 必须是ResNet变体之一。得到 {self.vision_backbone}。"
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "使用时序集成时 `n_action_steps` 必须为1。这是因为策略需要在每一步"
                "查询以计算集成动作。"
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"块大小是每次模型调用的动作步数的上限。得到 `n_action_steps` "
                f"为 {self.n_action_steps}，`chunk_size` 为 {self.chunk_size}。"
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"尚未处理多个观察步。得到 `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("输入中必须至少提供一个图像或环境状态。")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
