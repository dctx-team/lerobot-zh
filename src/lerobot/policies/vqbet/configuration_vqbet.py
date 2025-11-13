#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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
from lerobot.optim.schedulers import VQBeTSchedulerConfig


@PreTrainedConfig.register_subclass("vqbet")
@dataclass
class VQBeTConfig(PreTrainedConfig):
    """VQ-BeT的配置类。

    默认配置为使用PushT训练，提供本体感受和单摄像头观测。

    您最有可能需要更改的参数是依赖于环境/传感器的参数。
    它们是：`input_shapes`和`output_shapes`。

    关于输入和输出的注意事项：
        - "observation.state"是必需的输入键。
        - 至少需要一个以"observation.image"开头的键作为输入。
        - 如果有多个以"observation.image"开头的键，它们被视为多个摄像头视图。
          目前我们只支持所有图像具有相同的形状。
        - "action"是必需的输出键。

    参数:
        n_obs_steps: 传递给策略的环境步骤的观测数量（取当前步骤和向后的额外步骤）。
        n_action_pred_token: VQ-BeT预测的当前标记和未来标记的总数。
        action_chunk_size: 每个动作预测标记的动作块大小。
        input_shapes: 定义策略输入数据形状的字典。
            键表示输入数据名称，值是表示相应数据维度的列表。
            例如，"observation.image"指的是来自摄像头的输入，维度为[3, 96, 96]，
            表示它有三个颜色通道和96x96分辨率。
            重要的是，形状不包括批次维度或时间维度。
        output_shapes: 定义策略输出数据形状的字典。
            键表示输出数据名称，值是表示相应数据维度的列表。
            例如，"action"指的是输出形状为[14]，表示14维动作。
            重要的是，形状不包括批次维度或时间维度。
        input_normalization_modes: 一个字典，键表示模态（例如"observation.state"），
            值指定要应用的归一化模式。两种可用模式是"mean_std"
            （减去均值并除以标准差）和"min_max"（在[-1, 1]范围内重新缩放）。
        output_normalization_modes: 与`normalize_input_modes`类似的字典，但用于反归一化到
            原始尺度。请注意，这也用于归一化训练目标。
        vision_backbone: 用于编码图像的torchvision resnet骨干网络名称。
        crop_shape: (H, W)形状，作为视觉骨干网络的预处理步骤将图像裁剪到该形状。必须
            适合图像大小。如果为None，则不进行裁剪。
        crop_is_random: 裁剪在训练时是否应该是随机的（在评估模式下始终是中心裁剪）。
        pretrained_backbone_weights: 来自torchvision的预训练权重以初始化骨干网络。
            `None`表示没有预训练权重。
        use_group_norm: 是否在骨干网络中用组归一化替换批归一化。
            组大小设置为约16（准确地说，feature_dim // 16）。
        spatial_softmax_num_keypoints: SpatialSoftmax的关键点数量。
        n_vqvae_training_steps: 训练残差VQ的优化步数。
        vqvae_n_embed: RVQ字典中嵌入向量的数量（每层）。
        vqvae_embedding_dim: RVQ字典中每个嵌入向量的维度。
        vqvae_enc_hidden_dim: 残差VQ-VAE的编码器/解码器部分的隐藏维度大小
        gpt_block_size: minGPT的最大块大小（应大于输入标记数量）
        gpt_input_dim: GPT输出输入的大小。这也用作观测特征的维度。
        gpt_output_dim: GPT输出维度的大小。这也用作偏移/码预测头的输入维度。
        gpt_n_layer: GPT的层数
        gpt_n_head: GPT的头数
        gpt_hidden_dim: GPT的隐藏维度大小
        dropout: GPT的Dropout率
        mlp_hidden_dim: VQ-BeT的偏移头/码预测头部分的隐藏维度大小
        offset_loss_weight: 乘以偏移损失的常数
        primary_code_loss_weight: 乘以主代码预测损失的常数
        secondary_code_loss_weight: 乘以次代码预测损失的常数
        bet_softmax_temperature: 使用VQ-BeT推演时代码的采样温度
        sequentially_select: 是否顺序选择主/次代码（选择主代码，
            然后选择次代码），还是同时选择。
    """

    # 输入/输出结构。
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 架构/建模。
    # 视觉骨干网络。
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    # VQ-VAE
    n_vqvae_training_steps: int = 20000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 256
    vqvae_enc_hidden_dim: int = 128
    # VQ-BeT
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1
    mlp_hidden_dim: int = 1024
    offset_loss_weight: float = 10000.0
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_vqvae_lr: float = 1e-3
    optimizer_vqvae_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone`必须是ResNet变体之一。得到{self.vision_backbone}。"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> VQBeTSchedulerConfig:
        return VQBeTSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_vqvae_training_steps=self.n_vqvae_training_steps,
        )

    def validate_features(self) -> None:
        # 注意：此检查之前在VQBeTRgbEncoder内部以以下形式执行
        # assert len(image_keys) == 1
        if not len(self.image_features) == 1:
            raise ValueError("您必须在输入中仅提供一张图像。")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape`应该适合图像形状。得到{self.crop_shape} "
                        f"用于`crop_shape`，以及{image_ft.shape}用于 "
                        f"`{key}`。"
                    )

        # 检查所有输入图像是否具有相同的形状。
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}`与`{first_image_key}`不匹配，但我们期望所有图像形状都匹配。"
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.n_action_pred_token + self.action_chunk_size - 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
