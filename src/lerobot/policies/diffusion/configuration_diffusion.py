#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
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
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """扩散策略的配置类。

    默认配置用于使用 PushT 进行训练，提供本体感知和单相机观测。

    您最可能需要更改的参数是那些取决于环境/传感器的参数。
    它们是：`input_shapes` 和 `output_shapes`。

    关于输入和输出的注意事项：
        - "observation.state" 是必需的输入键。
        - 以下之一：
            - 至少需要一个以 "observation.image" 开头的键作为输入。
              和/或
            - 需要 "observation.environment_state" 键作为输入。
        - 如果有多个以 "observation.image" 开头的键，它们将被视为多个相机视图。
          目前我们仅支持所有图像具有相同的形状。
        - "action" 是必需的输出键。

    参数:
        n_obs_steps: 传递给策略的环境步骤的观测数量（获取当前步骤和追溯的额外步骤）。
        horizon: 扩散模型动作预测大小，详见 `DiffusionPolicy.select_action`。
        n_action_steps: 策略一次调用在环境中运行的动作步骤数。
            详见 `DiffusionPolicy.select_action`。
        input_shapes: 定义策略输入数据形状的字典。键表示输入数据名称，值是表示相应数据维度的列表。
            例如，"observation.image" 指的是来自相机的输入，维度为 [3, 96, 96]，
            表示它有三个颜色通道和 96x96 分辨率。重要的是，`input_shapes` 不包括批次维度或时间维度。
        output_shapes: 定义策略输出数据形状的字典。键表示输出数据名称，值是表示相应数据维度的列表。
            例如，"action" 指的是输出形状 [14]，表示 14 维动作。
            重要的是，`output_shapes` 不包括批次维度或时间维度。
        input_normalization_modes: 键表示模态（例如 "observation.state"）的字典，
            值指定要应用的归一化模式。两种可用模式是 "mean_std"，它减去均值并除以标准差，
            以及 "min_max"，它在 [-1, 1] 范围内重新缩放。
        output_normalization_modes: 与 `normalize_input_modes` 类似的字典，但用于反归一化到原始尺度。
            注意，这也用于归一化训练目标。
        vision_backbone: 用于编码图像的 torchvision resnet 主干网络的名称。
        crop_shape: (H, W) 形状，作为视觉主干网络的预处理步骤裁剪图像。
            必须适合图像尺寸。如果为 None，则不进行裁剪。
        crop_is_random: 训练时裁剪是否应该是随机的（评估模式下始终是中心裁剪）。
        pretrained_backbone_weights: 来自 torchvision 的预训练权重以初始化主干网络。
            `None` 表示没有预训练权重。
        use_group_norm: 是否在主干网络中用组归一化替换批归一化。
            组大小设置为约 16（准确地说，feature_dim // 16）。
        spatial_softmax_num_keypoints: SpatialSoftmax 的关键点数量。
        use_separate_rgb_encoders_per_camera: 是否为每个相机视图使用单独的 RGB 编码器。
        down_dims: 扩散建模 Unet 中每个时间下采样阶段的特征维度。
            您可以提供可变数量的维度，从而也控制下采样的程度。
        kernel_size: 扩散建模 Unet 的卷积核大小。
        n_groups: Unet 卷积块的组归一化中使用的组数。
        diffusion_step_embed_dim: Unet 通过一个小的非线性网络以扩散时间步为条件。
            这是该网络的输出维度，即嵌入维度。
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) 用于 Unet 条件生成。
            默认使用偏置调制，此参数指示是否也使用缩放调制。
        noise_scheduler_type: 要使用的噪声调度器的名称。支持的选项：["DDPM", "DDIM"]。
        num_train_timesteps: 前向扩散调度的扩散步数。
        beta_schedule: 扩散 beta 调度的名称，按照 Hugging Face diffusers 的 DDPMScheduler。
        beta_start: 第一个前向扩散步骤的 Beta 值。
        beta_end: 最后一个前向扩散步骤的 Beta 值。
        prediction_type: 扩散建模 Unet 做出的预测类型。从 "epsilon" 或 "sample" 中选择。
            从潜在变量建模的角度来看，它们具有等效的结果，但 "epsilon" 在许多深度神经网络设置中
            已被证明效果更好。
        clip_sample: 在推理时是否对每个去噪步骤将样本裁剪到 [-`clip_sample_range`, +`clip_sample_range`]。
            警告：您需要确保动作空间已归一化以适应此范围。
        clip_sample_range: 如上所述的裁剪范围的幅度。
        num_inference_steps: 在推理时使用的反向扩散步骤数（步骤均匀间隔）。
            如果未提供，则默认与 `num_train_timesteps` 相同。
        do_mask_loss_for_padding: 当存在复制填充的动作时是否屏蔽损失。详见 `LeRobotDataset` 和
            `load_previous_and_future_frames`。注意，这默认为 False，因为原始的扩散策略实现也是如此。
    """

    # 输入/输出结构。
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 原始实现不对最后 7 步进行帧采样，
    # 这避免了过度填充并改善了训练结果。
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # 架构/建模。
    # 视觉主干网络。
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet。
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # 噪声调度器。
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 推理
    num_inference_steps: int | None = None

    # 损失计算
    do_mask_loss_for_padding: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽性）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` 必须是 ResNet 变体之一。得到 {self.vision_backbone}。"
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` 必须是 {supported_prediction_types} 之一。得到 {self.prediction_type}。"
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` 必须是 {supported_noise_schedulers} 之一。"
                f"得到 {self.noise_scheduler_type}。"
            )

        # 检查 horizon 大小和 U-Net 下采样是否兼容。
        # U-Net 在每个阶段下采样 2 倍。
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "horizon 应该是下采样因子的整数倍（由 `len(down_dims)` 决定）。"
                f"得到 {self.horizon=} 和 {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("您必须在输入中至少提供一张图像或环境状态。")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` 应该适合图像的形状。得到 {self.crop_shape} "
                        f"作为 `crop_shape`，得到 {image_ft.shape} 作为 "
                        f"`{key}`。"
                    )

        # 检查所有输入图像是否具有相同的形状。
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` 与 `{first_image_key}` 不匹配，但我们期望所有图像形状都匹配。"
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
