# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # 输入/输出结构。
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 较短的状态和动作向量将被填充
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 图像预处理
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # 添加空图像。由 smolvla_aloha_sim 使用，除了顶部摄像头外，
    # 还添加空的左右手腕摄像头。
    empty_cameras: int = 0

    # 将关节和夹爪值从标准 Aloha 空间转换为
    # 用于训练基础模型的 pi 内部运行时使用的空间。
    adapt_to_pi_aloha: bool = False

    # 在传递给模型之前，将关节维度转换为相对于当前状态的增量。
    # 夹爪维度将保持绝对值。
    use_delta_joint_actions_aloha: bool = False

    # 分词器
    tokenizer_max_length: int = 48

    # 解码
    num_steps: int = 10

    # 注意力工具
    use_cache: bool = True

    # 微调设置
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # 选择 VLM 骨干网络。
    load_vlm_weights: bool = False  # 在从头训练专家时设置为 True。从预训练的 SmolVLA 权重初始化时为 True

    add_image_special_tokens: bool = False  # 是否在图像特征周围使用特殊图像令牌。

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # 小于或等于 0 是默认值，此时动作专家与 VLM 具有相同数量的层。否则专家层数较少。
    num_vlm_layers: int = 16  # VLM 中使用的层数（前 num_vlm_layers 层）
    self_attn_every_n_layers: int = 2  # 每 self_attn_every_n_layers 层交错一个自注意力层
    expert_width_multiplier: float = 0.75  # 动作专家的隐藏层大小（相对于 VLM）

    min_period: float = 4e-3  # 正弦-余弦位置编码中使用的时间步的敏感度范围
    max_period: float = 4.0

    def __post_init__(self):
        super().__post_init__()

        """输入验证（不完全）。"""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
