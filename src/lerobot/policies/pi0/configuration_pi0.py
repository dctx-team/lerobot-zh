# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
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
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # 添加空图像。由 pi0_aloha_sim 使用，它在顶部相机之外
    # 添加了左侧和右侧腕部相机。
    empty_cameras: int = 0

    # 将关节和夹爪值从标准 Aloha 空间转换为
    # 用于训练基础模型的 pi 内部运行时使用的空间。
    adapt_to_pi_aloha: bool = False

    # 将关节维度转换为相对于当前状态的增量，然后再传递给模型。
    # 夹爪维度将保持绝对值。
    use_delta_joint_actions_aloha: bool = False

    # 分词器
    tokenizer_max_length: int = 48

    # 投影器
    proj_width: int = 1024

    # 解码
    num_steps: int = 10

    # 注意力工具
    use_cache: bool = True
    attention_implementation: str = "eager"  # 或 fa2, flex

    # 微调设置
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # 训练预设
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # 待办: 添加 EMA

    def __post_init__(self):
        super().__post_init__()

        # 待办(Steven): 在所有策略配置中验证设备和 amp？
        """输入验证（不详尽）。"""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"块大小是每次模型调用的动作步数的上限。得到的 "
                f"`n_action_steps` 为 {self.n_action_steps}，`chunk_size` 为 {self.chunk_size}。"
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"尚未处理多个观测步。得到 `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` 由 pi0 用于 aloha 真实模型。它尚未移植到 LeRobot 中。"
            )

    def validate_features(self) -> None:
        # 待办: 实现 value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("您必须在输入中提供至少一张图像或环境状态。")

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
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
