# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def is_image_feature(key: str) -> bool:
    """检查特征键是否代表图像特征。

    参数:
        key: 要检查的特征键

    返回:
        如果该键代表图像特征则返回 True，否则返回 False
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    """actor 和 learner 的并发配置。
    可能的取值:
    - "threads": 为 actor 和 learner 使用线程。
    - "processes": 为 actor 和 learner 使用进程。
    """

    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """Soft Actor-Critic (SAC) 配置。

    SAC 是一种基于最大熵强化学习框架的离线策略 actor-critic 深度强化学习算法。
    它使用从环境中收集的经验同时学习策略和 Q 函数。

    此配置类包含定义 SAC 智能体所需的所有参数，包括网络架构、优化设置和
    算法特定的超参数。
    """

    # 特征类型到归一化模式的映射
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 用于归一化不同类型输入的统计信息
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # 架构细节
    # 运行模型的设备（例如，"cuda"、"cpu"）
    device: str = "cpu"
    # 存储模型的设备
    storage_device: str = "cpu"
    # 视觉编码器模型的名称（对于 hil serl resnet10，设置为 "helper2424/resnet10"）
    vision_encoder_name: str | None = None
    # 是否在训练期间冻结视觉编码器
    freeze_vision_encoder: bool = True
    # 图像编码器的隐藏维度大小
    image_encoder_hidden_dim: int = 32
    # 是否为 actor 和 critic 使用共享编码器
    shared_encoder: bool = True
    # 离散动作的数量，例如用于夹持器动作
    num_discrete_actions: int | None = None
    # 图像嵌入池化的维度
    image_embedding_pooling_dim: int = 8

    # 训练参数
    # 在线训练的步数
    online_steps: int = 1000000
    # 在线环境的种子
    online_env_seed: int = 10000
    # 在线重放缓冲区的容量
    online_buffer_capacity: int = 100000
    # 离线重放缓冲区的容量
    offline_buffer_capacity: int = 100000
    # 是否对缓冲区使用异步预取
    async_prefetch: bool = False
    # 开始学习之前的步数
    online_step_before_learning: int = 100
    # 策略更新频率
    policy_update_freq: int = 1

    # SAC 算法参数
    # SAC 算法的折扣因子
    discount: float = 0.99
    # 初始温度值
    temperature_init: float = 1.0
    # 集成中的 critic 数量
    num_critics: int = 2
    # 用于训练的子采样 critic 数量
    num_subsample_critics: int | None = None
    # critic 网络的学习率
    critic_lr: float = 3e-4
    # actor 网络的学习率
    actor_lr: float = 3e-4
    # 温度参数的学习率
    temperature_lr: float = 3e-4
    # critic 目标更新权重
    critic_target_update_weight: float = 0.005
    # UTD 算法的更新数据比（如果要启用 utd_ratio，需要将其设置为 >1）
    utd_ratio: int = 1
    # 状态编码器的隐藏维度大小
    state_encoder_hidden_dim: int = 256
    # 潜在空间的维度
    latent_dim: int = 256
    # SAC 算法的目标熵
    target_entropy: float | None = None
    # 是否为 SAC 算法使用备份熵
    use_backup_entropy: bool = True
    # SAC 算法的梯度裁剪范数
    grad_clip_norm: float = 40.0

    # 网络配置
    # critic 网络架构的配置
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # actor 网络架构的配置
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # 策略参数的配置
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # 离散 critic 网络的配置
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # actor-learner 架构的配置
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # 并发设置的配置（你可以为 actor 和 learner 使用线程或进程）
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # 优化
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # SAC 配置特定的任何验证

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation"
                " (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC 通常一次预测一个动作

    @property
    def reward_delta_indices(self) -> None:
        return None
