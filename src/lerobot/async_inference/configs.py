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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# CLI使用的聚合函数注册表
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """从注册表中按名称获取聚合函数。"""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """PolicyServer的配置类。

    该类定义了PolicyServer的所有可配置参数，
    包括网络设置和动作分块规范。
    """

    # 网络配置
    host: str = field(default="localhost", metadata={"help": "服务器绑定的主机地址"})
    port: int = field(default=8080, metadata={"help": "服务器绑定的端口号"})

    # 时序配置
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "每秒帧数"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "目标推理延迟（秒）"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "观测队列超时时间（秒）"}
    )

    def __post_init__(self):
        """初始化后验证配置。"""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """从字典创建PolicyServerConfig实例。"""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """环境时间步长，以秒为单位"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """将配置转换为字典。"""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """RobotClient的配置类。

    该类定义了RobotClient的所有可配置参数，
    包括网络连接、策略设置和控制行为。
    """

    # 策略配置
    policy_type: str = field(metadata={"help": "要使用的策略类型"})
    pretrained_name_or_path: str = field(metadata={"help": "预训练模型名称或路径"})

    # 机器人配置（用于CLI使用 - 机器人实例将从此创建）
    robot: RobotConfig = field(metadata={"help": "机器人配置"})

    # 策略通常最多输出K个动作，但我们可以使用较少的动作以避免浪费带宽（因为动作
    # 无论如何都会在客户端聚合，这取决于`chunk_size_threshold`的值）
    actions_per_chunk: int = field(metadata={"help": "每个块中的动作数量"})

    # 机器人执行的任务指令（例如，'fold my tshirt'）
    task: str = field(default="", metadata={"help": "机器人执行的任务指令"})

    # 网络配置
    server_address: str = field(default="localhost:8080", metadata={"help": "要连接的服务器地址"})

    # 设备配置
    policy_device: str = field(default="cpu", metadata={"help": "策略推理的设备"})

    # 控制行为配置
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "块大小控制的阈值"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "每秒帧数"})

    # 聚合函数配置（CLI兼容）
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"要使用的聚合函数名称。选项：{list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # 调试配置
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "可视化动作队列大小"}
    )

    # 验证配置
    verify_robot_cameras: bool = field(
        default=True, metadata={"help": "验证机器人相机是否与策略相机匹配"}
    )

    @property
    def environment_dt(self) -> float:
        """环境时间步长，以秒为单位"""
        return 1 / self.fps

    def __post_init__(self):
        """初始化后验证配置。"""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """从字典创建RobotClientConfig实例。"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """将配置转换为字典。"""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
