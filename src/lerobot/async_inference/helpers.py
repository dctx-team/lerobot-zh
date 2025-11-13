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

import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

# 注意：需要加载配置，以便客户端能够实例化策略配置
from lerobot.policies import ACTConfig, DiffusionConfig, PI0Config, SmolVLAConfig, VQBeTConfig  # noqa: F401
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import init_logging

Action = torch.Tensor
ActionChunk = torch.Tensor

# 从机器人接收到的观测
RawObservation = dict[str, torch.Tensor]

# LeRobot 数据集中记录的观测（键名不同）
LeRobotObservation = dict[str, torch.Tensor]

# 准备用于策略推理的观测（图像键已调整大小）
Observation = dict[str, torch.Tensor]


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def validate_robot_cameras_for_policy(
    lerobot_observation_features: dict[str, dict], policy_image_features: dict[str, PolicyFeature]
) -> None:
    image_keys = list(filter(is_image_key, lerobot_observation_features))
    assert set(image_keys) == set(policy_image_features.keys()), (
        f"Policy image features must match robot cameras! Received {list(policy_image_features.keys())} != {image_keys}"
    )


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W)，用于从机器人观测分辨率调整到策略图像分辨率
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # 为 interpolate 添加批次维度：(C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # 插值并移除批次维度：(1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    device: str,
) -> Observation:
    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLA 在观测中呈现自然语言指令
            if "image" in k:
                # 策略期望图像形状为 (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0).to(device)
            else:
                observation[k] = v.to(device)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """最小化预处理，将 int8 图像转换为 [0, 1] 范围内的 float32，并创建内存连续的张量"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> torch.Tensor:
    """从原始观测中提取状态。"""
    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> dict[str, torch.Tensor]:
    """从原始观测中提取图像。"""
    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """从原始观测创建 LeRobot 观测。"""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """将原始 robot_obs 字典中的键与给定策略期望的键（以 policy_image_features 传递）进行匹配。"""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. 获取所有 observation.images.<> 键
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state 的形状期望为 (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # 将图像特征转换为 (C, H, W)，H、W 与策略图像特征匹配。
    # 这会降低图像的分辨率
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    使用 utils.py 中的标准化日志设置获取日志记录器。

    Args:
        name: 日志记录器名称（例如 'policy_server'、'robot_client'）
        log_to_file: 是否同时记录到文件

    Returns:
        已配置的日志记录器实例
    """
    # 如果记录到文件，创建日志目录
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # 初始化标准化日志记录
    init_logging(log_file=log_file, display_pid=False)

    # 返回命名的日志记录器
    return logging.getLogger(name)


@dataclass
class TimedData:
    """带有时间戳和时间步信息的数据对象。

    Args:
        timestamp: 相对于数据创建时间的 Unix 时间戳。
        data: 要包装时间戳的实际数据。
        timestep: 数据的时间步。
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    """用于随时间跟踪 FPS 指标的实用类。"""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """计算平均 FPS 与目标 FPS 的比较"""
        self.total_obs_count += 1

        # 初始化第一次观测时间
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # 计算总体平均 FPS（自开始以来）
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """重置 FPS 跟踪器状态"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """检查两个观测状态在容差阈值下是否相似"""
    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """检查两个观测在容差阈值下是否相似。通过测量两个观测之间关节空间的差异来衡量观测之间的距离。

    注意(fracapuano): 这是一个非常简单的检查，对于当前的用例已经足够。
    下一步是使用（快速的）感知差异度量来比较某些相机视图，
    以超越这种关节空间相似性检查。
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
