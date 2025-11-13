#!/usr/bin/env python

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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so100_follower")
@dataclass
class SO100FollowerConfig(RobotConfig):
    """SO100从动机械臂配置"""

    # 连接到机械臂的端口
    port: str

    disable_torque_on_disconnect: bool = True  # 断开连接时禁用扭矩

    # `max_relative_target` 出于安全目的限制相对位置目标向量的幅度。
    # 将其设置为正标量可为所有电机使用相同的值，或设置为字典将电机名称映射到该电机的 max_relative_target 值。
    max_relative_target: float | dict[str, float] | None = None

    # 相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 设置为 `True` 以向后兼容之前的策略/数据集
    use_degrees: bool = False
