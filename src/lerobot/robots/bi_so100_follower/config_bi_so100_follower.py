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


@RobotConfig.register_subclass("bi_so100_follower")
@dataclass
class BiSO100FollowerConfig(RobotConfig):
    """双SO100从动机械臂配置"""

    left_arm_port: str  # 左臂连接端口
    right_arm_port: str  # 右臂连接端口

    # 可选参数
    left_arm_disable_torque_on_disconnect: bool = True  # 断开连接时禁用左臂扭矩
    left_arm_max_relative_target: float | dict[str, float] | None = None  # 左臂最大相对目标位置限制
    left_arm_use_degrees: bool = False  # 左臂是否使用角度制（向后兼容）
    right_arm_disable_torque_on_disconnect: bool = True  # 断开连接时禁用右臂扭矩
    right_arm_max_relative_target: float | dict[str, float] | None = None  # 右臂最大相对目标位置限制
    right_arm_use_degrees: bool = False  # 右臂是否使用角度制（向后兼容）

    # 摄像头（两个手臂共享）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)  # 相机配置字典
