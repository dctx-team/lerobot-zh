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


@RobotConfig.register_subclass("hope_jr_hand")
@dataclass
class HopeJrHandConfig(RobotConfig):
    port: str  # 连接到灵巧手的端口
    side: str  # "left" / "right" (左手/右手)

    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@RobotConfig.register_subclass("hope_jr_arm")
@dataclass
class HopeJrArmConfig(RobotConfig):
    port: str  # 连接到机械臂的端口
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` 为了安全起见，限制相对位置目标向量的幅度。
    # 将此设置为正标量值以对所有电机使用相同的值，或设置为将电机名称映射到该电机的 max_relative_target 值的字典。
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
