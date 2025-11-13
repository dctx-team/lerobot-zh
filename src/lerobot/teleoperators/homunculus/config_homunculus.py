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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("homunculus_glove")
@dataclass
class HomunculusGloveConfig(TeleoperatorConfig):
    """Homunculus 手套配置。

    该配置类用于设置 Homunculus 遥操作手套的连接参数。
    """

    port: str  # 连接手套的端口
    side: str  # "left" / "right" (左/右)
    baud_rate: int = 115_200  # 波特率，默认 115200

    def __post_init__(self):
        """初始化后验证参数。"""
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@TeleoperatorConfig.register_subclass("homunculus_arm")
@dataclass
class HomunculusArmConfig(TeleoperatorConfig):
    """Homunculus 机械臂配置。

    该配置类用于设置 Homunculus 遥操作机械臂的连接参数。
    """

    port: str  # 连接机械臂的端口
    baud_rate: int = 115_200  # 波特率，默认 115200
