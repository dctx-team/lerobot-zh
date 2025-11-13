#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("koch_leader")
@dataclass
class KochLeaderConfig(TeleoperatorConfig):
    # 连接到机械臂的端口
    port: str

    # 将机械臂设置为扭矩模式，夹爪电机设置为此值。这使得可以挤压夹爪，
    # 并使其自动弹回到打开位置。
    gripper_open_pos: float = 50.0
