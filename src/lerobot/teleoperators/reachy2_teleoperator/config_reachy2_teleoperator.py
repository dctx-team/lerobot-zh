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


@TeleoperatorConfig.register_subclass("reachy2_teleoperator")
@dataclass
class Reachy2TeleoperatorConfig(TeleoperatorConfig):
    # 用作遥操作器的 Reachy 2 机器人的 IP 地址
    ip_address: str | None = "localhost"

    # 是否使用关节的当前位置作为动作
    # 如果为 False，将使用关节的目标位置
    use_present_position: bool = False

    # 使用机器人的哪些部分
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    def __post_init__(self):
        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "未使用任何 Reachy2Teleoperator 部分。\n"
                "必须将机器人的至少一个部分设置为 True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
