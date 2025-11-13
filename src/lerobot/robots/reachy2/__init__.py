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

"""
Reachy2机器人模块

该模块提供了Reachy2类人机器人的配置类和控制接口。
Reachy2是一款具有双臂、颈部和天线的表达性人形机器人平台。
导出包括关节组定义（左臂、右臂、颈部、天线）和速度配置。
"""

from .configuration_reachy2 import Reachy2RobotConfig
from .robot_reachy2 import (
    REACHY2_ANTENNAS_JOINTS,  # Reachy2天线关节
    REACHY2_L_ARM_JOINTS,     # Reachy2左臂关节
    REACHY2_NECK_JOINTS,      # Reachy2颈部关节
    REACHY2_R_ARM_JOINTS,     # Reachy2右臂关节
    REACHY2_VEL,              # Reachy2速度配置
    Reachy2Robot,
)
