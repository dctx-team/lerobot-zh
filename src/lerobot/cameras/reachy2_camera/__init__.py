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
"""
Reachy2 相机模块

本模块提供了对 Reachy2 机器人相机系统的支持，包括相机配置和相机接口实现。
Reachy2 是一款人形机器人平台，本模块允许访问和控制其相机系统。
"""

from .configuration_reachy2_camera import Reachy2CameraConfig
from .reachy2_camera import Reachy2Camera
