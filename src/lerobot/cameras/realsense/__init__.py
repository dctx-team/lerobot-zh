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
Intel RealSense 相机模块

本模块提供了对 Intel RealSense 深度相机的支持，包括相机配置和相机接口实现。
RealSense 相机可以同时捕获 RGB 图像和深度信息，适用于机器人视觉和 3D 感知任务。
"""

from .camera_realsense import RealSenseCamera
from .configuration_realsense import RealSenseCameraConfig
