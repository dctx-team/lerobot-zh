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
OpenCV 相机模块

本模块提供了基于 OpenCV 的通用相机支持，包括相机配置和相机接口实现。
可以用于访问各种通过 OpenCV 兼容的 USB 相机、网络摄像头等常见相机设备。
"""

from .camera_opencv import OpenCVCamera
from .configuration_opencv import OpenCVCameraConfig
