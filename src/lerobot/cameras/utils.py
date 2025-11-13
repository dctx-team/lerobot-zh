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

import platform
from pathlib import Path
from typing import TypeAlias

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

IndexOrPath: TypeAlias = int | Path


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    """从配置字典创建相机实例字典。"""
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        else:
            raise ValueError(f"相机类型 '{cfg.type}' 无效。")

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    """将 Cv2Rotation 枚举转换为 OpenCV 旋转常量。"""
    import cv2

    if rotation == Cv2Rotation.ROTATE_90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation == Cv2Rotation.ROTATE_180:
        return cv2.ROTATE_180
    elif rotation == Cv2Rotation.ROTATE_270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return None


def get_cv2_backend() -> int:
    """根据操作系统返回适当的 OpenCV 后端。"""
    import cv2

    if platform.system() == "Windows":
        return cv2.CAP_MSMF  # 在 Windows 上使用 MSMF 而不是 AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux 和其他系统
        return cv2.CAP_ANY
