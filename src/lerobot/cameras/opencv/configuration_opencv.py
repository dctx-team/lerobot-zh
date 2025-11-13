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
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """基于 OpenCV 的相机设备或视频文件的配置类。

    此类为通过 OpenCV 访问的相机提供配置选项，
    支持物理相机设备和视频文件。它包含用于
    分辨率、帧率、颜色模式和图像旋转的设置。

    配置示例：
    ```python
    # 基本配置
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # 高级配置
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)     # 带 90° 旋转
    ```

    属性：
        index_or_path: 表示相机设备索引的整数，
                      或指向视频文件的 Path 对象。
        fps: 彩色流的请求每秒帧数。
        width: 彩色流的请求帧宽度（像素）。
        height: 彩色流的请求帧高度（像素）。
        color_mode: 图像输出的颜色模式（RGB 或 BGR）。默认为 RGB。
        rotation: 图像旋转设置（0°、90°、180° 或 270°）。默认为无旋转。
        warmup_s: 从 connect 返回前读取帧的时间（秒）

    注意：
        - 目前仅支持 3 通道彩色输出（RGB/BGR）。
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )
