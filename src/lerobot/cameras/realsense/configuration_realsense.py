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

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Intel RealSense 相机的配置类。

    该类为 Intel RealSense 相机提供专门的配置选项，
    包括对深度感知和通过序列号或名称进行设备识别的支持。

    Intel RealSense D405 的示例配置：
    ```python
    # 基本配置
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # 高级配置
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # 带深度感知
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # 带 90° 旋转
    ```

    属性：
        fps: 颜色流请求的每秒帧数。
        width: 颜色流请求的帧宽度（像素）。
        height: 颜色流请求的帧高度（像素）。
        serial_number_or_name: 用于识别相机的唯一序列号或人类可读名称。
        color_mode: 图像输出的颜色模式（RGB 或 BGR）。默认为 RGB。
        use_depth: 是否启用深度流。默认为 False。
        rotation: 图像旋转设置（0°、90°、180° 或 270°）。默认为无旋转。
        warmup_s: 从 connect 返回前读取帧的时间（秒）

    注意：
        - 必须指定名称或序列号。
        - 深度流配置（如果启用）将使用与颜色流相同的 FPS。
        - 实际的分辨率和 FPS 可能会由相机调整为最近的支持模式。
        - 对于 `fps`、`width` 和 `height`，要么全部设置，要么全部不设置。
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` 预期为 {ColorMode.RGB.value} 或 {ColorMode.BGR.value}，但提供了 {self.color_mode}。"
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` 预期在 {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)} 中，但提供了 {self.rotation}。"
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "对于 `fps`、`width` 和 `height`，要么全部设置，要么全部不设置。"
            )
