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

import abc
from dataclasses import dataclass
from enum import Enum

import draccus


class ColorMode(str, Enum):
    """颜色模式枚举。

    RGB: 红-绿-蓝颜色顺序
    BGR: 蓝-绿-红颜色顺序
    """
    RGB = "rgb"
    BGR = "bgr"


class Cv2Rotation(int, Enum):
    """OpenCV 旋转角度枚举。

    NO_ROTATION: 不旋转（0度）
    ROTATE_90: 顺时针旋转90度
    ROTATE_180: 旋转180度
    ROTATE_270: 逆时针旋转90度（或顺时针270度）
    """
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


@dataclass(kw_only=True)
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    """摄像头配置基类。

    用于定义摄像头的基本配置参数。子类可以扩展此类以添加特定于实现的配置。

    属性：
        fps (int | None): 每秒帧数，如果为 None 则使用摄像头默认值
        width (int | None): 帧宽度（像素），如果为 None 则使用摄像头默认值
        height (int | None): 帧高度（像素），如果为 None 则使用摄像头默认值
    """
    fps: int | None = None
    width: int | None = None
    height: int | None = None

    @property
    def type(self) -> str:
        """返回配置类型名称。"""
        return self.get_choice_name(self.__class__)
