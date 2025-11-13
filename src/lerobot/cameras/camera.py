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
from typing import Any

import numpy as np

from .configs import CameraConfig, ColorMode


class Camera(abc.ABC):
    """摄像头实现的基类。

    为不同后端的摄像头操作定义了标准接口。
    子类必须实现所有抽象方法。

    管理基本的摄像头属性（FPS、分辨率）和核心操作：
    - 连接/断开连接
    - 帧捕获（同步/异步）

    属性：
        fps (int | None): 配置的每秒帧数
        width (int | None): 帧宽度（像素）
        height (int | None): 帧高度（像素）

    示例：
        class MyCamera(Camera):
            def __init__(self, config): ...
            @property
            def is_connected(self) -> bool: ...
            def connect(self, warmup=True): ...
            # 以及其他必需的方法
    """

    def __init__(self, config: CameraConfig):
        """使用给定的配置初始化摄像头。

        参数：
            config: 包含 FPS 和分辨率的摄像头配置。
        """
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """检查摄像头当前是否已连接。

        返回：
            bool: 如果摄像头已连接并准备好捕获帧则返回 True，
                  否则返回 False。
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_cameras() -> list[dict[str, Any]]:
        """检测连接到系统的可用摄像头。

        返回：
            List[Dict[str, Any]]: 字典列表，
            其中每个字典包含有关检测到的摄像头的信息。
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """建立与摄像头的连接。

        参数：
            warmup: 如果为 True（默认值），则在返回之前捕获一个预热帧。这对于
                   需要时间来调整捕获设置的摄像头很有用。
                   如果为 False，则跳过预热帧。
        """
        pass

    @abc.abstractmethod
    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """从摄像头捕获并返回单个帧。

        参数：
            color_mode: 输出帧所需的颜色模式。如果为 None，
                        则使用摄像头的默认颜色模式。

        返回：
            np.ndarray: 捕获的帧，作为 numpy 数组。
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> np.ndarray:
        """异步捕获并返回摄像头的单个帧。

        参数：
            timeout_ms: 等待帧的最大时间（毫秒）。
                        默认为实现特定的超时时间。

        返回：
            np.ndarray: 捕获的帧，作为 numpy 数组。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与摄像头的连接并释放资源。"""
        pass
