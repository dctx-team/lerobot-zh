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
提供 Reachy2Camera 类，用于使用 Reachy 2 的 CameraManager 从 Reachy 2 相机捕获帧。
"""

import logging
import os
import platform
import time
from threading import Event, Lock, Thread
from typing import Any

# 在导入 cv2 之前修复 Windows 的 MSMF 硬件转换兼容性
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
from reachy2_sdk.media.camera import CameraView
from reachy2_sdk.media.camera_manager import CameraManager

from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_reachy2_camera import ColorMode, Reachy2CameraConfig

logger = logging.getLogger(__name__)


class Reachy2Camera(Camera):
    """
    使用 Reachy 2 CameraManager 管理 Reachy 2 相机。

    此类提供了一个高级接口来连接、配置和读取来自 Reachy 2 相机的帧。
    它支持同步和异步帧读取。

    Reachy2Camera 实例需要在配置中指定相机名称（例如 "teleop"）和图像类型（例如 "left"）。

    除非在配置中覆盖，否则将使用相机的默认设置（FPS、分辨率、颜色模式）。
    """

    def __init__(self, config: Reachy2CameraConfig):
        """
        初始化 Reachy2Camera 实例。

        参数:
            config: 相机的配置设置。
        """
        super().__init__(config)

        self.config = config

        self.fps = config.fps
        self.color_mode = config.color_mode

        self.cam_manager: CameraManager | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name}, {self.config.image_type})"

    @property
    def is_connected(self) -> bool:
        """检查相机当前是否已连接并打开。"""
        if self.config.name == "teleop":
            return self.cam_manager._grpc_connected and self.cam_manager.teleop if self.cam_manager else False
        elif self.config.name == "depth":
            return self.cam_manager._grpc_connected and self.cam_manager.depth if self.cam_manager else False
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

    def connect(self, warmup: bool = True):
        """
        按照配置中指定的方式连接到 Reachy2 CameraManager。
        """
        self.cam_manager = CameraManager(host=self.config.ip_address, port=self.config.port)
        self.cam_manager.initialize_cameras()

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras(ip_address: str = "localhost", port: int = 50065) -> list[dict[str, Any]]:
        """
        检测可用的 Reachy 2 相机。

        返回:
            List[Dict[str, Any]]: 字典列表，
            其中每个字典包含 'name'、'stereo'
            以及默认配置属性（width、height、fps）。
        """
        initialized_cameras = []
        camera_manager = CameraManager(host=ip_address, port=port)

        for camera in [camera_manager.teleop, camera_manager.depth]:
            if camera is None:
                continue

            height, width, _, _, _, _, _ = camera.get_parameters()

            camera_info = {
                "name": camera._cam_info.name,
                "stereo": camera._cam_info.stereo,
                "default_profile": {
                    "width": width,
                    "height": height,
                    "fps": 30,
                },
            }
            initialized_cameras.append(camera_info)

        camera_manager.disconnect()
        return initialized_cameras

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        从相机同步读取单个帧。

        这是一个阻塞调用。

        参数:
            color_mode (Optional[ColorMode]): 如果指定，则覆盖此读取操作的默认
                颜色模式（`self.color_mode`）（例如，即使默认为 BGR 也请求 RGB）。

        返回:
            np.ndarray: 捕获的帧，格式为 NumPy 数组
                       (height, width, channels)，使用指定或默认的
                       颜色模式并应用任何配置的旋转。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frame = None

        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        else:
            if self.config.name == "teleop" and hasattr(self.cam_manager, "teleop"):
                if self.config.image_type == "left":
                    frame = self.cam_manager.teleop.get_frame(CameraView.LEFT, size=(640, 480))[0]
                elif self.config.image_type == "right":
                    frame = self.cam_manager.teleop.get_frame(CameraView.RIGHT, size=(640, 480))[0]
            elif self.config.name == "depth" and hasattr(self.cam_manager, "depth"):
                if self.config.image_type == "depth":
                    frame = self.cam_manager.depth.get_depth_frame()[0]
                elif self.config.image_type == "rgb":
                    frame = self.cam_manager.depth.get_frame(size=(640, 480))[0]

            if frame is None:
                return np.empty((0, 0, 3), dtype=np.uint8)

            if self.config.color_mode == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _read_loop(self):
        """
        后台线程运行的内部循环，用于异步读取。

        在每次迭代中：
        1. 读取彩色帧
        2. 将结果存储在 latest_frame 中（线程安全）
        3. 设置 new_frame_event 以通知监听器

        遇到 DeviceNotConnectedError 时停止，记录其他错误并继续。
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """如果后台读取线程未运行，则启动或重新启动它。"""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """向后台读取线程发出停止信号并等待其加入。"""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        异步读取最新可用帧。

        此方法检索后台读取线程捕获的最新帧。
        它不会直接阻塞等待相机硬件，但可能会等待最多 timeout_ms 以便后台线程提供帧。

        参数:
            timeout_ms (float): 等待帧变为可用的最大时间（毫秒）。
                默认为 200ms（0.2 秒）。

        返回:
            np.ndarray: 最新捕获的帧，格式为 NumPy 数组
                       (height, width, channels)，根据配置进行处理。

        异常:
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定的超时时间内没有帧可用。
            RuntimeError: 如果发生意外错误。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """
        停止后台读取线程（如果正在运行）。

        异常:
            DeviceNotConnectedError: 如果相机已经断开连接。
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.cam_manager is not None:
            self.cam_manager.disconnect()

        logger.info(f"{self} disconnected.")
