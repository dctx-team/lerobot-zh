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
提供使用 OpenCV 从相机捕获帧的 OpenCVCamera 类。
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

# 在导入 cv2 之前修复 Windows 的 MSMF 硬件转换兼容性问题
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# 注意(Steven): OpenCV 设备索引的最大值取决于您的操作系统。例如，
# 如果您有 3 个相机，它们应该关联到索引 0、1 和 2。这在
# MacOS 上是这样的。但是，在 Ubuntu 上，索引是不同的，比如 6、16、23。
# 当您更换 USB 端口或重启计算机时，操作系统可能会
# 将相同的相机视为新设备。因此我们选择一个较高的上限来搜索索引。
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


class OpenCVCamera(Camera):
    """
    使用 OpenCV 管理相机交互以实现高效的帧录制。

    此类提供了一个高级接口，用于连接、配置和读取
    兼容 OpenCV 的 VideoCapture 的相机的帧。它同时支持
    同步和异步帧读取。

    OpenCVCamera 实例需要一个相机索引（例如 0）或设备路径
    （例如 Linux 上的 '/dev/video0'）。相机索引可能在重启或
    端口更改时不稳定，尤其是在 Linux 上。使用提供的实用脚本来查找
    可用的相机索引或路径：
    ```bash
    lerobot-find-cameras opencv
    ```

    除非在配置中覆盖，否则使用相机的默认设置（FPS、分辨率、颜色模式）。

    示例：
        ```python
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig, ColorMode, Cv2Rotation

        # 使用相机索引 0 的基本用法
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        camera.connect()

        # 同步读取 1 帧
        color_image = camera.read()
        print(color_image.shape)

        # 异步读取 1 帧
        async_image = camera.async_read()

        # 完成后，正确断开相机连接
        camera.disconnect()

        # 使用自定义设置的示例
        custom_config = OpenCVCameraConfig(
            index_or_path='/dev/video0', # 或使用索引
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.ROTATE_90
        )
        custom_camera = OpenCVCamera(custom_config)
        # ... 连接、读取、断开连接 ...
        ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        初始化 OpenCVCamera 实例。

        参数：
            config: 相机的配置设置。
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.videocapture: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """检查相机当前是否已连接并打开。"""
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True):
        """
        连接到配置中指定的 OpenCV 相机。

        初始化 OpenCV VideoCapture 对象，设置所需的相机属性
        (FPS、宽度、高度)，并执行初始检查。

        异常：
            DeviceAlreadyConnectedError: 如果相机已连接。
            ConnectionError: 如果未找到指定的相机索引/路径，或找到相机但无法打开。
            RuntimeError: 如果相机打开但未能应用请求的 FPS/分辨率设置。
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # 为 OpenCV 操作使用 1 个线程以避免潜在的冲突或
        # 多线程应用中的阻塞，尤其是在数据收集期间。
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        self._configure_capture_settings()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _configure_capture_settings(self) -> None:
        """
        将指定的 FPS、宽度和高度设置应用到已连接的相机。

        此方法尝试通过 OpenCV 设置相机属性。它检查
        相机是否成功应用了设置，如果没有则引发错误。

        参数：
            fps: 所需的每秒帧数。如果为 None，则跳过该设置。
            width: 所需的捕获宽度。如果为 None，则跳过该设置。
            height: 所需的捕获高度。如果为 None，则跳过该设置。

        异常：
            RuntimeError: 如果相机未能将任何指定属性设置为
                          请求的值。
            DeviceNotConnectedError: 如果在尝试配置设置时
                                     相机未连接。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()

    def _validate_fps(self) -> None:
        """验证并设置相机的每秒帧数（FPS）。"""

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        # 使用 math.isclose 进行稳健的浮点数比较
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_width_and_height(self) -> None:
        """验证并设置相机的帧捕获宽度和高度。"""

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用 OpenCV 相机。

        在 Linux 上，它扫描 '/dev/video*' 路径。在其他系统（如 macOS、Windows）上，
        它检查从 0 到 `MAX_OPENCV_INDEX` 的索引。

        返回：
            List[Dict[str, Any]]: 字典列表，
            其中每个字典包含 'type'、'id'（端口索引或路径）
            以及默认配置文件属性（宽度、高度、fps、格式）。
        """
        found_cameras_info = []

        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = list(range(MAX_OPENCV_INDEX))

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                camera.release()

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        从相机同步读取单帧。

        这是一个阻塞调用。它等待相机硬件通过 OpenCV 提供的
        下一个可用帧。

        参数：
            color_mode (Optional[ColorMode]): 如果指定，将覆盖此读取操作的
                默认颜色模式（`self.color_mode`）（例如，
                即使默认为 BGR 也请求 RGB）。

        返回：
            np.ndarray: 捕获的帧作为 NumPy 数组，格式为
                       (height, width, channels)，使用指定或默认的
                       颜色模式并应用任何配置的旋转。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从相机读取帧失败，或如果
                          接收到的帧尺寸在旋转前不匹配预期。
            ValueError: 如果请求的 `color_mode` 无效。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        对原始帧应用颜色转换、尺寸验证和旋转。

        参数：
            image (np.ndarray): 原始图像帧（预期从 OpenCV 获得的 BGR 格式）。
            color_mode (Optional[ColorMode]): 目标颜色模式（RGB 或 BGR）。如果为 None，
                                             使用实例的默认 `self.color_mode`。

        返回：
            np.ndarray: 处理后的图像帧。

        异常：
            ValueError: 如果请求的 `color_mode` 无效。
            RuntimeError: 如果原始帧尺寸与配置的
                          `width` 和 `height` 不匹配。
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        后台线程运行的异步读取内部循环。

        在每次迭代中：
        1. 读取一个彩色帧
        2. 将结果存储在 latest_frame 中（线程安全）
        3. 设置 new_frame_event 以通知监听器

        在 DeviceNotConnectedError 时停止，记录其他错误并继续。
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
        """如果后台读取线程未运行，则启动或重启它。"""
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
        异步读取最新可用的帧。

        此方法检索后台读取线程捕获的最新帧。
        它不会直接阻塞等待相机硬件，
        但可能会等待最多 timeout_ms 以便后台线程提供一个帧。

        参数：
            timeout_ms (float): 等待帧变为可用的最大时间（毫秒）。
                默认为 200ms（0.2 秒）。

        返回：
            np.ndarray: 最新捕获的帧作为 NumPy 数组，格式为
                       (height, width, channels)，根据配置处理。

        异常：
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
        断开与相机的连接并清理资源。

        停止后台读取线程（如果正在运行）并释放 OpenCV
        VideoCapture 对象。

        异常：
            DeviceNotConnectedError: 如果相机已断开连接。
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        logger.info(f"{self} disconnected.")
