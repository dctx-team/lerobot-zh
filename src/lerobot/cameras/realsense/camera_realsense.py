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
提供用于从 Intel RealSense 相机捕获帧的 RealSenseCamera 类。
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    logging.info(f"无法导入 realsense: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


class RealSenseCamera(Camera):
    """
    管理与 Intel RealSense 相机的交互，用于帧和深度记录。

    该类提供了与 `OpenCVCamera` 类似的接口，但专为 RealSense 设备量身定制，
    利用 `pyrealsense2` 库。它使用相机的唯一序列号进行识别，提供比设备
    索引更高的稳定性，特别是在 Linux 上。它还支持在捕获彩色帧的同时捕获深度图。

    使用提供的实用脚本查找可用的相机索引和默认配置文件：
    ```bash
    lerobot-find-cameras realsense
    ```

    `RealSenseCamera` 实例需要一个配置对象，指定相机的序列号或唯一的设备名称。
    如果使用名称，请确保只连接一个具有该名称的相机。

    除非在配置中覆盖，否则将使用流配置文件中相机的默认设置（FPS、分辨率、颜色模式）。

    示例：
        ```python
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # 使用序列号的基本用法
        config = RealSenseCameraConfig(serial_number_or_name="0123456789") # 替换为实际序列号
        camera = RealSenseCamera(config)
        camera.connect()

        # 同步读取 1 帧
        color_image = camera.read()
        print(color_image.shape)

        # 异步读取 1 帧
        async_image = camera.async_read()

        # 完成后，正确断开相机连接
        camera.disconnect()

        # 带有深度捕获和自定义设置的示例
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="0123456789", # 替换为实际序列号
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # 请求 BGR 输出
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        depth_camera.connect()

        # 读取 1 帧深度图
        depth_map = depth_camera.read_depth()

        # 使用唯一相机名称的示例
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # 如果唯一
        name_camera = RealSenseCamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        初始化 RealSenseCamera 实例。

        参数：
            config: 相机的配置设置。
        """

        super().__init__(config)

        self.config = config

        if config.serial_number_or_name.isdigit():
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """检查相机管道是否已启动且流是否处于活动状态。"""
        return self.rs_pipeline is not None and self.rs_profile is not None

    def connect(self, warmup: bool = True):
        """
        连接到配置中指定的 RealSense 相机。

        初始化 RealSense 管道，配置所需的流（颜色和可选的深度），启动管道，
        并验证实际的流设置。

        抛出异常：
            DeviceAlreadyConnectedError: 如果相机已连接。
            ValueError: 如果配置无效（例如，缺少序列号/名称，名称不唯一）。
            ConnectionError: 如果找到相机但无法启动管道，或者根本未检测到 RealSense 设备。
            RuntimeError: 如果管道启动但无法应用请求的设置。
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已经连接。")

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        self._configure_rs_pipeline_config(rs_config)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            raise ConnectionError(
                f"无法打开 {self}。运行 `lerobot-find-cameras realsense` 以查找可用的相机。"
            ) from e

        self._configure_capture_settings()

        if warmup:
            time.sleep(
                1
            )  # 注意(Steven): RS 相机需要一点时间预热才能进行第一次读取。如果不等待，预热的第一次读取将引发异常。
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} 已连接。")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用 Intel RealSense 相机。

        返回：
            List[Dict[str, Any]]: 字典列表，
            其中每个字典包含 'type'、'id'（序列号）、'name'、
            固件版本、USB 类型和其他可用规格，以及默认配置文件属性（width、height、fps、format）。

        抛出异常：
            OSError: 如果未安装 pyrealsense2。
            ImportError: 如果未安装 pyrealsense2。
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # 获取每个传感器的流配置文件
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """根据给定的唯一相机名称查找序列号。"""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"未找到名为 '{name}' 的 RealSense 相机。可用的相机名称：{available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"找到多个名为 '{name}' 的 RealSense 相机。"
                f"请使用唯一的序列号。找到的序列号：{serial_numbers}"
            )

        serial_number = str(found_devices[0]["serial_number"])
        return serial_number

    def _configure_rs_pipeline_config(self, rs_config):
        """创建并配置 RealSense 管道配置对象。"""
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            rs_config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
            if self.use_depth:
                rs_config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
        else:
            rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth)

    def _configure_capture_settings(self) -> None:
        """如果尚未配置，则从设备流设置 fps、width 和 height。

        使用颜色流配置文件更新未设置的属性。在需要时通过交换 width/height 来处理旋转。
        始终存储原始捕获尺寸。

        抛出异常：
            DeviceNotConnectedError: 如果设备未连接。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"无法验证 {self} 的设置，因为它未连接。")

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        从相机同步读取单帧（深度）。

        这是一个阻塞调用。它通过 RealSense 管道等待来自相机硬件的一致帧集（深度）。

        参数：
            timeout_ms (int): 等待帧的最长时间（毫秒）。默认为 200 毫秒。

        返回：
            np.ndarray: 深度图作为 NumPy 数组（height, width），
                  类型为 `np.uint16`（原始深度值，单位为毫米）和旋转。

        抛出异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从管道读取帧失败或帧无效。
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接。")
        if not self.use_depth:
            raise RuntimeError(
                f"无法捕获深度帧 '.read_depth()'。{self} 的深度流未启用。"
            )

        start_time = time.perf_counter()

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)

        if not ret or frame is None:
            raise RuntimeError(f"{self} read_depth 失败（状态={ret}）。")

        depth_frame = frame.get_depth_frame()
        depth_map = np.asanyarray(depth_frame.get_data())

        depth_map_processed = self._postprocess_image(depth_map, depth_frame=True)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} 读取耗时：{read_duration_ms:.1f}ms")

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        从相机同步读取单帧（颜色）。

        这是一个阻塞调用。它通过 RealSense 管道等待来自相机硬件的一致帧集（颜色）。

        参数：
            timeout_ms (int): 等待帧的最长时间（毫秒）。默认为 200 毫秒。

        返回：
            np.ndarray: 捕获的彩色帧作为 NumPy 数组
              （height, width, channels），根据 `color_mode` 和旋转进行处理。

        抛出异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从管道读取帧失败或帧无效。
            ValueError: 如果请求的 `color_mode` 无效。
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接。")

        start_time = time.perf_counter()

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)

        if not ret or frame is None:
            raise RuntimeError(f"{self} read 失败（状态={ret}）。")

        color_frame = frame.get_color_frame()
        color_image_raw = np.asanyarray(color_frame.get_data())

        color_image_processed = self._postprocess_image(color_image_raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} 读取耗时：{read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
        self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        对原始彩色帧应用颜色转换、尺寸验证和旋转。

        参数：
            image (np.ndarray): 原始图像帧（RealSense 预期的 RGB 格式）。
            color_mode (Optional[ColorMode]): 目标颜色模式（RGB 或 BGR）。如果为 None，
                                             使用实例的默认 `self.color_mode`。

        返回：
            np.ndarray: 根据 `self.color_mode` 和 `self.rotation` 处理后的图像帧。

        抛出异常：
            ValueError: 如果请求的 `color_mode` 无效。
            RuntimeError: 如果原始帧尺寸与配置的
                          `width` 和 `height` 不匹配。
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"无效的请求颜色模式 '{color_mode}'。预期为 {ColorMode.RGB} 或 {ColorMode.BGR}。"
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} 帧通道数={c} 与预期的 3 个通道（RGB/BGR）不匹配。")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} 帧 width={w} 或 height={h} 与配置的 width={self.capture_width} 或 height={self.capture_height} 不匹配。"
            )

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        后台线程运行的内部循环，用于异步读取。

        每次迭代时：
        1. 读取一个彩色帧，超时时间为 500 毫秒
        2. 将结果存储在 latest_frame 中（线程安全）
        3. 设置 new_frame_event 以通知监听器

        在 DeviceNotConnectedError 上停止，记录其他错误并继续。
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"在 {self} 的后台线程中读取帧时出错：{e}")

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

    def _stop_read_thread(self):
        """向后台读取线程发出停止信号并等待其加入。"""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    # 注意(Steven): 目前缺少深度的实现
    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        异步读取最新可用的帧数据（颜色）。

        此方法检索后台读取线程捕获的最新彩色帧。它不会阻塞等待相机硬件，
        但可能会等待最多 timeout_ms 以让后台线程提供帧。

        参数：
            timeout_ms (float): 等待帧可用的最长时间（毫秒）。
                默认为 200 毫秒（0.2 秒）。

        返回：
            np.ndarray:
            最新捕获的帧数据（彩色图像），根据配置进行处理。

        抛出异常：
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定的超时时间内没有帧数据可用。
            RuntimeError: 如果后台线程意外死亡或发生其他错误。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接。")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"在 {timeout_ms} 毫秒后等待来自相机 {self} 的帧超时。"
                f"读取线程存活：{thread_alive}。"
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"内部错误：事件已设置但 {self} 没有可用的帧。")

        return frame

    def disconnect(self):
        """
        断开与相机的连接，停止管道并清理资源。

        停止后台读取线程（如果正在运行）并停止 RealSense 管道。

        抛出异常：
            DeviceNotConnectedError: 如果相机已断开连接（管道未运行）。
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"尝试断开 {self}，但它似乎已经断开连接。"
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

        logger.info(f"{self} 已断开连接。")
