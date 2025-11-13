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

"""
帮助查找系统中可用的摄像头设备。

示例:

```shell
lerobot-find-cameras
```
"""

# 注意(Steven): RealSense 也可以被识别/作为 OpenCV 摄像头打开。如果你知道摄像头是 RealSense，请使用 `lerobot-find-cameras realsense` 标志以避免混淆。
# 注意(Steven): macOS 摄像头有时在初始化时报告不同的 FPS，这在这里不是问题，因为我们在打开摄像头时不指定 FPS，但显示的信息可能不准确。

import argparse
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    查找所有连接到系统的可用 OpenCV 摄像头。

    返回:
        包含所有可用 OpenCV 摄像头及其元数据的列表。
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("正在搜索 OpenCV 摄像头...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"找到 {len(opencv_cameras)} 个 OpenCV 摄像头。")
    except Exception as e:
        logger.error(f"查找 OpenCV 摄像头时出错: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    查找所有连接到系统的可用 RealSense 摄像头。

    返回:
        包含所有可用 RealSense 摄像头及其元数据的列表。
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("正在搜索 RealSense 摄像头...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"找到 {len(realsense_cameras)} 个 RealSense 摄像头。")
    except ImportError:
        logger.warning("跳过 RealSense 摄像头搜索: 未找到或无法导入 pyrealsense2 库。")
    except Exception as e:
        logger.error(f"查找 RealSense 摄像头时出错: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    根据可选过滤器查找可用摄像头并打印其信息。

    参数:
        camera_type_filter: 可选字符串，用于过滤摄像头("realsense" 或 "opencv")。
                            如果为 None，则列出所有摄像头。

    返回:
        包含所有匹配过滤器的可用摄像头及其元数据的列表。
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"未检测到 {camera_type_filter} 摄像头。")
        else:
            logger.warning("未检测到摄像头(OpenCV 或 RealSense)。")
    else:
        print("\n--- 检测到的摄像头 ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"摄像头 #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
):
    """
    使用 Pillow 将单张图像保存到磁盘。必要时处理颜色转换。
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"已保存图像: {path}")
    except Exception as e:
        logger.error(f"保存摄像头 {camera_identifier} (类型 {camera_type}) 的图像失败: {e}")


def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
    """根据元数据创建并连接到摄像头实例。"""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"正在准备 {cam_type} ID {cam_id} (使用默认配置)")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"未知的摄像头类型: {cam_type} (ID {cam_id})。跳过。")
            return None

        if instance:
            logger.info(f"正在连接到 {cam_type} 摄像头: {cam_id}...")
            instance.connect(warmup=False)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"连接或配置 {cam_type} 摄像头 {cam_id} 失败: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
    """从单个摄像头捕获并处理图像。"""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"从 {cam_type_str} 摄像头 {cam_id_str} 读取超时 (时间 {current_time:.2f}秒)。"
        )
    except Exception as e:
        logger.error(f"从 {cam_type_str} 摄像头 {cam_id_str} 读取时出错: {e}")
    return None


def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
    """断开所有摄像头连接。"""
    logger.info(f"正在断开 {len(cameras_to_use)} 个摄像头的连接...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"断开摄像头 {cam_dict['meta'].get('id')} 的连接时出错: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
):
    """
    连接到检测到的摄像头(可选按类型过滤)并保存每个摄像头的图像。
    使用默认的流配置文件设置宽度、高度和 FPS。

    参数:
        output_dir: 保存图像的目录。
        record_time_s: 录制图像的持续时间(秒)。
        camera_type: 可选字符串，用于过滤摄像头("realsense" 或 "opencv")。
                            如果为 None，则使用所有检测到的摄像头。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"正在将图像保存到 {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("未检测到符合条件的摄像头。无法保存图像。")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("无法连接任何摄像头。终止图像保存。")
        return

    logger.info(f"开始从 {len(cameras_to_use)} 个摄像头捕获图像，持续 {record_time_s} 秒。")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []
                current_capture_time = time.perf_counter()

                for cam_dict in cameras_to_use:
                    future = process_camera_image(cam_dict, output_dir, current_capture_time)
                    if future:
                        futures.append(future)

                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("捕获被用户中断。")
        finally:
            print("\n正在完成图像保存...")
            executor.shutdown(wait=True)
            cleanup_cameras(cameras_to_use)
            print(f"图像捕获完成。图像已保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="用于列出摄像头和捕获图像的统一摄像头工具脚本。"
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="指定要捕获的摄像头类型(例如 'realsense'、'opencv')。如果省略则从所有摄像头捕获。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="保存图像的目录。默认: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="尝试捕获帧的持续时间。默认: 6 秒。",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
