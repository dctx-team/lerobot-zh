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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    """LeKiwi机器人的默认相机配置"""
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    """LeKiwi机器人配置"""

    port: str = "/dev/ttyACM0"  # 连接到总线的端口

    disable_torque_on_disconnect: bool = True  # 断开连接时禁用扭矩

    # `max_relative_target` 限制相对位置目标向量的幅度以确保安全。
    # 将其设置为正标量可对所有电机使用相同的值，或设置为字典以将电机
    # 名称映射到该电机的 max_relative_target 值。
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)  # 相机配置

    # 设置为 `True` 以向后兼容以前的策略/数据集
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    """LeKiwi主机端配置"""

    # 网络配置
    port_zmq_cmd: int = 5555  # ZMQ命令端口
    port_zmq_observations: int = 5556  # ZMQ观测数据端口

    # 应用程序持续时间
    connection_time_s: int = 30  # 连接超时时间（秒）

    # 看门狗：如果超过 0.5 秒未收到命令，则停止机器人。
    watchdog_timeout_ms: int = 500  # 看门狗超时时间（毫秒）

    # 如果机器人抖动，请降低频率并使用 `top` 命令监控 CPU 负载
    max_loop_freq_hz: int = 30  # 最大循环频率（Hz）


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    """LeKiwi客户端配置"""

    # 网络配置
    remote_ip: str  # 远程主机IP地址
    port_zmq_cmd: int = 5555  # ZMQ命令端口
    port_zmq_observations: int = 5556  # ZMQ观测数据端口

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # 移动
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # 速度控制
            "speed_up": "r",
            "speed_down": "f",
            # 退出遥控操作
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)  # 相机配置

    polling_timeout_ms: int = 15  # 轮询超时时间（毫秒）
    connect_timeout_s: int = 5  # 连接超时时间（秒）
