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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_viperx import ViperXConfig

logger = logging.getLogger(__name__)


class ViperX(Robot):
    """
    由 Trossen Robotics 开发的 [ViperX](https://www.trossenrobotics.com/viperx-300) 机器人
    """

    config_class = ViperXConfig
    name = "viperx"

    def __init__(
        self,
        config: ViperXConfig,
    ):
        raise NotImplementedError
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": Motor(1, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "shoulder": Motor(2, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "shoulder_shadow": Motor(3, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "elbow": Motor(4, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "elbow_shadow": Motor(5, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "forearm_roll": Motor(6, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_angle": Motor(7, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_rotate": Motor(8, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_100),
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        我们假设在连接时，机械臂处于静止位置，
        可以安全地禁用扭矩以运行校准。
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO(aliberts): 调整以下代码（从 koch 复制）
        logger.info(f"\n正在运行 {self} 的校准")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input("将机器人移动到其运动范围的中间位置，然后按 ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"依次移动除 {full_turn_motors} 外的所有关节通过它们的整个"
            "运动范围。\n正在记录位置。按 ENTER 停止..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"校准已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            # 为肩部和肘部设置次要/影子 ID。这些关节有两个电机。
            # 因此，如果只需要其中一个移动到某个位置，
            # 另一个将跟随。这是为了避免损坏电机。
            self.bus.write("Secondary_ID", "shoulder_shadow", 2)
            self.bus.write("Secondary_ID", "elbow_shadow", 4)

            # 按照 Trossen Robotics 的建议设置速度限制为 131
            # TODO(aliberts): 删除，因为在位置控制中实际上没有用
            self.bus.write("Velocity_Limit", 131)

            # 对除夹爪外的所有电机使用"扩展位置模式"，因为在关节模式下，舵机
            # 无法旋转超过 360 度（从 0 到 4095），组装
            # 机械臂时可能会出现一些错误，你可能会得到一个在关键点位置为 0 或 4095 的舵机。
            # 参见：https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # 对跟随夹爪使用"基于电流的位置控制"，以受到电流限制的限制。
            # 即使其目标位置是完全抓取（两个夹爪手指被命令合拢并达到接触），
            # 它也可以抓住物体而不会用力过大。
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        """返回的观测值没有批次维度。"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # 读取机械臂位置
        start = time.perf_counter()
        obs_dict[OBS_STATE] = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 从相机捕获图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """命令机械臂移动到目标关节配置。

        相对动作幅度可能会根据配置参数 `max_relative_target` 被裁剪。
        在这种情况下，发送的动作与原始动作不同。
        因此，此函数始终返回实际发送的动作。

        参数：
            action (dict[str, float]): 电机的目标位置。

        返回：
            dict[str, float]: 发送到电机的动作，可能已被裁剪。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 当目标位置离当前位置太远时限制目标位置。
        # /!\ 由于需要从跟随器读取，预计 fps 会较慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 将目标位置发送到机械臂
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
