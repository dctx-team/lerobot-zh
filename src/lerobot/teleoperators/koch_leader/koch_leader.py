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

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_koch_leader import KochLeaderConfig

logger = logging.getLogger(__name__)


class KochLeader(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot)，带或不带腕到肘的扩展，
        由 [Tau Robotics](https://tau-robotics.com) 的 Alexander Koch 开发
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) 由 Jess Moss 开发
    """

    config_class = KochLeaderConfig
    name = "koch_leader"

    def __init__(self, config: KochLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "电机中的校准值与校准文件不匹配或未找到校准文件"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        if self.calibration:
            # 校准文件存在，询问用户是使用它还是运行新的校准
            user_input = input(
                f"按 ENTER 使用与 ID {self.id} 关联的提供的校准文件，或输入 'c' 并按 ENTER 运行校准: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"将与 ID {self.id} 关联的校准文件写入电机")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\n正在运行 {self} 的校准")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
        drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}

        input(f"将 {self} 移动到其运动范围的中间位置并按 ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"依次移动除 {full_turn_motors} 之外的所有关节通过它们的"
            "整个运动范围。\n正在记录位置。按 ENTER 停止..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"校准已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                # 对除夹爪外的所有电机使用"扩展位置模式"，因为在关节模式下，舵机
                # 不能旋转超过 360 度（从 0 到 4095），并且在组装机械臂时可能会发生一些错误，
                # 你可能会在关键点得到一个位置为 0 或 4095 的舵机
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # 对夹爪使用"基于电流的位置控制"以受电流限制的限制。
        # 对于从动夹爪，这意味着它可以抓取物体而不会施加太多力，即使
        # 它的目标位置是完全抓取（两个夹爪手指被命令合拢并接触）。
        # 对于主动夹爪，这意味着我们可以将其用作物理触发器，因为我们可以用手指施力
        # 使其移动，当我们释放力时，它会移回到其原始目标位置。
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # 在电流位置模式下设置夹爪的目标位置，以便我们可以将其用作触发器。
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"仅将控制器板连接到 '{motor}' 电机并按回车。")
            self.bus.setup_motor(motor)
            print(f"'{motor}' 电机 ID 已设置为 {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): 实现力反馈
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
