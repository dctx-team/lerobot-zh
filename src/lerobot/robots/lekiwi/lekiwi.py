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
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_lekiwi import LeKiwiConfig

logger = logging.getLogger(__name__)


class LeKiwi(Robot):
    """
    该机器人包括一个三全向轮移动底座和一个远程跟随臂。
    领导臂在本地（在笔记本电脑上）连接，其关节位置被记录然后
    转发到远程跟随臂（应用安全限位后）。
    同时，键盘遥控操作用于为车轮生成原始速度命令。
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # 机械臂
                "arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # 底座
                "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.arm_motors = [motor for motor in self.bus.motors if motor.startswith("arm")]
        self.base_motors = [motor for motor in self.bus.motors if motor.startswith("base")]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "电机中的校准值与校准文件不匹配或未找到校准文件"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # 校准文件存在，询问用户是使用它还是运行新的校准
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")

        motors = self.arm_motors + self.base_motors

        self.bus.disable_torque(self.arm_motors)
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings(self.arm_motors)

        homing_offsets.update(dict.fromkeys(self.base_motors, 0))

        full_turn_motor = [
            motor for motor in motors if any(keyword in motor for keyword in ["wheel", "wrist"])
        ]
        unknown_range_motors = [motor for motor in motors if motor not in full_turn_motor]

        print(
            f"Move all arm joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motor:
            range_mins[name] = 0
            range_maxes[name] = 4095

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # 设置机械臂执行器（位置模式）
        # 我们假设在连接时，机械臂处于静止位置，
        # 并且可以安全地禁用力矩以运行校准。
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # 将 P_Coefficient 设置为较低值以避免抖动（默认为 32）
            self.bus.write("P_Coefficient", name, 16)
            # 将 I_Coefficient 和 D_Coefficient 设置为默认值 0 和 32
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # 将值限制在有符号 16 位范围内（-32768 到 32767）
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> 最大正值
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> 最小负值
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        将期望的机体坐标系速度转换为车轮原始命令。

        参数：
          x_cmd      : x 方向的线速度（米/秒）。
          y_cmd      : y 方向的线速度（米/秒）。
          theta_cmd  : 旋转速度（度/秒）。
          wheel_radius: 每个车轮的半径（米）。
          base_radius : 从旋转中心到每个车轮的距离（米）。
          max_raw    : 每个车轮允许的最大原始命令（刻度）。

        返回：
          包含车轮原始命令的字典：
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}。

        注意：
          - 在内部，该方法将 theta_cmd 转换为 rad/s 用于运动学计算。
          - 原始命令是从车轮的角速度（度/秒）使用 _degps_to_raw() 计算的。
            如果任何命令超过 max_raw，所有命令将按比例缩小。
        """
        # 将旋转速度从 deg/s 转换为 rad/s。
        theta_rad = theta * (np.pi / 180.0)
        # 创建机体速度向量 [x, y, theta_rad]。
        velocity_vector = np.array([x, y, theta_rad])

        # 定义车轮安装角度，带有 -90° 偏移。
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # 构建运动学矩阵：每行将机体速度映射到车轮的线速度。
        # 第三列（base_radius）考虑旋转的影响。
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 计算每个车轮的线速度（米/秒），然后是其角速度（rad/s）。
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # 将车轮角速度从 rad/s 转换为 deg/s。
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # 缩放
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # 将每个车轮的角速度（度/秒）转换为原始整数。
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        将车轮原始命令反馈转换回机体坐标系速度。

        参数：
          wheel_raw   : 包含原始车轮命令的向量（"base_left_wheel"、"base_back_wheel"、"base_right_wheel"）。
          wheel_radius: 每个车轮的半径（米）。
          base_radius : 从机器人中心到每个车轮的距离（米）。

        返回：
          一个字典（x.vel、y.vel、theta.vel），全部以 m/s 为单位
        """

        # 将每个原始命令转换回角速度（度/秒）。
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # 从 deg/s 转换为 rad/s。
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # 从角速度计算每个车轮的线速度（米/秒）。
        wheel_linear_speeds = wheel_radps * wheel_radius

        # 定义车轮安装角度，带有 -90° 偏移。
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 求解逆运动学：body_velocity = M⁻¹ · wheel_linear_speeds。
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s 和 deg/s

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取机械臂的执行器位置和底座的速度
        start = time.perf_counter()
        arm_pos = self.bus.sync_read("Present_Position", self.arm_motors)
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        arm_state = {f"{k}.pos": v for k, v in arm_pos.items()}

        obs_dict = {**arm_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 从相机捕获图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """命令 lekiwi 移动到目标关节配置。

        根据配置参数 `max_relative_target`，相对动作幅度可能会被裁剪。
        在这种情况下，发送的动作与原始动作不同。
        因此，此函数始终返回实际发送的动作。

        引发：
            RobotDeviceNotConnectedError: 如果机器人未连接。

        返回：
            np.ndarray: 发送到电机的动作，可能已裁剪。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # 当目标位置距离当前位置太远时进行限制。
        # /!\ 由于需要从跟随器读取数据，预期 fps 会较慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", self.arm_motors)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
            arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            arm_goal_pos = arm_safe_goal_pos

        # 将目标位置发送到执行器
        arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        self.bus.sync_write("Goal_Position", arm_goal_pos_raw)
        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**arm_goal_pos, **base_goal_vel}

    def stop_base(self):
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
