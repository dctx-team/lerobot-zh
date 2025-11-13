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

# 文档：
# hebi: https://docs.hebi.us/tools.html#mobile-io
# teleop: https://github.com/SpesRobotics/teleop

import logging
import threading
import time

import hebi
import numpy as np
from teleop import Teleop

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)


class BasePhone:
    _enabled: bool = False
    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "phone.pos": np.ndarray,  # shape (3,)
            "phone.rot": Rotation,  # scipy.spatial.transform.Rotation
            "phone.raw_inputs": dict,  # analogs/buttons or webXR meta
            "phone.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # 尚未实现触觉或其他反馈
        pass

    def configure(self) -> None:
        # 手机遥操作不需要额外配置
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # 我们可以在这里添加触觉反馈（振动），但尚未实现
        raise NotImplementedError


class IOSPhone(BasePhone, Teleoperator):
    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._group = None

    @property
    def is_connected(self) -> bool:
        return self._group is not None

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
        lookup = hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
        self._group = group
        logger.info(f"{self} connected to HEBI group with {group.size} module(s).")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")
        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        阻塞执行，直到从iOS设备检测到校准触发器。

        此方法进入循环，持续读取手机的状态。它等待用户在HEBI Mobile I/O应用中按住"B1"按钮。
        一旦按下B1，循环就会中断并返回手机在该时刻的姿态。

        返回：
            一个元组，包含触发器激活时手机的位置（np.ndarray）和旋转（Rotation）。
        """
        while True:
            has_pose, position, rotation, fb_pose = self._read_current_pose()
            if not has_pose:
                time.sleep(0.01)
                continue

            io = getattr(fb_pose, "io", None)
            button_b = getattr(io, "b", None) if io is not None else None
            button_b1_pressed = False
            if button_b is not None:
                button_b1_pressed = bool(button_b.get_int(1))
            if button_b1_pressed:
                return position, rotation

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        通过HEBI SDK从连接的iOS设备读取瞬时6自由度姿态。

        此方法获取来自HEBI组的最新反馈数据包，提取ARKit位置和方向，并将它们转换为标准格式。
        它还应用配置的相机偏移量，以将姿态从相机坐标系调整到手机的物理坐标系。

        返回：
            一个元组，包含：
            - 一个布尔值，指示是否成功读取了有效的姿态。
            - 3D位置作为NumPy数组，如果不可用则为None。
            - 方向作为`Rotation`对象，如果不可用则为None。
            - 原始HEBI反馈对象，用于访问其他数据如按钮按下。
        """
        fbk = self._group.get_next_feedback()
        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, None
        # HEBI以w, x, y, z格式提供方向。
        # Scipy的Rotation期望x, y, z, w格式。
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz转换为xyzw
        rot = Rotation.from_quat(quat_xyzw)
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def get_action(self) -> dict:
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()
        if not has_pose or not self.is_calibrated:
            return {}

        # 收集原始输入（iOS上的B1/模拟量，Android上的move/scale）
        raw_inputs: dict[str, float | int | bool] = {}
        io = getattr(fb_pose, "io", None)
        if io is not None:
            bank_a, bank_b = io.a, io.b
            if bank_a:
                for ch in range(1, 9):
                    if bank_a.has_float(ch):
                        raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
            if bank_b:
                for ch in range(1, 9):
                    if bank_b.has_int(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                    elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))

        enable = bool(raw_inputs.get("b1", 0))

        # 上升沿时从当前原始姿态立即重新捕获校准
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_position)

        # 应用校准
        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._group = None


class AndroidPhone(BasePhone, Teleoperator):
    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        阻塞执行，直到从Android设备检测到校准触发器。

        此方法进入循环，持续检查从WebXR会话接收的最新消息。它等待用户在屏幕上触摸并移动手指，
        这会生成一个`move`事件。一旦检测到此事件，循环就会中断并返回手机的当前姿态。

        返回：
            一个元组，包含触发器激活时手机的位置（np.ndarray）和旋转（Rotation）。
        """
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        读取从Android设备的WebXR会话接收的最新6自由度姿态。

        此方法访问由`_android_callback`存储的最新姿态数据。它使用线程锁安全地读取共享的
        `_latest_pose`变量。姿态是一个4x4矩阵，然后被分解为位置和旋转，并应用配置的相机偏移量。

        返回：
            一个元组，包含：
            - 一个布尔值，指示是否有有效的姿态可用。
            - 3D位置作为NumPy数组，如果尚未接收到姿态则为None。
            - 方向作为`Rotation`对象，如果尚未接收到姿态则为None。
            - 从teleop流接收的原始4x4姿态矩阵。
        """
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        """
        处理来自Android teleop流的传入数据的回调函数。

        此方法由`teleop`包的订阅者线程执行，每当从Android手机上的WebXR会话接收到
        新的姿态和消息时。它使用新数据更新内部状态（`_latest_pose`和`_latest_message`）。
        使用线程锁来确保这些共享变量被原子地更新，防止与读取它们的主线程发生竞态条件。

        参数：
            pose: 一个4x4的NumPy数组，表示手机的变换矩阵。
            message: 一个包含附加数据的字典，如按钮按下或触摸事件。
        """
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # 收集原始输入（iOS上的B1/模拟量，Android上的move/scale）
        raw_inputs: dict[str, float | int | bool] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))

        # 上升沿时从当前原始姿态立即重新捕获校准
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        # 应用校准
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """
    基于手机的遥操作器，使用ARKit（iOS通过HEBI Mobile I/O应用）或teleop Python包（Android通过WebXR API）。
    对于HEBI Mobile I/O，我们还暴露8个模拟输入（a1-a8）和8个数字输入（b1-b8）。

    按住**B1**以启用遥操作。启用时，首次按下B1会捕获参考姿态和旋转，
    禁用后再次按下时会重新应用位置。
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config

        self._phone_impl: Teleoperator

        if self.config.phone_os == PhoneOS.IOS:
            self._phone_impl = IOSPhone(config)
        elif self.config.phone_os == PhoneOS.ANDROID:
            self._phone_impl = AndroidPhone(config)
        else:
            raise ValueError(f"Invalid config phone_os: {self.config.phone_os}")

    @property
    def is_connected(self) -> bool:
        return self._phone_impl.is_connected

    def connect(self) -> None:
        return self._phone_impl.connect()

    def calibrate(self) -> None:
        return self._phone_impl.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._phone_impl.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        return self._phone_impl.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self._phone_impl.feedback_features

    def configure(self) -> None:
        return self._phone_impl.configure()

    def get_action(self) -> dict:
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        return self._phone_impl.disconnect()
