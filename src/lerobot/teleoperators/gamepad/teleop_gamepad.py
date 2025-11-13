# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    使用游戏手柄输入进行控制的遥操作类。
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # 在 macOS 上使用 HidApi
        if sys.platform == "darwin":
            # 注意：在 macOS 上，pygame 无法可靠地检测某些控制器的输入，因此我们使用 hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    def get_action(self) -> dict[str, Any]:
        # 更新控制器以获取最新输入
        self.gamepad.update()

        # 从控制器获取移动增量
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # 从游戏手柄输入创建动作
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # 默认夹爪动作为保持
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        从游戏手柄获取额外的控制事件，例如干预状态、
        回合终止、成功指示器等。

        返回：
            包含以下内容的字典：
                - is_intervention: bool - 人类当前是否正在干预
                - terminate_episode: bool - 是否终止当前回合
                - success: bool - 回合是否成功
                - rerecord_episode: bool - 是否重新记录回合
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # 更新游戏手柄状态以获取最新输入
        self.gamepad.update()

        # 检查干预是否激活
        is_intervention = self.gamepad.should_intervene()

        # 获取回合结束状态
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """断开与游戏手柄的连接。"""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        """检查游戏手柄是否已连接。"""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """校准游戏手柄。"""
        # 游戏手柄不需要校准
        pass

    def is_calibrated(self) -> bool:
        """检查游戏手柄是否已校准。"""
        # 游戏手柄不需要校准
        return True

    def configure(self) -> None:
        """配置游戏手柄。"""
        # 不需要额外的配置
        pass

    def send_feedback(self, feedback: dict) -> None:
        """向游戏手柄发送反馈。"""
        # 游戏手柄不支持反馈
        pass
