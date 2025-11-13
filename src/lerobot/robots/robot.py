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

import abc
import builtins
from pathlib import Path
from typing import Any

import draccus

from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    所有 LeRobot 兼容机器人的基础抽象类。

    该类为与物理机器人交互提供了标准化接口。
    子类必须实现所有抽象方法和属性才能使用。

    属性:
        config_class (RobotConfig): 该机器人预期的配置类。
        name (str): 用于标识该机器人类型的唯一机器人名称。
    """

    # Set these in ALL subclasses
    config_class: builtins.type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        描述机器人产生的观察数据的结构和类型的字典。
        其结构（键）应与 :pymeth:`get_observation` 返回的结构匹配。
        字典的值应该是以下之一：
            - 如果是简单值，则为该值的类型，例如 `float` 用于单个本体感觉值（关节的位置/速度）
            - 如果是数组类型值，则为表示形状的元组，例如 `(height, width, channel)` 用于图像

        注意：无论机器人是否连接，都应该能够调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        描述机器人预期的动作的结构和类型的字典。其结构（键）应与传递给 :pymeth:`send_action`
        的结构匹配。字典的值应该是简单值的类型，例如 `float` 用于单个本体感觉值
        （关节的目标位置/速度）

        注意：无论机器人是否连接，都应该能够调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        机器人当前是否已连接。如果为 `False`，调用 :pymeth:`get_observation` 或
        :pymeth:`send_action` 应该引发错误。
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与机器人的通信。

        参数:
            calibrate (bool): 如果为 True，在连接后自动校准机器人（如果它未校准或需要校准）
                （这取决于硬件）。
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """机器人当前是否已校准。如果不适用，应始终为 `True`"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        校准机器人（如果适用）。如果不适用，此方法应该是空操作。

        此方法应收集任何必要的数据（例如，电机偏移量）并相应地更新
        :pyattr:`calibration` 字典。
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        从指定文件加载校准数据的辅助方法。

        参数:
            fpath (Path | None): 校准文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        将校准数据保存到指定文件的辅助方法。

        参数:
            fpath (Path | None): 保存校准文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        对机器人应用一次性或运行时配置。
        这可能包括设置电机参数、控制模式或初始状态。
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        从机器人获取当前观察数据。

        返回:
            dict[str, Any]: 表示机器人当前感官状态的扁平字典。其结构应与
                :pymeth:`observation_features` 匹配。
        """

        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        向机器人发送动作命令。

        参数:
            action (dict[str, Any]): 表示期望动作的字典。其结构应与
                :pymeth:`action_features` 匹配。

        返回:
            dict[str, Any]: 实际发送到电机的动作，可能已被裁剪或修改，例如
                被速度安全限制所限制。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与机器人的连接并执行任何必要的清理。"""
        pass
