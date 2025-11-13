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

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    所有 LeRobot 兼容遥操作设备的基础抽象类。

    该类为与物理遥操作器交互提供了标准化接口。
    子类必须实现所有抽象方法和属性才能使用。

    属性:
        config_class (RobotConfig): 该遥操作器所需的配置类。
        name (str): 用于识别该遥操作器类型的唯一名称。
    """

    # 在所有子类中设置这些属性
    config_class: builtins.type[TeleoperatorConfig]
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        描述遥操作器产生的动作的结构和类型的字典。其结构（键）应与 :pymeth:`get_action` 返回的结构
        匹配。字典的值应为简单值的类型，例如 `float` 表示单个本体感受值（关节的目标位置/速度）

        注意：无论机器人是否连接，此属性都应该能够被调用。
        """
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        """
        描述机器人期望的反馈动作的结构和类型的字典。其结构（键）应与传递给 :pymeth:`send_feedback`
        的结构匹配。字典的值应为简单值的类型，例如 `float` 表示单个本体感受值（关节的目标位置/速度）

        注意：无论机器人是否连接，此属性都应该能够被调用。
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        遥操作器当前是否已连接。如果为 `False`，调用 :pymeth:`get_action` 或 :pymeth:`send_feedback`
        应抛出错误。
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与遥操作器的通信。

        参数:
            calibrate (bool): 如果为 True，在连接后如果未校准或需要校准（取决于硬件），
                自动校准遥操作器。
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """遥操作器当前是否已校准。如果不适用，应始终为 `True`"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        如果适用，校准遥操作器。如果不适用，此方法应为空操作。

        此方法应收集任何必要的数据（例如电机偏移量）并相应地更新 :pyattr:`calibration` 字典。
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
        对遥操作器应用任何一次性或运行时配置。
        这可能包括设置电机参数、控制模式或初始状态。
        """
        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        """
        从遥操作器检索当前动作。

        返回:
            dict[str, Any]: 表示遥操作器当前动作的扁平字典。其结构应与 :pymeth:`observation_features`
                匹配。
        """
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        向遥操作器发送反馈动作命令。

        参数:
            feedback (dict[str, Any]): 表示所需反馈的字典。其结构应与 :pymeth:`feedback_features`
                匹配。

        返回:
            dict[str, Any]: 实际发送到电机的动作，可能经过裁剪或修改，例如通过速度安全限制。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与遥操作器的连接并执行任何必要的清理工作。"""
        pass
