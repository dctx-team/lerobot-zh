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
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """机器人配置基类。

    用于定义机器人配置的抽象基类，包含通用的配置字段。
    """
    # Allows to distinguish between different robots of the same type
    # 用于区分同一类型的不同机器人
    id: str | None = None
    # Directory to store calibration file
    # 存储校准文件的目录
    calibration_dir: Path | None = None

    def __post_init__(self):
        """初始化后验证配置。

        验证相机配置中的必需属性（宽度、高度、帧率）是否已设置。

        Raises:
            ValueError: 如果相机配置缺少必需的属性。
        """
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )

    @property
    def type(self) -> str:
        """获取机器人配置的类型名称。

        Returns:
            str: 机器人配置的类型标识符。
        """
        return self.get_choice_name(self.__class__)
