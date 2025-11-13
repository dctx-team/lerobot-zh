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
帮助重新校准设备（机器人或遥操作器）的工具。

示例：

```shell
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.utils import init_logging


@dataclass
class CalibrateConfig:
    # 遥操作器配置，如果不校准遥操作器则为 None
    teleop: TeleoperatorConfig | None = None
    # 机器人配置，如果不校准机器人则为 None
    robot: RobotConfig | None = None

    def __post_init__(self):
        # 确保只选择遥操作器或机器人其中之一
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        # 将选中的设备（机器人或遥操作器）赋值给 device
        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    # 初始化日志系统
    init_logging()
    # 记录配置信息
    logging.info(pformat(asdict(cfg)))

    # 根据设备类型创建相应的实例
    if isinstance(cfg.device, RobotConfig):
        # 从配置创建机器人实例
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        # 从配置创建遥操作器实例
        device = make_teleoperator_from_config(cfg.device)

    # 连接设备但不进行校准
    device.connect(calibrate=False)
    # 执行设备校准
    device.calibrate()
    # 断开设备连接
    device.disconnect()


def main():
    calibrate()


if __name__ == "__main__":
    main()
