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
用于设置电机 ID 和波特率的辅助工具。

示例：

```shell
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751
```
"""

from dataclasses import dataclass

import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)

# 兼容的设备列表
COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "lekiwi",
]


@dataclass
class SetupConfig:
    """
    设置配置类，用于配置遥操作器或机器人。

    属性：
        teleop: 遥操作器配置对象，可选
        robot: 机器人配置对象，可选
    """
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        """初始化后处理，验证配置的有效性。"""
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("请选择遥操作器或机器人其中之一。")

        # 根据实际配置设置设备对象
        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    """
    设置电机的主函数。

    参数：
        cfg: 设置配置对象，包含遥操作器或机器人的配置信息。

    异常：
        NotImplementedError: 当设备类型不在兼容列表中时抛出。
    """
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError

    # 根据配置类型创建相应的设备实例
    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    # 执行电机设置
    device.setup_motors()


def main():
    """主入口函数，调用电机设置流程。"""
    setup_motors()


if __name__ == "__main__":
    main()
