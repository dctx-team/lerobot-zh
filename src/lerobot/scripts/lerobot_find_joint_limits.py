#!/usr/bin/env python

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

"""
通过远程操作控制机器人的简单脚本。

示例：

```shell
lerobot-find-joint-limits \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.id=black \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem58760431551 \
  --teleop.id=blue
```
"""

import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
)
from lerobot.utils.robot_utils import busy_wait


@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # 远程操作的时间（秒）。默认为 30 秒。
    teleop_time_s: float = 30
    # 是否显示数据（包括相机画面等）。默认不显示。
    display_data: bool = False


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    start_episode_t = time.perf_counter()
    robot_type = getattr(robot.config, "robot_type", "so101")
    if "so100" in robot_type or "so101" in robot_type:
        # 注意：为了与代码库的其余部分保持兼容，
        # 我们对 so101 和 so100 使用新的校准方法
        robot_type = "so_new_calibration"
    kinematics = RobotKinematics(cfg.robot.urdf_path, cfg.robot.target_frame_name)

    # 初始化最小/最大值
    observation = robot.get_observation()
    joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
    ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]

    max_pos = joint_positions.copy()
    min_pos = joint_positions.copy()
    max_ee = ee_pos.copy()
    min_ee = ee_pos.copy()

    while True:
        action = teleop.get_action()
        robot.send_action(action)

        observation = robot.get_observation()
        joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
        ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]

        # 跳过初始预热期
        if (time.perf_counter() - start_episode_t) < 5:
            continue

        # 更新最小/最大值
        max_ee = np.maximum(max_ee, ee_pos)
        min_ee = np.minimum(min_ee, ee_pos)
        max_pos = np.maximum(max_pos, joint_positions)
        min_pos = np.minimum(min_pos, joint_positions)

        if time.perf_counter() - start_episode_t > cfg.teleop_time_s:
            print(f"Max ee position {np.round(max_ee, 4).tolist()}")
            print(f"Min ee position {np.round(min_ee, 4).tolist()}")
            print(f"Max joint pos position {np.round(max_pos, 4).tolist()}")
            print(f"Min joint pos position {np.round(min_pos, 4).tolist()}")
            break

        busy_wait(0.01)


def main():
    find_joint_and_ee_bounds()


if __name__ == "__main__":
    main()
