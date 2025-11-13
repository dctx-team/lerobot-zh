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
在机器人上重放数据集中某个回合的动作。

示例：

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0
```

使用双臂 so100 重放的示例：
```shell
lerobot-replay \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # 数据集标识符。按照惯例应该匹配 '{hf_username}/{dataset_name}' (例如 `lerobot/test`)。
    repo_id: str
    # 要重放的回合。
    episode: int
    # 存储数据集的根目录 (例如 'dataset/path')。
    root: str | Path | None = None
    # 限制每秒帧数。默认情况下使用策略的帧率。
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # 使用语音合成来朗读事件。
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # 过滤数据集，仅包含指定回合的帧，因为在数据集 V3.0 中回合是分块的
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(len(episode_frames)):
        start_episode_t = time.perf_counter()

        action_array = actions[idx][ACTION]
        action = {}
        for i, name in enumerate(dataset.features[ACTION]["names"]):
            action[name] = action_array[i]

        robot_obs = robot.get_observation()

        processed_action = robot_action_processor((action, robot_obs))

        _ = robot.send_action(processed_action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / dataset.fps - dt_s)

    robot.disconnect()


def main():
    replay()


if __name__ == "__main__":
    main()
