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
"""
LeRobot - 机器人学习的现实世界基准

本模块是 LeRobot 库的主入口点，提供对可用环境、数据集、策略、机器人、
相机和电机的访问。LeRobot 是一个用于机器人学习的端到端框架，支持模拟
和真实世界的机器人实验。

本文件包含可用环境、数据集和策略的列表，以反映 LeRobot 库的当前状态。
我们不希望导入所有依赖项，而是保持轻量级以确保快速访问这些变量。

示例：
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_datasets_per_env)
        print(lerobot.available_real_world_datasets)
        print(lerobot.available_policies)
        print(lerobot.available_policies_per_env)
        print(lerobot.available_robots)
        print(lerobot.available_cameras)
        print(lerobot.available_motors)
    ```

实现可通过 LeRobotDataset 加载的新数据集时，请遵循以下步骤：
- 在 `lerobot/__init__.py` 中更新 `available_datasets_per_env`

实现新环境（例如 `gym_aloha`）时，请遵循以下步骤：
- 在 `lerobot/__init__.py` 中更新 `available_tasks_per_env` 和 `available_datasets_per_env`

实现新策略类（例如 `DiffusionPolicy`）时，请遵循以下步骤：
- 在 `lerobot/__init__.py` 中更新 `available_policies` 和 `available_policies_per_env`
- 设置所需的 `name` 类属性
- 通过导入新策略类更新 `tests/test_available.py` 中的变量
"""

import itertools

from lerobot.__version__ import __version__  # noqa: F401

# TODO(rcadene): 改进策略和环境的设计。目前，`available_policies` 中的项
# 同时指代 yaml 配置文件和模型类名称。`available_envs` 也是如此，它同时指代
# yaml 配置文件和环境名称。应该使这两种用途的区别更加明显。
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
}
available_envs = list(available_tasks_per_env.keys())

available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
        "lerobot/aloha_sim_insertion_human_image",
        "lerobot/aloha_sim_insertion_scripted_image",
        "lerobot/aloha_sim_transfer_cube_human_image",
        "lerobot/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): 添加 "lerobot/pusht_keypoints"。目前无法添加，
    # 因为它与测试代码耦合太紧密。
    "pusht": ["lerobot/pusht", "lerobot/pusht_image"],
    "xarm": [
        "lerobot/xarm_lift_medium",
        "lerobot/xarm_lift_medium_replay",
        "lerobot/xarm_push_medium",
        "lerobot/xarm_push_medium_replay",
        "lerobot/xarm_lift_medium_image",
        "lerobot/xarm_lift_medium_replay_image",
        "lerobot/xarm_push_medium_image",
        "lerobot/xarm_push_medium_replay_image",
    ],
}

available_real_world_datasets = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_coffee_new",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_pro_pencil",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_vinh_cup",
    "lerobot/aloha_static_vinh_cup_left",
    "lerobot/aloha_static_ziploc_slide",
    "lerobot/umi_cup_in_the_wild",
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
    "lerobot/nyu_rot_dataset",
    "lerobot/utokyo_saytap",
    "lerobot/imperialcollege_sawyer_wrist_cam",
    "lerobot/utokyo_xarm_bimanual",
    "lerobot/tokyo_u_lsmo",
    "lerobot/utokyo_pr2_opening_fridge",
    "lerobot/cmu_franka_exploration_dataset",
    "lerobot/cmu_stretch",
    "lerobot/asu_table_top",
    "lerobot/utokyo_pr2_tabletop_manipulation",
    "lerobot/utokyo_xarm_pick_and_place",
    "lerobot/ucsd_kitchen_dataset",
    "lerobot/austin_buds_dataset",
    "lerobot/dlr_sara_grid_clamp",
    "lerobot/conq_hose_manipulation",
    "lerobot/columbia_cairlab_pusht_real",
    "lerobot/dlr_sara_pour",
    "lerobot/dlr_edan_shared_control",
    "lerobot/ucsd_pick_and_place_dataset",
    "lerobot/berkeley_cable_routing",
    "lerobot/nyu_franka_play_dataset",
    "lerobot/austin_sirius_dataset",
    "lerobot/cmu_play_fusion",
    "lerobot/berkeley_gnm_sac_son",
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/berkeley_fanuc_manipulation",
    "lerobot/jaco_play",
    "lerobot/viola",
    "lerobot/kaist_nonprehensile",
    "lerobot/berkeley_mvp",
    "lerobot/uiuc_d3field",
    "lerobot/berkeley_gnm_recon",
    "lerobot/austin_sailor_dataset",
    "lerobot/utaustin_mutex",
    "lerobot/roboturk",
    "lerobot/stanford_hydra_dataset",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/stanford_robocook",
    "lerobot/toto",
    "lerobot/fmb",
    "lerobot/droid_100",
    "lerobot/berkeley_rpt",
    "lerobot/stanford_kuka_multimodal_dataset",
    "lerobot/iamlab_cmu_pickup_insert",
    "lerobot/taco_play",
    "lerobot/berkeley_gnm_cory_hall",
    "lerobot/usc_cloth_sim",
]

available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# 列出 `lerobot/policies` 中所有可用的策略
available_policies = ["act", "diffusion", "tdmpc", "vqbet"]

# 列出 `lerobot/robots` 中所有可用的机器人
available_robots = [
    "koch",
    "koch_bimanual",
    "aloha",
    "so100",
    "so101",
]

# 列出 `lerobot/cameras` 中所有可用的相机
available_cameras = [
    "opencv",
    "intelrealsense",
]

# 列出 `lerobot/motors` 中所有可用的电机
available_motors = [
    "dynamixel",
    "feetech",
]

# 键和值都指向 yaml 配置文件
available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion", "vqbet"],
    "xarm": ["tdmpc"],
    "koch_real": ["act_koch_real"],
    "aloha_real": ["act_aloha_real"],
}

env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
