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

########################################################################################
# 工具函数
########################################################################################


import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot


@cache
def is_headless():
    """
    检测 Python 脚本是否在无头环境中运行（例如没有显示器）。

    此函数尝试导入 `pynput`，这是一个需要图形环境的库。
    如果导入失败，则假定环境为无头环境。结果会被缓存以避免重复检查。

    返回:
        如果确定环境为无头环境，则返回 True，否则返回 False。
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "尝试导入 pynput 时出错。切换到无头模式。"
            "因此，相机的视频流将不会显示，"
            "并且您将无法通过键盘更改控制流程。"
            "有关更多信息，请参见下面的追踪信息。\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    执行单步推理以从观测中预测机器人动作。

    此函数封装了完整的推理流程：
    1. 通过将观测转换为 PyTorch 张量并添加批次维度来准备观测。
    2. 在观测上运行预处理流程。
    3. 将处理后的观测输入策略以获得原始动作。
    4. 在原始动作上运行后处理流程。
    5. 通过移除批次维度并将其移至 CPU 来格式化最终动作。

    参数：
        observation: 表示机器人当前观测的 NumPy 数组字典。
        policy: 用于动作预测的 `PreTrainedPolicy` 模型。
        device: 运行推理的 `torch.device`（例如 'cuda' 或 'cpu'）。
        preprocessor: 用于预处理观测的 `PolicyProcessorPipeline`。
        postprocessor: 用于后处理动作的 `PolicyProcessorPipeline`。
        use_amp: 一个布尔值，用于启用/禁用 CUDA 推理的自动混合精度。
        task: 任务的可选字符串标识符。
        robot_type: 机器人类型的可选字符串标识符。

    返回：
        包含预测动作的 `torch.Tensor`，准备用于机器人。
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # 转换为 pytorch 格式: 通道优先，批次维度中的 float32 [0,1]
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # 使用策略计算下一个动作
        # 基于当前观测
        action = policy.select_action(observation)

        action = postprocessor(action)

        # 移除批次维度
        action = action.squeeze(0)

        # 移动到 cpu，如果尚未在 cpu 上
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    """
    初始化用于实时用户交互的非阻塞键盘监听器。

    此函数设置特定按键（右箭头、左箭头、Esc）的监听器，以在执行期间控制
    程序流程，例如停止录制或退出循环。它优雅地处理不支持键盘监听的无头环境。

    返回：
        包含以下内容的元组：
        - `pynput.keyboard.Listener` 实例，如果在无头环境中则为 `None`。
        - 由按键设置的事件标志字典（例如 `exit_early`）。
    """
    # 允许在录制回合或重置环境时通过点击右箭头键 '->' 提前退出。
    # 这可能需要 sudo 权限以允许您的终端监控键盘事件。
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "检测到无头环境。屏幕摄像头显示和键盘输入将不可用。"
        )
        listener = None
        return listener, events

    # 仅在非无头环境中导入 pynput
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("按下右箭头键。退出循环...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("按下左箭头键。退出循环并重新录制最后一个回合...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("按下 Esc 键。停止数据录制...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"处理按键时出错: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    针对策略配置的存在验证数据集仓库名称。

    此函数强制执行命名约定：当且仅当提供了用于评估目的的策略配置时，
    数据集仓库 ID 才应以 "eval_" 开头。

    参数：
        repo_id: 数据集的 Hugging Face Hub 仓库 ID。
        policy_cfg: 策略的配置对象，或 `None`。

    抛出：
        ValueError: 如果违反命名约定。
    """
    _, dataset_name = repo_id.split("/")
    # repo_id 不以 "eval_" 开头且没有策略
    # 或者 repo_id 以 "eval_" 开头且有策略

    # 检查 dataset_name 是否以 "eval_" 开头但缺少策略
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"您的数据集名称以 'eval_' 开头 ({dataset_name})，但没有提供策略 ({policy_cfg.type})。"
        )

    # 检查 dataset_name 是否不以 "eval_" 开头但提供了策略
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"您的数据集名称不以 'eval_' 开头 ({dataset_name})，但提供了策略 ({policy_cfg.type})。"
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    检查数据集的元数据是否与当前机器人和录制设置兼容。

    此函数将数据集中的关键元数据字段（`robot_type`、`fps` 和 `features`）
    与当前配置进行比较，以确保追加的数据将是一致的。

    参数：
        dataset: 要检查的 `LeRobotDataset` 实例。
        robot: 表示当前硬件设置的 `Robot` 实例。
        fps: 当前录制频率（每秒帧数）。
        features: 当前录制会话的特征字典。

    抛出：
        ValueError: 如果任何检查的元数据字段不匹配。
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "数据集元数据兼容性检查失败，存在以下不匹配:\n" + "\n".join(mismatches)
        )
