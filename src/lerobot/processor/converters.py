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

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, DONE, OBS_PREFIX, REWARD, TRUNCATED

from .core import EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    将各种数据类型转换为PyTorch张量，支持配置选项

    这是一个统一的张量转换函数，使用单分派来适当处理不同的输入类型。

    Args:
        value: 要转换的输入值（张量、数组、标量、序列等）
        dtype: 目标张量数据类型。如果为None，则保留原始数据类型
        device: 张量的目标设备

    Returns:
        PyTorch张量

    Raises:
        TypeError: 如果输入类型不受支持
    """
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理现有PyTorch张量的转换"""
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    *,
    dtype=torch.float32,
    device=None,
    **kwargs,
) -> torch.Tensor:
    """处理numpy数组的转换"""
    # 检查numpy标量（零维数组）并将其视为标量
    if value.ndim == 0:
        # Numpy标量应转换为零维张量
        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)

    # 从numpy数组创建张量
    tensor = torch.from_numpy(value)

    # 如果指定了dtype和device，则应用转换
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理标量值的转换，包括numpy标量"""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理序列（列表、元组）的转换"""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    """通过递归转换字典的值来处理字典的转换"""
    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):
            # 递归处理嵌套字典
            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue

        # 将单个值转换为张量
        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    """
    将PyTorch张量转换为numpy数组或标量（如果适用）

    如果输入不是张量，则返回原样。

    Args:
        x: 输入，可以是张量或任何其他类型

    Returns:
        numpy数组、标量或原始输入
    """
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """
    从批次字典中提取补充数据

    包括填充标志、任务描述和索引。

    Args:
        batch: 批次字典

    Returns:
        包含提取的补充数据的字典
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}

    return {**pad_keys, **task_key, **index_key, **task_index_key}


def create_transition(
    observation: dict[str, Any] | None = None,
    action: PolicyAction | RobotAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """
    创建带有合理默认值的`EnvTransition`字典

    Args:
        observation: 观测字典
        action: 动作字典
        reward: 标量奖励值
        done: 回合终止标志
        truncated: 回合截断标志
        info: 附加信息字典
        complementary_data: 补充数据字典

    Returns:
        完整的`EnvTransition`字典
    """
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def robot_action_observation_to_transition(
    action_observation: tuple[RobotAction, RobotObservation],
) -> EnvTransition:
    """
    将原始机器人动作和观测字典转换为标准化的`EnvTransition`

    Args:
        action: 来自遥操作设备或控制器的原始动作字典
        observation: 来自环境的原始观测字典

    Returns:
        包含格式化观测的`EnvTransition`
    """
    if not isinstance(action_observation, tuple):
        raise ValueError("action_observation should be a tuple type with an action and observation")

    action, observation = action_observation

    if action is not None and not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")

    if observation is not None and not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")

    return create_transition(action=action, observation=observation)


def robot_action_to_transition(action: RobotAction) -> EnvTransition:
    """
    将原始机器人动作字典转换为标准化的`EnvTransition`

    Args:
        action: 来自遥操作设备或控制器的原始动作字典

    Returns:
        包含格式化动作的`EnvTransition`
    """
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")
    return create_transition(action=action)


def observation_to_transition(observation: RobotObservation) -> EnvTransition:
    """
    将原始机器人观测字典转换为标准化的`EnvTransition`

    Args:
        observation: 来自环境的原始观测字典

    Returns:
        包含格式化观测的`EnvTransition`
    """
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")
    return create_transition(observation=observation)


def transition_to_robot_action(transition: EnvTransition) -> RobotAction:
    """
    从`EnvTransition`中提取机器人的原始动作字典

    此函数搜索格式为"action.*.pos"或"action.*.vel"的键，
    并将它们转换为适合发送到机器人控制器的扁平字典。

    Args:
        transition: 包含动作的`EnvTransition`

    Returns:
        表示原始机器人动作的字典
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type (dict) got {type(action)}")
    return transition.get(TransitionKey.ACTION)


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """
    将`EnvTransition`转换为`PolicyAction`
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return action


def transition_to_observation(transition: EnvTransition) -> RobotObservation:
    """
    将`EnvTransition`转换为`RobotObservation`
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    observation = transition.get(TransitionKey.OBSERVATION)
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation (dict) type got {type(observation)}")
    return observation


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    """
    将`PolicyAction`转换为`EnvTransition`
    """
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """
    将来自数据集/数据加载器的批次字典转换为`EnvTransition`

    此函数将批次中识别的键映射到`EnvTransition`结构，
    用合理的默认值填充缺失的键。

    Args:
        batch: 批次字典

    Returns:
        `EnvTransition`字典

    Raises:
        ValueError: 如果输入不是字典
    """

    # 验证输入类型
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    # 提取观测和补充数据键
    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """
    将`EnvTransition`转换回LeRobot中使用的规范批次格式

    这是`batch_to_transition`的逆操作。

    Args:
        transition: 要转换的`EnvTransition`

    Returns:
        具有规范LeRobot字段名称的批次字典
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        "info": transition.get(TransitionKey.INFO, {}),
    }

    # 添加补充数据
    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)

    # 展平观测字典
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


def identity_transition(transition: EnvTransition) -> EnvTransition:
    """
    转换的恒等函数，返回未更改的输入

    作为处理管道中的默认值或占位符很有用。

    Args:
        tr: `EnvTransition`

    Returns:
        相同的`EnvTransition`
    """
    return transition
