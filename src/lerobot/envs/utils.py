#!/usr/bin/env python

# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非遵守许可证，否则你不得使用此文件。
# 你可以在以下地址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解许可证下的特定语言权限和
# 限制。
import warnings
from collections.abc import Mapping, Sequence
from functools import singledispatch
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import get_channel_first_image_shape


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): 重构此代码以使用环境中的特征（无需硬编码）
    """将环境观察转换为 LeRobot 格式的观察。
    Args:
        observations: 来自 Gym 向量化环境的观察批次字典。
    Returns:
        观察批次字典，键已重命名为 LeRobot 格式，值为张量。
    """
    # 将环境观察映射到策略所期望的输入格式
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            # 多相机情况：将每个相机图像映射到 observation.images.{camera_name}
            imgs = {f"{OBS_IMAGES}.{key}": img for key, img in observations["pixels"].items()}
        else:
            # 单相机情况：映射到 observation.image
            imgs = {OBS_IMAGE: observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): 考虑使用 transforms.ToTensor()?
            img = torch.from_numpy(img)

            # 在非向量化环境中预处理观察时，我们需要添加批次维度。
            # 这适用于人机交互强化学习（HIL-SERL），此时只有一个环境。
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # 健全性检查：图像应该是通道在最后（HWC 格式）
            _, h, w, c = img.shape
            assert c < h and c < w, f"期望通道在最后的图像，但得到 {img.shape=}"

            # 健全性检查：图像应该是 uint8 类型
            assert img.dtype == torch.uint8, f"期望 torch.uint8 类型，但得到 {img.dtype=}"

            # 转换为通道在前（CHW）的 float32 类型，范围为 [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        # 环境状态（如 PushT 的完整环境状态）
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations[OBS_ENV_STATE] = env_state

    # TODO(rcadene): 通过在环境中移除 `obs_type="pixels"` 来启用仅像素基线
    # 智能体位置（机器人状态，如关节位置）
    agent_pos = torch.from_numpy(observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)
    return_observations[OBS_STATE] = agent_pos

    return return_observations


def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    """将环境配置特征转换为策略特征。

    Args:
        env_cfg: 环境配置对象。

    Returns:
        策略特征字典，键为映射后的策略键，值为 PolicyFeature 对象。

    Raises:
        ValueError: 如果视觉特征的维度不等于 3。

    Note:
        对于视觉特征，会将通道在最后的形状（HWC）转换为通道在前的形状（CHW）。
    """
    # TODO(aliberts, rcadene): 移除对键的硬编码，直接使用嵌套键
    # （还需要重构 preprocess_observation 并从策略中外部化归一化）
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"{key} 的维度数量不等于 3（形状={ft.shape}）")

            # 将 HWC 转换为 CHW
            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        # 使用环境的特征映射将环境键映射到策略键
        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature

    return policy_features


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    """检查向量化环境中的所有环境是否为相同类型。

    Args:
        env: Gym 向量化环境。

    Returns:
        如果所有环境类型相同则返回 True，否则返回 False。
    """
    first_type = type(env.envs[0])  # 获取第一个环境的类型作为基准
    return all(type(e) is first_type for e in env.envs)  # 快速类型检查（使用 is 而非 ==）


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    """检查环境属性并验证所有环境是否为相同类型。

    Args:
        env: Gym 向量化环境。

    Warns:
        如果环境缺少 'task_description' 和 'task' 属性，或者环境类型不同。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # 仅在此函数调用中应用过滤器

        if not (hasattr(env.envs[0], "task_description") and hasattr(env.envs[0], "task")):
            warnings.warn(
                "环境没有 'task_description' 和 'task' 属性。某些策略（如基于语言的策略）需要这些特征。",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "向量化环境中包含不同类型的环境。请确保从每个环境正确推断任务。将传递空任务字符串代替。",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    """根据第一个环境的属性将任务特征添加到观察字典中。

    Args:
        env: Gym 向量化环境。
        observation: 观察字典。

    Returns:
        包含任务特征的观察字典。

    Raises:
        TypeError: 如果 task_description 或 task 返回的不是字符串列表。
    """
    if hasattr(env.envs[0], "task_description"):
        # 优先使用 task_description（通常是自然语言描述）
        task_result = env.call("task_description")

        if isinstance(task_result, tuple):
            task_result = list(task_result)

        if not isinstance(task_result, list):
            raise TypeError(f"期望 task_description 返回列表，但得到 {type(task_result)}")
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("task_description 结果中的所有项必须是字符串")

        observation["task"] = task_result
    elif hasattr(env.envs[0], "task"):
        # 备用：使用 task（通常是任务名称）
        task_result = env.call("task")

        if isinstance(task_result, tuple):
            task_result = list(task_result)

        if not isinstance(task_result, list):
            raise TypeError(f"期望 task 返回列表，但得到 {type(task_result)}")
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("task 结果中的所有项必须是字符串")

        observation["task"] = task_result
    else:  # 对于没有语言指令的环境，例如 Aloha 转移立方体等
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]  # 使用空字符串作为占位符
    return observation


def _close_single_env(env: Any) -> None:
    """安全地关闭单个环境，捕获任何异常。

    Args:
        env: 要关闭的环境实例。
    """
    try:
        env.close()
    except Exception as exc:
        print(f"关闭环境 {env} 时发生异常：{exc}")


@singledispatch
def close_envs(obj: Any) -> None:
    """关闭环境或环境集合。

    这是一个单分派泛型函数，可以处理不同类型的环境容器：
    - Mapping（字典）: 递归关闭所有环境值
    - Sequence（列表、元组）: 递归关闭所有环境元素
    - gym.Env: 关闭单个环境

    Args:
        obj: 要关闭的环境、映射或序列。

    Raises:
        NotImplementedError: 如果对象类型无法识别。
    """
    # 默认情况：如果类型无法识别则抛出异常
    raise NotImplementedError(f"类型 {type(obj).__name__} 未实现 close_envs")


@close_envs.register
def _(env: Mapping) -> None:
    """关闭映射（字典）中的所有环境。

    Args:
        env: 包含环境的映射。
    """
    for v in env.values():
        if isinstance(v, Mapping):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(envs: Sequence) -> None:
    """关闭序列（列表、元组）中的所有环境。

    Args:
        envs: 包含环境的序列。

    Note:
        忽略字符串和字节序列（它们也是 Sequence 但不是环境容器）。
    """
    if isinstance(envs, (str, bytes)):
        return
    for v in envs:
        if isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            close_envs(v)  # 递归处理嵌套结构
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(env: gym.Env) -> None:
    """关闭单个 Gym 环境。

    Args:
        env: Gym 环境实例。
    """
    _close_single_env(env)
