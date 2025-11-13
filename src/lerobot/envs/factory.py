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
import importlib

import gymnasium as gym

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """根据配置创建一个健身房环境向量环境。

    参数:
        cfg (EnvConfig): 要实例化的环境的配置。
        n_envs (int, 可选): 要返回的并行环境数量。默认为 1。
        use_async_envs (bool, 可选): 是否返回 AsyncVectorEnv 或 SyncVectorEnv。默认为
            False。

    异常:
        ValueError: 如果 n_envs < 1
        ModuleNotFoundError: 如果请求的环境包未安装

    返回:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            从套件名称到索引向量化环境的映射。
            - 对于多任务基准测试（例如 LIBERO）：每个套件一个条目，每个 task_id 一个向量环境。
            - 对于单任务环境：一个套件条目（cfg.type），task_id=0。

    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    if "libero" in cfg.type:
        from lerobot.envs.libero import create_libero_envs

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    package_name = f"gym_{cfg.type}"
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    def _make_one():
        return gym.make(gym_handle, disable_env_checker=cfg.disable_env_checker, **(cfg.gym_kwargs or {}))

    vec = env_cls([_make_one for _ in range(n_envs)])

    # 标准化为 {suite: {task_id: vec_env}} 以保持一致性
    suite_name = cfg.type  # 例如："pusht"、"aloha"
    return {suite_name: {0: vec}}
