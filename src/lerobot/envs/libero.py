#!/usr/bin/env python

# 版权所有 2025 HuggingFace Inc. 团队。保留所有权利。
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
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """将相机名称标准化为非空字符串列表。

    Args:
        camera_name: 相机名称，可以是逗号分隔的字符串或字符串序列。

    Returns:
        相机名称列表。

    Raises:
        TypeError: 如果 camera_name 不是字符串或字符串序列。
        ValueError: 如果解析后的相机名称列表为空。
    """
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name 必须是字符串或字符串序列，但得到 {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name 解析后为空列表。")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """根据名称实例化 LIBERO 测试套件并进行验证。

    Args:
        name: LIBERO 测试套件的名称。

    Returns:
        LIBERO 测试套件实例。

    Raises:
        ValueError: 如果测试套件名称未知或测试套件没有任务。
    """
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"未知的 LIBERO 套件 '{name}'。可用套件：{', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"套件 '{name}' 没有任务。")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """验证并标准化任务 ID。如果为 None，则返回所有任务。

    Args:
        total_tasks: 任务总数。
        task_ids: 要选择的任务 ID 列表，如果为 None 则选择所有任务。

    Returns:
        排序后的任务 ID 列表。

    Raises:
        ValueError: 如果任务 ID 超出有效范围。
    """
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"任务 ID {t} 超出有效范围 [0, {total_tasks - 1}]。")
    return ids


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    """获取指定任务的初始状态。

    Args:
        task_suite: LIBERO 测试套件。
        i: 任务索引。

    Returns:
        任务的初始状态数组。
    """
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614 安全性已验证
    return init_states


def get_libero_dummy_action():
    """获取虚拟/无操作动作，用于在机器人不动作的情况下推进模拟。"""
    return [0, 0, 0, 0, 0, 0, -1]


OBS_STATE_DIM = 8
ACTION_DIM = 7
AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # 最长训练演示有 193 步
    "libero_object": 280,  # 最长训练演示有 254 步
    "libero_goal": 300,  # 最长训练演示有 270 步
    "libero_10": 520,  # 最长训练演示有 505 步
    "libero_90": 400,  # 最长训练演示有 373 步
}


class LiberoEnv(gym.Env):
    """LIBERO 环境的 Gym 接口。

    这是一个 Gymnasium 环境包装器，用于 LIBERO 基准测试任务。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
    ):
        """初始化 LIBERO 环境。

        Args:
            task_suite: LIBERO 测试套件对象。
            task_id: 要运行的任务 ID。
            task_suite_name: 测试套件名称（例如 "libero_10"）。
            camera_name: 相机名称或相机名称列表。
            obs_type: 观察类型（"pixels" 或 "pixels_agent_pos"）。
            render_mode: 渲染模式。
            observation_width: 观察图像宽度。
            observation_height: 观察图像高度。
            visualization_width: 可视化图像宽度。
            visualization_height: 可视化图像高度。
            init_states: 是否使用预定义的初始状态。
            episode_index: 情节索引（用于选择初始状态）。
            camera_name_mapping: 相机名称映射字典。
            num_steps_wait: 重置后等待的步数（用于物理稳定）。
        """
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # 例如：agentview_image（主相机）或 robot0_eye_in_hand_image（手腕相机）

        # 将原始相机名称映射到 "image" 和 "image2"。
        # 预处理步骤 `preprocess_observation` 将在这些名称前加上 `.images.*` 前缀，
        # 遵循 LeRobot 约定（例如 `observation.images.image`、`observation.images.image2`）。
        # 这确保策略始终以期望的格式接收观察值，而不管原始相机命名如何。
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        # 仅加载一次并保存
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index  # 将每个子环境绑定到固定的初始状态

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "LiberoEnv 不支持 'state' 观察类型。"
                "请切换到基于图像的观察类型（例如 'pixels' 或 'pixels_agent_pos'）。"
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=AGENT_POS_LOW,
                        high=AGENT_POS_HIGH,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        """渲染环境的当前状态。

        Returns:
            当前观察的主相机图像。
        """
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        """为指定任务创建 LIBERO 环境。

        Args:
            task_suite: LIBERO 测试套件。
            task_id: 任务 ID。

        Returns:
            配置好的 LIBERO 环境实例。
        """
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """将原始 LIBERO 观察格式化为 LeRobot 格式。

        Args:
            raw_obs: 来自 LIBERO 环境的原始观察字典。

        Returns:
            格式化后的观察字典。

        Raises:
            NotImplementedError: 如果观察类型不支持。
        """
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            image = image[::-1, ::-1]  # 旋转 180 度
            images[self.camera_name_mapping[camera_name]] = image
        # 将机器人状态拼接为：末端执行器位置 + 末端执行器四元数转轴角 + 夹爪位置
        state = np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        )
        agent_pos = state
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}
        if self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images.copy(),
                "agent_pos": agent_pos,
            }
        raise NotImplementedError(
            f"LiberoEnv 不支持观察类型 '{self.obs_type}'。"
            "请切换到基于图像的观察类型（例如 'pixels' 或 'pixels_agent_pos'）。"
        )

    def reset(self, seed=None, **kwargs):
        """重置环境到初始状态。

        Args:
            seed: 随机种子。
            **kwargs: 其他关键字参数。

        Returns:
            observation: 初始观察。
            info: 信息字典。
        """
        super().reset(seed=seed)
        self._env.seed(seed)
        if self.init_states and self._init_states is not None:
            self._env.set_init_state(self._init_states[self._init_state_id])
        raw_obs = self._env.reset()

        # 重置后，物体可能不稳定（略微漂浮、相互交叉等）。
        # 使用无操作动作推进模拟器几帧，让一切稳定下来。
        # 增加此值可以提高重置之间的确定性和可重复性。
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行一个动作并返回结果。

        Args:
            action: 要执行的动作，必须是一维数组。

        Returns:
            observation: 新的观察。
            reward: 奖励值。
            terminated: 是否终止（完成或成功）。
            truncated: 是否截断。
            info: 信息字典。

        Raises:
            ValueError: 如果动作不是一维数组。
        """
        if action.ndim != 1:
            raise ValueError(
                f"期望动作为一维数组（形状为 (action_dim,)），"
                f"但得到形状 {action.shape}，维度为 {action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info["is_success"] = is_success

        observation = self._format_raw_obs(raw_obs)
        if done:
            self.reset()
            info.update(
                {
                    "task": self.task,
                    "task_id": self.task_id,
                    "done": done,
                    "is_success": is_success,
                }
            )
        truncated = False  # LIBERO 不使用截断，仅使用终止
        return observation, reward, terminated, truncated, info

    def close(self):
        """关闭环境并释放资源。"""
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], LiberoEnv]]:
    """为单个（套件，任务ID）构建 n_envs 个工厂可调用对象。

    Args:
        suite: LIBERO 测试套件。
        suite_name: 套件名称。
        task_id: 任务 ID。
        n_envs: 环境数量。
        camera_names: 相机名称列表。
        init_states: 是否使用初始状态。
        gym_kwargs: 传递给环境的额外参数。

    Returns:
        环境工厂函数列表。
    """

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_index=episode_index,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- 主要 API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """创建向量化的 LIBERO 环境，返回一致的结构。

    Args:
        task: 单个套件名称或逗号分隔的套件名称列表。
        n_envs: 每个任务的并行环境数量（情节索引 = 0..n_envs-1）。
        gym_kwargs: 传递给环境的额外参数，可以包含 task_ids 列表来限制每个套件的任务。
        camera_name: 相机名称或相机名称列表。
        init_states: 是否使用预定义的初始状态。
        env_cls: 用于包装环境工厂列表的可调用对象。

    Returns:
        dict[suite_name][task_id] -> vec_env（env_cls([...])，包含恰好 n_envs 个工厂）

    Raises:
        ValueError: 如果 env_cls 不是可调用对象或 n_envs 不是正整数。

    Notes:
        - n_envs 是每个任务的滚动数量（episode_index = 0..n_envs-1）。
        - `task` 可以是单个套件或逗号分隔的套件列表。
        - 可以在 `gym_kwargs` 中传递 `task_ids`（list[int]）来限制每个套件的任务。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls 必须是一个可调用对象，用于包装环境工厂可调用对象列表。")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs 必须是正整数，但得到 {n_envs}。")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # 可选：限制为特定任务

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` 必须包含至少一个 LIBERO 套件名称。")

    print(
        f"正在创建 LIBERO 环境 | 套件={suite_names} | 每任务环境数={n_envs} | 使用初始状态={init_states}"
    )
    if task_ids_filter is not None:
        print(f"限制为任务 ID={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)

        if not selected:
            raise ValueError(f"套件 '{suite_name}' 未选择任何任务（可用任务数：{total}）。")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
            )
            out[suite_name][tid] = env_cls(fns)
            print(f"已构建向量环境 | 套件={suite_name} | 任务ID={tid} | 环境数={n_envs}")

    # 返回普通字典（非 defaultdict）以确保可预测性
    return {suite: dict(task_map) for suite, task_map in out.items()}
