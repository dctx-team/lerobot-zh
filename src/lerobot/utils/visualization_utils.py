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

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """
    初始化 Rerun SDK 用于可视化控制循环。

    Args:
        session_name: Rerun 会话的名称,默认为 "lerobot_control_loop"。
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return (
        isinstance(x, float)
        or isinstance(x, numbers.Real)
        or isinstance(x, (np.integer, np.floating))
        or (isinstance(x, np.ndarray) and x.ndim == 0)
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    将观测和动作数据记录到 Rerun 用于实时可视化。

    此函数遍历提供的观测和动作字典,并将其内容发送到 Rerun 查看器。它适当地处理不同的数据类型:
    - 标量值(浮点数、整数)被记录为 `rr.Scalar`。
    - 类似图像的 3D NumPy 数组(例如,第一维为 1、3 或 4 通道)会从 CHW 格式转置为 HWC 格式,
      并记录为 `rr.Image`。
    - 1D NumPy 数组被记录为一系列独立的标量,每个元素都有索引。
    - 其他多维数组被展平并记录为独立的标量。

    如果键尚未包含命名空间,则会自动添加 "observation." 或 "action." 前缀。

    Args:
        observation: 包含要记录的观测数据的可选字典。
        action: 包含要记录的动作数据的可选字典。
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # 在需要时将 CHW 格式转换为 HWC 格式
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
                else:
                    # 对于更高维数组,回退到展平处理
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
