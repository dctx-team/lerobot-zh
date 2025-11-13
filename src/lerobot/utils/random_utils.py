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
import random
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict
from lerobot.utils.constants import RNG_STATE


def serialize_python_rng_state() -> dict[str, torch.Tensor]:
    """
    以 dict[str, torch.Tensor] 的扁平字典形式返回 `random` 的随机数生成器状态，
    以便使用 `safetensors.save_file()` 或 `torch.save()` 保存。
    """
    py_state = random.getstate()
    return {
        "py_rng_version": torch.tensor([py_state[0]], dtype=torch.int64),
        "py_rng_state": torch.tensor(py_state[1], dtype=torch.int64),
    }


def deserialize_python_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    从 `serialize_python_rng_state()` 产生的字典中恢复 `random` 的随机数生成器状态。
    """
    py_state = (rng_state_dict["py_rng_version"].item(), tuple(rng_state_dict["py_rng_state"].tolist()), None)
    random.setstate(py_state)


def serialize_numpy_rng_state() -> dict[str, torch.Tensor]:
    """
    以 dict[str, torch.Tensor] 的扁平字典形式返回 `numpy` 的随机数生成器状态，
    以便使用 `safetensors.save_file()` 或 `torch.save()` 保存。
    """
    np_state = np.random.get_state()
    # 确保 numpy 没有破坏性更改
    assert np_state[0] == "MT19937"
    return {
        "np_rng_state_values": torch.tensor(np_state[1], dtype=torch.int64),
        "np_rng_state_index": torch.tensor([np_state[2]], dtype=torch.int64),
        "np_rng_has_gauss": torch.tensor([np_state[3]], dtype=torch.int64),
        "np_rng_cached_gaussian": torch.tensor([np_state[4]], dtype=torch.float32),
    }


def deserialize_numpy_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    从 `serialize_numpy_rng_state()` 产生的字典中恢复 `numpy` 的随机数生成器状态。
    """
    np_state = (
        "MT19937",
        rng_state_dict["np_rng_state_values"].numpy(),
        rng_state_dict["np_rng_state_index"].item(),
        rng_state_dict["np_rng_has_gauss"].item(),
        rng_state_dict["np_rng_cached_gaussian"].item(),
    )
    np.random.set_state(np_state)


def serialize_torch_rng_state() -> dict[str, torch.Tensor]:
    """
    以 dict[str, torch.Tensor] 的扁平字典形式返回 `torch` 的随机数生成器状态，
    以便使用 `safetensors.save_file()` 或 `torch.save()` 保存。
    """
    torch_rng_state_dict = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        torch_rng_state_dict["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
    return torch_rng_state_dict


def deserialize_torch_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    从 `serialize_torch_rng_state()` 产生的字典中恢复 `torch` 的随机数生成器状态。
    """
    torch.set_rng_state(rng_state_dict["torch_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state" in rng_state_dict:
        torch.cuda.set_rng_state(rng_state_dict["torch_cuda_rng_state"])


def serialize_rng_state() -> dict[str, torch.Tensor]:
    """
    以 dict[str, torch.Tensor] 的扁平字典形式返回 `random`、`numpy` 和 `torch` 的随机数生成器状态，
    以便使用 `safetensors.save_file()` 或 `torch.save()` 保存。
    """
    py_rng_state_dict = serialize_python_rng_state()
    np_rng_state_dict = serialize_numpy_rng_state()
    torch_rng_state_dict = serialize_torch_rng_state()

    return {
        **py_rng_state_dict,
        **np_rng_state_dict,
        **torch_rng_state_dict,
    }


def deserialize_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    从 `serialize_rng_state()` 产生的字典中恢复 `random`、`numpy` 和 `torch` 的随机数生成器状态。
    """
    py_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("py")}
    np_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("np")}
    torch_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("torch")}

    deserialize_python_rng_state(py_rng_state_dict)
    deserialize_numpy_rng_state(np_rng_state_dict)
    deserialize_torch_rng_state(torch_rng_state_dict)


def save_rng_state(save_dir: Path) -> None:
    rng_state_dict = serialize_rng_state()
    flat_rng_state_dict = flatten_dict(rng_state_dict)
    save_file(flat_rng_state_dict, save_dir / RNG_STATE)


def load_rng_state(save_dir: Path) -> None:
    flat_rng_state_dict = load_file(save_dir / RNG_STATE)
    rng_state_dict = unflatten_dict(flat_rng_state_dict)
    deserialize_rng_state(rng_state_dict)


def get_rng_state() -> dict[str, Any]:
    """获取 `random`、`numpy` 和 `torch` 的随机状态。"""
    random_state_dict = {
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state_dict["torch_cuda_random_state"] = torch.cuda.random.get_rng_state()
    return random_state_dict


def set_rng_state(random_state_dict: dict[str, Any]):
    """设置 `random`、`numpy` 和 `torch` 的随机状态。

    参数:
        random_state_dict: 由 `get_rng_state` 返回形式的字典。
    """
    random.setstate(random_state_dict["random_state"])
    np.random.set_state(random_state_dict["numpy_random_state"])
    torch.random.set_rng_state(random_state_dict["torch_random_state"])
    if torch.cuda.is_available():
        torch.cuda.random.set_rng_state(random_state_dict["torch_cuda_random_state"])


def set_seed(seed) -> None:
    """设置随机种子以确保可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def seeded_context(seed: int) -> Generator[None, None, None]:
    """在进入上下文时设置种子，在退出时恢复先前的随机状态。

    示例用法:

    ```
    a = random.random()  # 产生某个随机数
    with seeded_context(1337):
        b = random.random()  # 产生另一个随机数
    c = random.random()  # 产生另一个随机数，但与从未生成 `b` 时相同
    ```
    """
    random_state_dict = get_rng_state()
    set_seed(seed)
    yield None
    set_rng_state(random_state_dict)
