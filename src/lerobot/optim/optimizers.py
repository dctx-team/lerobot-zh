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
import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """构建优化器。它可以是单个优化器或优化器字典。

        注意: 当您有不同的模型需要优化时，多个优化器很有用。
        例如，在强化学习设置中，您可以为策略使用一个优化器，为价值函数使用另一个优化器。

        返回:
            优化器或优化器字典。
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """具有不同参数组的多个 Adam 优化器配置。

    这会创建一个 Adam 优化器字典，每个优化器都有自己的超参数。

    参数:
        lr: 默认学习率（如果未为组指定则使用）
        weight_decay: 默认权重衰减（如果未为组指定则使用）
        optimizer_groups: 将参数组名称映射到其超参数的字典
        grad_clip_norm: 梯度裁剪范数
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build(self, params_dict: dict[str, list]) -> dict[str, torch.optim.Optimizer]:
        """构建多个 Adam 优化器。

        参数:
            params_dict: 将参数组名称映射到参数列表的字典
                         键应该与 optimizer_groups 中的键匹配

        返回:
            将参数组名称映射到其优化器的字典
        """
        optimizers = {}

        for name, params in params_dict.items():
            # 获取组特定的超参数或使用默认值
            group_config = self.optimizer_groups.get(name, {})

            # 使用合并参数（默认值 + 组特定值）创建优化器
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """将优化器状态保存到磁盘。

    参数:
        optimizer: 单个优化器或优化器字典。
        save_dir: 保存优化器状态的目录。
    """
    if isinstance(optimizer, dict):
        # 处理优化器字典
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        # 处理单个优化器
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """将单个优化器的状态保存到磁盘。

    参数:
        optimizer: 要保存状态的 PyTorch 优化器。
        save_dir: 保存优化器状态文件的目录。
    """
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)  # 保存优化器状态张量
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)  # 保存参数组配置


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """从磁盘加载优化器状态。

    参数:
        optimizer: 单个优化器或优化器字典。
        save_dir: 加载优化器状态的目录。

    返回:
        加载了状态的更新后的优化器。
    """
    if isinstance(optimizer, dict):
        # 处理优化器字典
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        # 处理单个优化器
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    """从磁盘加载单个优化器的状态。

    参数:
        optimizer: 要加载状态的 PyTorch 优化器。
        save_dir: 包含优化器状态文件的目录。

    返回:
        加载了状态的优化器。
    """
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)  # 加载优化器状态张量
    state = unflatten_dict(flat_state)

    # 处理 'state' 键可能不存在的情况（针对新创建的优化器）
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}  # 将字符串键转换为整数
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        # 从 JSON 文件加载参数组配置
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
