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

import logging
from collections import deque

import torch
from torch import nn


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # 忽略队列中不存在的键（将确保队列拥有所需键的责任留给调用者）。
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # 通过多次复制第一个观测值来初始化，直到队列填满
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # 将最新观测值添加到队列
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """通过检查模块的一个参数来获取模块的设备。

    注意：假设所有参数都在同一设备上
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """通过检查模块的一个参数来获取模块参数的数据类型。

    注意：假设所有参数都具有相同的数据类型。
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    在给定输入形状的情况下计算 PyTorch 模块的输出形状。

    参数：
        module (nn.Module): 一个 PyTorch 模块
        input_shape (tuple): 表示输入形状的元组，例如 (batch_size, channels, height, width)

    返回：
        tuple: 模块的输出形状。
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """在加载模型时记录缺失和意外的键。

    参数：
        missing_keys (list[str]): 预期但未找到的键。
        unexpected_keys (list[str]): 找到但不在预期中的键。
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")
