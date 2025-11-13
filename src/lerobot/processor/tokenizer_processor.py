#!/usr/bin/env python

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

"""
此脚本定义了一个处理器，用于对来自环境转换的自然语言指令进行分词。

它使用来自 Hugging Face `transformers` 库的分词器将任务描述（文本）转换为
令牌 ID 和注意力掩码，然后将它们添加到观测字典中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available

from .core import EnvTransition, TransitionKey
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

# 用于类型检查和延迟加载的条件导入
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    用于对自然语言任务描述进行分词的处理器步骤。

    此步骤从 `EnvTransition` 的 `complementary_data` 中提取任务字符串，
    使用 Hugging Face `transformers` 分词器对其进行分词，并将生成的
    令牌 ID 和注意力掩码添加到 `observation` 字典中。

    需要安装 `transformers` 库。

    属性：
        tokenizer_name: Hugging Face Hub 中预训练分词器的名称（例如 "bert-base-uncased"）。
        tokenizer: 预初始化的分词器对象。如果提供，则忽略 `tokenizer_name`。
        max_length: 填充或截断序列的最大长度。
        task_key: `complementary_data` 中存储任务字符串的键。
        padding_side: 填充的一侧（'left' 或 'right'）。
        padding: 填充策略（'max_length'、'longest' 等）。
        truncation: 是否截断长度超过 `max_length` 的序列。
        input_tokenizer: 内部分词器实例，在初始化期间加载。
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # Use `Any` for compatibility without a hard dependency
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # 内部分词器实例（不是配置的一部分）
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        在创建数据类后初始化分词器。

        它检查 `transformers` 库的可用性，并从提供的对象
        或通过 Hugging Face Hub 的名称加载分词器。

        异常：
            ImportError: 如果未安装 `transformers` 库。
            ValueError: 如果既未提供 `tokenizer` 也未提供 `tokenizer_name`。
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:
            # 直接使用提供的分词器对象
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        从转换的补充数据中提取任务描述。

        参数：
            transition: 环境转换。

        返回：
            任务字符串列表，如果未找到任务键或值为 None，则返回 None。
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # 将其标准化为分词器的字符串列表
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        对任务描述进行分词并将其添加到观测字典中。

        此方法检索任务，对其进行分词，将生成的张量移动到
        与转换中其他数据相同的设备，并更新观测。

        参数：
            observation: 原始观测字典。

        返回：
            更新后的观测字典，包括令牌 ID 和注意力掩码。
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # 对任务进行分词（这将创建 CPU 张量）
        tokenized_prompt = self._tokenize_text(task)

        # 从转换中的现有张量检测设备以确保一致性
        target_device = self._detect_device(self.transition)

        # 将新的分词张量移动到检测到的设备
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # 创建新的观测字典以避免就地修改原始字典
        new_observation = dict(observation)

        # 将分词数据添加到观测
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        从转换中的现有张量检测 torch.device。

        它首先检查观测字典中的张量，然后检查动作张量。

        参数：
            transition: 环境转换。

        返回：
            检测到的 `torch.device`，如果未找到张量则返回 None。
        """
        # 首先检查观测张量（最有可能找到张量的地方）
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # 回退到检查动作张量
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # 未找到张量，默认为 CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        分词器调用的包装器。

        参数：
            text: 要分词的字符串或字符串列表。

        返回：
            包含分词后的 'input_ids' 和 'attention_mask' 的字典，作为 PyTorch 张量。
        """
        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器的可序列化配置。

        注意：分词器对象本身不会被序列化。如果处理器是使用
        分词器名称初始化的，该名称将包含在配置中。

        返回：
            包含处理器配置参数的字典。
        """
        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

        # 仅在使用 tokenizer_name 创建分词器时才保存它
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        为语言令牌和注意力掩码添加特征定义。

        这会更新策略特征字典以包含添加到观测的新数据，
        确保下游组件了解它们的形状和类型。

        参数：
            features: 现有策略特征的字典。

        返回：
            更新后的策略特征字典。
        """
        # 如果令牌 ID 的特征尚不存在，则添加它
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # 如果注意力掩码的特征尚不存在，则添加它
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features
