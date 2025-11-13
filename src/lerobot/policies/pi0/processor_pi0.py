#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register(name="pi0_new_line_processor")
class Pi0NewLineProcessor(ComplementaryDataProcessorStep):
    """
    确保任务描述字符串以换行符结尾。

    此处理步骤是为了与 PaliGemma 分词器兼容而必需的，
    该分词器期望文本提示符的末尾有一个换行符。它处理补充数据中
    'task' 键的单个字符串和字符串列表。
    """

    def complementary_data(self, complementary_data):
        """
        如果 'task' 字段尚未包含换行符，则添加换行符。

        Args:
            complementary_data: 可能包含 'task' 键的字典，其值为字符串或字符串列表。

        Returns:
            包含修改后的 'task' 字段的新字典。
        """
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # 处理字符串和字符串列表
        if isinstance(task, str):
            # 单个字符串：如果不存在则添加换行符
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # 字符串列表：如果不存在则为每个字符串添加换行符
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # 如果 task 既不是字符串也不是字符串列表，则保持不变

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        此步骤不会更改特征定义。

        Args:
            features: 输入特征字典。

        Returns:
            未更改的特征字典。
        """
        return features


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    构建 PI0 策略的预处理器和后处理器管道。

    预处理管道通过以下步骤为模型准备输入数据：
    1. 重命名特征以匹配预训练配置。
    2. 基于数据集统计信息对输入和输出特征进行归一化。
    3. 添加批次维度。
    4. 为分词器兼容性在任务描述末尾添加换行符。
    5. 使用 PaliGemma 分词器对文本提示进行分词。
    6. 将所有数据移动到指定设备。

    后处理管道通过以下步骤处理模型的输出：
    1. 将数据移动到 CPU。
    2. 对输出特征进行反归一化以恢复其原始尺度。

    Args:
        config: PI0 策略的配置对象。
        dataset_stats: 用于归一化的统计信息字典。
        preprocessor_kwargs: 预处理器管道的额外参数。
        postprocessor_kwargs: 后处理器管道的额外参数。

    Returns:
        包含已配置的预处理器和后处理器管道的元组。
    """

    # 添加其余的处理器
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # 模仿与预训练相同的处理器
        AddBatchDimensionProcessorStep(),
        Pi0NewLineProcessor(),  # 在 PaliGemma 分词之前添加换行符
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
