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

from typing import Any

import torch

from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.processor import (
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action


def make_classifier_processor(
    config: RewardClassifierConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为奖励分类器构建预处理器和后处理器管道。

    预处理管道通过以下步骤为分类器准备输入数据：
    1. 基于数据集统计信息对输入和输出特征进行归一化。
    2. 将数据移动到指定的设备。

    后处理管道通过以下步骤处理分类器的输出：
    1. 将数据移动到 CPU。
    2. 应用恒等步骤，因为输出 logits 不需要反归一化。

    参数:
        config: RewardClassifier 的配置对象。
        dataset_stats: 用于归一化的统计信息字典。
        preprocessor_kwargs: 预处理器管道的额外参数。
        postprocessor_kwargs: 后处理器管道的额外参数。

    返回:
        包含配置好的预处理器和后处理器管道的元组。
    """

    input_steps = [
        NormalizerProcessorStep(
            features=config.input_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        NormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [DeviceProcessorStep(device="cpu"), IdentityProcessorStep()]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name="classifier_preprocessor",
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name="classifier_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
