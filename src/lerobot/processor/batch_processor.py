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
此脚本定义了用于向环境转换的各个组件添加批次维度的处理器步骤。

这些步骤旨在处理动作、观测和补充数据，通过添加前导维度使它们适合批处理。
这是将数据输入神经网络模型之前的常见要求。
"""

from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .core import EnvTransition, PolicyAction
from .pipeline import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_action")
class AddBatchDimensionActionStep(PolicyActionProcessorStep):
    """
    处理器步骤：向一维张量动作添加批次维度。

    这对于从单个动作样本创建大小为1的批次很有用。
    """

    def action(self, action: PolicyAction) -> PolicyAction:
        """
        如果动作是一维张量，则添加批次维度。

        Args:
            action: 动作张量。

        Returns:
            添加了批次维度的动作张量。
        """
        if action.dim() != 1:
            return action
        return action.unsqueeze(0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。

        添加批次维度不会改变特征定义。

        Args:
            features: 策略特征字典。

        Returns:
            原始的策略特征字典。
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
class AddBatchDimensionObservationStep(ObservationProcessorStep):
    """
    处理器步骤：向观测添加批次维度。

    它处理不同类型的观测：
    - 状态向量（一维张量）。
    - 单个图像（三维张量）。
    - 多个图像的字典（三维张量）。
    """

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        向观测字典中基于张量的观测添加批次维度。

        Args:
            observation: 观测字典。

        Returns:
            张量已添加批次维度的观测字典。
        """
        # 处理状态观测 - 如果是 1D，则添加批次维度
        for state_key in [OBS_STATE, OBS_ENV_STATE]:
            if state_key in observation:
                state_value = observation[state_key]
                if isinstance(state_value, Tensor) and state_value.dim() == 1:
                    observation[state_key] = state_value.unsqueeze(0)

        # 处理单个图像观测 - 如果是 3D，则添加批次维度
        if OBS_IMAGE in observation:
            image_value = observation[OBS_IMAGE]
            if isinstance(image_value, Tensor) and image_value.dim() == 3:
                observation[OBS_IMAGE] = image_value.unsqueeze(0)

        # 处理多个图像观测 - 如果是 3D，则添加批次维度
        for key, value in observation.items():
            if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3:
                observation[key] = value.unsqueeze(0)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。

        添加批次维度不会改变特征定义。

        Args:
            features: 策略特征字典。

        Returns:
            原始的策略特征字典。
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
class AddBatchDimensionComplementaryDataStep(ComplementaryDataProcessorStep):
    """
    处理器步骤：向补充数据字段添加批次维度。

    处理特定的键，如'task'、'index'和'task_index'，使它们成为批处理的。
    - 'task'（字符串）包装在列表中。
    - 'index'和'task_index'（零维张量）获得批次维度。
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        向补充数据字典中的特定字段添加批次维度。

        Args:
            complementary_data: 补充数据字典。

        Returns:
            已添加批次维度的补充数据字典。
        """
        # 处理任务字段 - 将字符串包装在列表中以添加批次维度
        if "task" in complementary_data:
            task_value = complementary_data["task"]
            if isinstance(task_value, str):
                complementary_data["task"] = [task_value]

        # 处理索引字段 - 如果是 0D，则添加批次维度
        if "index" in complementary_data:
            index_value = complementary_data["index"]
            if isinstance(index_value, Tensor) and index_value.dim() == 0:
                complementary_data["index"] = index_value.unsqueeze(0)

        # 处理 task_index 字段 - 如果是 0D，则添加批次维度
        if "task_index" in complementary_data:
            task_index_value = complementary_data["task_index"]
            if isinstance(task_index_value, Tensor) and task_index_value.dim() == 0:
                complementary_data["task_index"] = task_index_value.unsqueeze(0)
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。

        添加批次维度不会改变特征定义。

        Args:
            features: 策略特征字典。

        Returns:
            原始的策略特征字典。
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class AddBatchDimensionProcessorStep(ProcessorStep):
    """
    一个复合处理器步骤，向整个环境转换添加批次维度。

    此步骤结合了动作、观测和补充数据的各个处理器，
    从单实例转换创建批处理转换（批次大小为1）。

    Attributes:
        to_batch_action_processor: 动作组件的处理器。
        to_batch_observation_processor: 观测组件的处理器。
        to_batch_complementary_data_processor: 补充数据组件的处理器。
    """

    to_batch_action_processor: AddBatchDimensionActionStep = field(
        default_factory=AddBatchDimensionActionStep
    )
    to_batch_observation_processor: AddBatchDimensionObservationStep = field(
        default_factory=AddBatchDimensionObservationStep
    )
    to_batch_complementary_data_processor: AddBatchDimensionComplementaryDataStep = field(
        default_factory=AddBatchDimensionComplementaryDataStep
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        将批处理过程应用于环境转换的所有相关部分。

        Args:
            transition: 要处理的环境转换。

        Returns:
            已添加批次维度的环境转换。
        """
        if transition[TransitionKey.ACTION] is not None:
            transition = self.to_batch_action_processor(transition)
        if transition[TransitionKey.OBSERVATION] is not None:
            transition = self.to_batch_observation_processor(transition)
        if transition[TransitionKey.COMPLEMENTARY_DATA] is not None:
            transition = self.to_batch_complementary_data_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。

        添加批次维度不会改变特征定义。

        Args:
            features: 策略特征字典。

        Returns:
            原始的策略特征字典。
        """
        # 注意：在转换特征时，我们忽略批次维度
        return features
