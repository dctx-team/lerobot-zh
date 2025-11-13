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

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .converters import to_tensor
from .core import EnvAction, EnvTransition, PolicyAction
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("torch2numpy_action_processor")
@dataclass
class Torch2NumpyActionProcessorStep(ActionProcessorStep):
    """
    将 PyTorch 张量动作转换为 NumPy 数组。

    当策略的输出（通常是 torch.Tensor）需要传递给期望 NumPy 数组的环境或组件时，
    此步骤非常有用。

    属性：
        squeeze_batch_dim: 如果为 True，则在数组的第一个维度大小为 1 时移除该维度。
                           这对于将大小为 (1, D) 的批处理动作转换为大小为 (D,) 的单个动作很有用。
    """

    squeeze_batch_dim: bool = True

    def action(self, action: PolicyAction) -> EnvAction:
        if not isinstance(action, PolicyAction):
            raise TypeError(
                f"Expected PolicyAction or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        numpy_action = action.detach().cpu().numpy()

        # 移除批次维度但保留动作维度。
        # 仅在存在批次维度（第一个维度 == 1）时压缩。
        if (
            self.squeeze_batch_dim
            and numpy_action.shape
            and len(numpy_action.shape) > 1
            and numpy_action.shape[0] == 1
        ):
            numpy_action = numpy_action.squeeze(0)

        return numpy_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("numpy2torch_action_processor")
@dataclass
class Numpy2TorchActionProcessorStep(ProcessorStep):
    """当动作存在时，将 NumPy 数组动作转换为 PyTorch 张量。"""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """如果动作存在，则将 NumPy 动作转换为 torch 张量，否则直接通过。"""
        from .core import TransitionKey

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, EnvAction):
                raise TypeError(
                    f"Expected np.ndarray or None, got {type(action).__name__}. "
                    "Use appropriate processor for non-tensor actions."
                )
            torch_action = to_tensor(action, dtype=None)  # 保留原始数据类型
            new_transition[TransitionKey.ACTION] = torch_action

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
