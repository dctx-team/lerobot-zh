#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry


@dataclass
class _NormalizationMixin:
    """
    提供归一化和反归一化核心功能的混入类。

    该类管理归一化统计信息（`stats`），将其转换为张量以进行高效计算，处理设备放置，
    并实现应用归一化转换（均值/标准差和最小值/最大值）的逻辑。该类设计为由具体的
    `ProcessorStep` 实现继承，不应直接使用。

    **统计信息覆盖保护：**
    当在构造期间显式提供统计信息时（例如，通过 `DataProcessorPipeline.from_pretrained()`
    中的 overrides），即使调用 `load_state_dict()` 也会保留这些统计信息。这允许用户覆盖
    已保存模型的归一化统计信息，同时保持模型状态的其余部分不变。

    示例：
        ```python
        # 常见用例：使用数据集统计信息覆盖
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset("my_dataset")
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": dataset.meta.stats}}
        )
        # 将使用 dataset.meta.stats，而不是已保存模型的统计信息

        # 自定义统计信息覆盖
        custom_stats = {"action": {"mean": [0.0], "std": [1.0]}}
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": custom_stats}}
        )
        ```

    属性：
        features: 将特征名称映射到 `PolicyFeature` 对象的字典，定义要处理的数据结构。
        norm_map: 将 `FeatureType` 映射到 `NormalizationMode` 的字典，指定每种特征类型
            使用的归一化方法。
        stats: 包含每个特征的归一化统计信息（例如，均值、标准差、最小值、最大值）的字典。
        device: 用于存储和执行张量操作的 PyTorch 设备。
        eps: 用于防止归一化计算中除零的小 epsilon 值。
        normalize_observation_keys: 可选的键集合，用于有选择地对特定观测特征应用归一化。
        _tensor_stats: 内部字典，保存作为 PyTorch 张量的归一化统计信息。
        _stats_explicitly_provided: 内部标志，跟踪在构造期间是否显式提供了统计信息
            （用于覆盖保护）。
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """
        在数据类构造后初始化混入类。

        此方法处理从 JSON 兼容格式（枚举变为字符串，元组变为列表）稳健反序列化 `features`
        和 `norm_map`，并将提供的 `stats` 字典转换为指定设备上的张量字典（`_tensor_stats`）。
        """
        # 跟踪统计信息是否被显式提供（不是 None 且不为空）
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)
        # 稳健的 JSON 反序列化处理（防护空映射）。
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        if self.norm_map:
            # 如果键是字符串（JSON），则重建枚举映射
            if all(isinstance(k, str) for k in self.norm_map.keys()):
                reconstructed = {}
                for ft_type_str, norm_mode_str in self.norm_map.items():
                    reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
                self.norm_map = reconstructed

        # 在初始化期间将统计信息转换为张量并移动到目标设备一次。
        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:
        """
        将处理器的归一化统计信息移动到指定设备。

        参数：
            device: 目标 PyTorch 设备。

        返回：
            类的实例，允许方法链式调用。
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        """
        将归一化统计信息作为扁平状态字典返回。

        在返回之前，所有张量都会移动到 CPU，这是保存状态字典的标准做法。

        返回：
            扁平字典，从 `'feature_name.stat_name'` 映射到 CPU 上的对应统计信息张量。
        """
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()  # 始终保存到 CPU
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """
        从状态字典加载归一化统计信息。

        加载的张量会移动到处理器配置的设备。

        **统计信息覆盖保护：**
        如果在构造期间显式提供了统计信息（例如，通过 `DataProcessorPipeline.from_pretrained()`
        中的 overrides），则会保留这些统计信息并忽略状态字典。这允许用户覆盖归一化统计信息，
        同时仍然加载模型状态的其余部分。

        此行为对于用户希望使具有不同统计信息的新数据集适应预训练模型而不重新训练整个模型的
        场景至关重要。

        参数：
            state: 扁平状态字典，键的格式为 `'feature_name.stat_name'`。

        注意：
            当由于显式提供而保留统计信息时，仅更新张量表示以确保与当前设备和 dtype 设置一致。
        """
        # 如果统计信息在构造期间被显式提供，则保留它们
        if self._stats_explicitly_provided and self.stats is not None:
            # 不从 state_dict 加载，保留显式提供的统计信息
            # 但确保 _tensor_stats 正确初始化
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)  # type: ignore[assignment]
            return

        # 正常行为：从 state_dict 加载统计信息
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            # 加载到处理器配置的设备。
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )

        # 从张量统计信息重建原始统计信息字典，以与 to() 方法和依赖于 self.stats 的其他函数兼容
        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {}
            for stat_name, tensor in tensor_dict.items():
                # 将张量转换回 python/numpy 格式
                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器配置的可序列化字典。

        此方法在将处理器保存到磁盘时使用，确保其配置可以稍后重建。

        返回：
            包含配置的 JSON 可序列化字典。
        """
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_observation(self, observation: dict[str, Any], inverse: bool) -> dict[str, Tensor]:
        """
        对观测字典中的所有相关特征应用归一化或反归一化。

        参数：
            observation: 要处理的观测字典。
            inverse: 如果为 `True`，应用反归一化；否则应用归一化。

        返回：
            包含转换后张量值的新观测字典。
        """
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                # 转换为张量但保留原始数据类型以用于适配逻辑
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        """
        对动作张量应用归一化或反归一化。

        参数：
            action: 要处理的动作张量。
            inverse: 如果为 `True`，应用反归一化；否则应用归一化。

        返回：
            转换后的动作张量。
        """
        processed_action = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)
        return processed_action

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """
        对张量应用归一化或反归一化转换的核心逻辑。

        此方法根据特征类型选择适当的归一化模式（例如，均值/标准差、最小值/最大值），
        并应用相应的数学运算。

        参数：
            tensor: 要转换的输入张量。
            key: 与张量对应的特征键。
            feature_type: 张量的 `FeatureType`。
            inverse: 如果为 `True`，应用逆转换（反归一化）。

        返回：
            转换后的张量。

        异常：
            ValueError: 如果遇到不支持的归一化模式。
        """
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # 为了与 Accelerate 兼容：确保统计信息与输入张量在同一设备和数据类型上
        if self._tensor_stats and key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD and "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            # 通过添加小的 epsilon 避免除零。
            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX and "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            # 当 min_val == max_val 时，用小的 epsilon 替换分母以防止除零。
            # 这始终将等于 min_val 的输入映射到 -1，确保稳定的转换。
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                # 从 [-1, 1] 映射回 [min, max]
                return (tensor + 1) / 2 * denom + min_val
            # 从 [min, max] 映射到 [-1, 1]
            return 2 * (tensor - min_val) / denom - 1

        # 如果缺少必要的统计信息，则返回未更改的输入。
        return tensor


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    对转换中的观测和动作应用归一化的处理器步骤。

    此类使用 `_NormalizationMixin` 的逻辑执行正向归一化（例如，缩放数据使其具有零均值和
    单位方差，或缩放到范围 [-1, 1]）。它通常在将数据馈送到策略之前用于预处理管道。
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_observation_keys: set[str] | None = None,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> NormalizerProcessorStep:
        """
        使用 `LeRobotDataset` 的统计信息创建 `NormalizerProcessorStep` 实例。

        参数：
            dataset: 从中提取归一化统计信息的数据集。
            features: 处理器的特征定义。
            norm_map: 从特征类型到归一化模式的映射。
            normalize_observation_keys: 要归一化的观测键的可选集合。
            eps: 用于数值稳定性的小 epsilon 值。
            device: 处理器的目标设备。

        返回：
            `NormalizerProcessorStep` 的新实例。
        """
        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_observation_keys=normalize_observation_keys,
            eps=eps,
            device=device,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # 处理观测归一化。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False
            )

        # 处理动作归一化。
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition

        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    对观测和动作应用反归一化的处理器步骤。

    此类反转归一化过程，将数据缩放回其原始范围。它通常在后处理管道中使用，将策略的
    归一化动作输出转换为机器人或环境可以执行的格式。
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        device: torch.device | str | None = None,
    ) -> UnnormalizerProcessorStep:
        """
        使用 `LeRobotDataset` 的统计信息创建 `UnnormalizerProcessorStep`。

        参数：
            dataset: 从中提取归一化统计信息的数据集。
            features: 处理器的特征定义。
            norm_map: 从特征类型到归一化模式的映射。
            device: 处理器的目标设备。

        返回：
            `UnnormalizerProcessorStep` 的新实例。
        """
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # 处理观测反归一化。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)

        # 处理动作反归一化。
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """
    替换现有 `PolicyProcessorPipeline` 实例中的归一化统计信息。

    此函数创建提供的管道的深拷贝，并更新其包含的任何 `NormalizerProcessorStep` 或
    `UnnormalizerProcessorStep` 的统计信息。这对于使训练的策略适应具有不同数据分布的
    新环境或数据集非常有用，而无需重建整个管道。

    参数：
        policy_processor: 要修改的策略处理器管道。
        stats: 要应用的新归一化统计信息字典。

    返回：
        具有更新统计信息的新 `PolicyProcessorPipeline` 实例。
    """
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            # 在正确的设备上重新初始化 tensor_stats。
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)  # type: ignore[assignment]
    return rp
