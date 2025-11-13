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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rename_observations_processor")
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """
    重命名观测字典中键的处理器步骤。

    此步骤对于创建标准化数据接口非常有用，可以将键从环境格式映射到 LeRobot 策略或
    其他下游组件期望的格式。

    属性：
        rename_map: 从旧键名映射到新键名的字典。观测中存在但不在此映射中的键将保持其原始名称。
    """

    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        processed_obs = {}
        for key, value in observation.items():
            if key in self.rename_map:
                processed_obs[self.rename_map[key]] = value
            else:
                processed_obs[key] = value

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """转换：
        - 观测中出现在 `rename_map` 中的每个键都被重命名为其值。
        - 不在 `rename_map` 中的键保持不变。
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            self.rename_map.get(k, k): v for k, v in features[PipelineFeatureType.OBSERVATION].items()
        }
        return new_features


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    使用提供的映射重命名统计信息字典中的顶级键。

    这是一个辅助函数，通常用于使归一化统计信息与重命名的观测或动作特征保持一致。
    它执行防御性深拷贝以避免修改原始 `stats` 字典。

    参数：
        stats: 嵌套的统计信息字典，其中顶级键是特征名称
               （例如，`{"observation.state": {"mean": 0.5}}`）。
        rename_map: 将旧特征名称映射到新特征名称的字典。

    返回：
        顶级键已重命名的新统计信息字典。如果输入 `stats` 为空，则返回空字典。
    """
    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed
