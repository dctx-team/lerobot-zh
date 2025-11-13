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

import re
from collections.abc import Sequence
from typing import Any

from lerobot.configs.types import PipelineFeatureType
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR


def create_initial_features(
    action: dict[str, Any] | None = None, observation: dict[str, Any] | None = None
) -> dict[PipelineFeatureType, dict[str, Any]]:
    """
    从动作和观测规格创建数据集的初始特征字典。

    参数：
        action：动作特征名称到其类型/形状的字典。
        observation：观测特征名称到其类型/形状的字典。

    返回：
        按 PipelineFeatureType 结构化的初始特征字典。
    """
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    if action:
        features[PipelineFeatureType.ACTION] = action
    if observation:
        features[PipelineFeatureType.OBSERVATION] = observation
    return features


# 基于正则表达式模式过滤状态/动作键的辅助函数。
def should_keep(key: str, patterns: tuple[str]) -> bool:
    if patterns is None:
        return True
    return any(re.search(pat, key) for pat in patterns)


def strip_prefix(key: str, prefixes_to_strip: tuple[str]) -> str:
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


# 定义要从特征键中剥离的前缀以获得简洁的名称。
# 处理完全限定形式（例如 "action.state"）和短形式（例如 "state"）。
PREFIXES_TO_STRIP = tuple(
    f"{token}." for const in (ACTION, OBS_STATE, OBS_IMAGES) for token in (const, const.split(".")[-1])
)


def aggregate_pipeline_dataset_features(
    pipeline: DataProcessorPipeline,
    initial_features: dict[PipelineFeatureType, dict[str, Any]],
    *,
    use_videos: bool = True,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """
    聚合和过滤管道特征以创建数据集就绪的特征字典。

    此函数使用管道转换初始特征，将它们分类为动作或观测（图像或状态），
    基于 `use_videos` 和 `patterns` 对它们进行过滤，最后格式化它们以便与
    Hugging Face LeRobot 数据集一起使用。

    参数：
        pipeline：要应用的 DataProcessorPipeline。
        initial_features：动作和观测的原始特征规格字典。
        use_videos：如果为 False，则排除图像特征。
        patterns：用于过滤动作和状态特征的正则表达式模式序列。
                  图像特征不受此过滤器影响。

    返回：
        为 Hugging Face LeRobot 数据集格式化的特征字典。
    """
    all_features = pipeline.transform_features(initial_features)

    # 用于分类和过滤特征的中间存储。
    processed_features: dict[str, dict[str, Any]] = {
        ACTION: {},
        OBS_STR: {},
    }
    images_token = OBS_IMAGES.split(".")[-1]

    # 遍历管道转换的所有特征。
    for ptype, feats in all_features.items():
        if ptype not in [PipelineFeatureType.ACTION, PipelineFeatureType.OBSERVATION]:
            continue

        for key, value in feats.items():
            # 1. 对特征进行分类。
            is_action = ptype == PipelineFeatureType.ACTION
            # 如果观测的键与图像相关的标记匹配，或者特征的形状为 3，则将观测分类为图像。
            # 所有其他观测都被视为状态。
            is_image = not is_action and (
                (isinstance(value, tuple) and len(value) == 3)
                or (
                    key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{images_token}.")
                    or f".{images_token}." in key
                )
            )

            # 2. 应用过滤规则。
            if is_image and not use_videos:
                continue
            if not is_image and not should_keep(key, patterns):
                continue

            # 3. 将特征添加到具有简洁名称的相应组中。
            name = strip_prefix(key, PREFIXES_TO_STRIP)
            if is_action:
                processed_features[ACTION][name] = value
            else:
                processed_features[OBS_STR][name] = value

    # 将处理后的特征转换为最终的数据集格式。
    dataset_features = {}
    if processed_features[ACTION]:
        dataset_features.update(hw_to_dataset_features(processed_features[ACTION], ACTION, use_videos))
    if processed_features[OBS_STR]:
        dataset_features.update(hw_to_dataset_features(processed_features[OBS_STR], OBS_STR, use_videos))

    return dataset_features
