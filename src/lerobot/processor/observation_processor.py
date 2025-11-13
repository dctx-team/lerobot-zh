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

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessorStep(ObservationProcessorStep):
    """
    将标准 Gymnasium 观测处理为 LeRobot 格式。

    此步骤处理来自典型观测字典的图像和状态数据，
    为在 LeRobot 策略中使用做准备。

    **图像处理：**
    -   将通道在后 (H, W, C) 的 `uint8` 图像转换为通道在前 (C, H, W) 的
        `float32` 张量。
    -   将像素值从 [0, 255] 范围归一化到 [0, 1]。
    -   如果尚未存在批次维度，则添加一个。
    -   识别 `"pixels"` 键下的单个图像并将其映射到
        `"observation.image"`。
    -   识别 `"pixels"` 键下的图像字典并将它们映射到
        `"observation.images.{camera_name}"`。

    **状态处理：**
    -   将 `"environment_state"` 键映射到 `"observation.environment_state"`。
    -   将 `"agent_pos"` 键映射到 `"observation.state"`。
    -   将 NumPy 数组转换为 PyTorch 张量。
    -   如果尚未存在批次维度，则添加一个。
    """

    def _process_single_image(self, img: np.ndarray) -> Tensor:
        """
        将单个 NumPy 图像数组处理为通道在前的归一化张量。

        参数：
            img: 表示图像的 NumPy 数组，预期为通道在后 (H, W, C) 格式，
                 dtype 为 `uint8`。

        返回：
            通道在前 (B, C, H, W) 格式的 `float32` PyTorch 张量，
            像素值归一化到 [0, 1] 范围。

        异常：
            ValueError: 如果输入图像不是通道在后格式或 dtype 不是 `uint8`。
        """
        # 转换为张量
        img_tensor = torch.from_numpy(img)

        # 如果需要，添加批次维度
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 验证图像格式
        _, h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")

        if img_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected torch.uint8 images, but got {img_tensor.dtype}")

        # 转换为通道优先格式
        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()

        # 转换为 float32 并归一化到 [0, 1]
        img_tensor = img_tensor.type(torch.float32) / 255.0

        return img_tensor

    def _process_observation(self, observation):
        """
        处理图像和状态观测。
        """

        processed_obs = observation.copy()

        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")

            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._process_single_image(img)

        if "environment_state" in processed_obs:
            env_state_np = processed_obs.pop("environment_state")
            env_state = torch.from_numpy(env_state_np).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos

        return processed_obs

    def observation(self, observation):
        return self._process_observation(observation)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        将特征键从 Gym 标准转换为 LeRobot 标准。

        此方法通过根据 LeRobot 的约定重命名键来标准化特征字典，
        确保可以正确构建策略。它处理各种原始键格式，
        包括带有 "observation." 前缀的格式。

        **重命名规则：**
        - `pixels` 或 `observation.pixels` -> `observation.image`
        - `pixels.{cam}` 或 `observation.pixels.{cam}` -> `observation.images.{cam}`
        - `environment_state` 或 `observation.environment_state` -> `observation.environment_state`
        - `agent_pos` 或 `observation.agent_pos` -> `observation.state`

        参数：
            features: 带有 Gym 风格键的策略特征字典。

        返回：
            带有标准化 LeRobot 键的策略特征字典。
        """
        # 构建由相同 FeatureType 存储桶键入的新特征映射
        # 我们假设调用者已将特征放置在正确的 FeatureType 中。
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {ft: {} for ft in features.keys()}

        exact_pairs = {
            "pixels": OBS_IMAGE,
            "environment_state": OBS_ENV_STATE,
            "agent_pos": OBS_STATE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }

        # 遍历所有传入的特征存储桶并归一化/移动每个条目
        for src_ft, bucket in features.items():
            for key, feat in list(bucket.items()):
                handled = False

                # 基于前缀的规则（例如 pixels.cam1 -> OBS_IMAGES.cam1）
                for old_prefix, new_prefix in prefix_pairs.items():
                    prefixed_old = f"{OBS_STR}.{old_prefix}"
                    if key.startswith(prefixed_old):
                        suffix = key[len(prefixed_old) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # 精确名称规则（pixels、environment_state、agent_pos）
                for old, new in exact_pairs.items():
                    if key == old or key == f"{OBS_STR}.{old}":
                        new_key = new
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # 默认：将键保留在同一源 FeatureType 存储桶中
                new_features[src_ft][key] = feat

        return new_features
