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

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

from .core import PolicyAction, RobotAction
from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    将策略的扁平动作张量映射到结构化的增量动作字典。

    此步骤通常在策略输出连续动作向量后使用。
    它将向量分解为末端执行器（x, y, z）的增量移动命名组件，以及可选的夹爪。

    属性：
        use_gripper: 如果为 True，则假设张量的第 4 个元素是夹爪动作。
    """

    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): 添加旋转
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    将遥操作设备的增量动作映射到机器人目标动作以进行逆运动学计算。

    此步骤将增量移动字典（例如，来自游戏手柄）转换为目标动作格式，
    该格式包括"enabled"标志和目标末端执行器位置。它还处理缩放和噪声过滤。

    属性：
        position_scale: 用于缩放增量位置输入的因子。
        rotation_scale: 用于缩放增量旋转输入的因子（当前未使用）。
        noise_threshold: 低于此幅度的增量输入被视为噪声，不会触发"enabled"状态。
    """

    # 增量移动的缩放因子
    position_scale: float = 1.0
    rotation_scale: float = 0.0  # 游戏手柄/键盘无旋转增量
    noise_threshold: float = 1e-3  # 1 毫米阈值以过滤噪声

    def action(self, action: RobotAction) -> RobotAction:
        # 注意 (maractingi): 动作可以是来自 teleop_devices 的字典或来自策略的张量
        # TODO (maractingi): 从 teleop_devices 更改此 target_xyz 命名约定
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        # 确定遥控设备是否正在主动提供输入
        # 如果检测到任何显著的移动增量，则视为启用
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # 使用欧几里得范数表示位置
        enabled = position_magnitude > self.noise_threshold  # 小阈值以避免噪声

        # 适当地缩放增量
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # 对于游戏手柄/键盘，我们没有旋转输入，因此设置为 0
        # 未来可以为更复杂的遥控设备扩展这些
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # 使用机器人目标格式更新动作
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)

        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
