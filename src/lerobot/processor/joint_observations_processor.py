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
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STATE


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessorStep(ObservationProcessorStep):
    """
    计算并将关节速度信息附加到观测状态中。

    该步骤通过计算当前和上一次观测的关节位置之间的有限差分来计算每个关节的速度。
    生成的速度向量随后与原始状态向量连接。

    Attributes:
        dt: 观测之间的时间步长（delta time），以秒为单位，用于计算速度
        last_joint_positions: 存储上一步的关节位置，以便进行速度计算
    """

    dt: float = 0.1

    last_joint_positions: torch.Tensor | None = None

    def observation(self, observation: dict) -> dict:
        """
        计算关节速度并将其添加到观测状态中。

        Args:
            observation: 输入观测字典，预期包含带有关节位置的 `observation.state` 键

        Returns:
            一个新的观测字典，其 `observation.state` 张量扩展为包含关节速度

        Raises:
            ValueError: 如果观测中未找到 `observation.state`
        """
        # 获取当前关节位置（假设它们在 observation.state 中）
        current_positions = observation.get(OBS_STATE)
        if current_positions is None:
            raise ValueError(f"{OBS_STATE} is not in observation")

        # 如果尚未设置，则初始化上次关节位置
        if self.last_joint_positions is None:
            self.last_joint_positions = current_positions.clone()
            joint_velocities = torch.zeros_like(current_positions)
        else:
            # 计算速度
            joint_velocities = (current_positions - self.last_joint_positions) / self.dt

        self.last_joint_positions = current_positions.clone()

        # 用速度扩展观测
        extended_state = torch.cat([current_positions, joint_velocities], dim=-1)

        # 创建新的观测字典
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        返回用于序列化的步骤配置。

        Returns:
            包含时间步长 `dt` 的字典
        """
        return {
            "dt": self.dt,
        }

    def reset(self) -> None:
        """重置内部状态，清除上次已知的关节位置。"""
        self.last_joint_positions = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        更新 `observation.state` 特征以反映添加的速度。

        此方法将 `observation.state` 形状的第一维度大小加倍，
        以考虑位置和速度向量的连接。

        Args:
            features: 策略特征字典

        Returns:
            更新后的策略特征字典
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # 将形状加倍以考虑位置 + 速度
            new_shape = (original_feature.shape[0] * 2,) + original_feature.shape[1:]

            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessorStep(ObservationProcessorStep):
    """
    从机器人读取电机电流并将其附加到观测状态中。

    该步骤查询机器人的硬件接口以获取每个电机的当前电流，
    并将此信息连接到现有的状态向量中。

    Attributes:
        robot: `lerobot` Robot 类的实例，提供对硬件总线的访问
    """

    robot: Robot | None = None

    def observation(self, observation: dict) -> dict:
        """
        获取电机电流并将其添加到观测状态中。

        Args:
            observation: 输入观测字典

        Returns:
            一个新的观测字典，其 `observation.state` 张量扩展为包含电机电流

        Raises:
            ValueError: 如果未设置 `robot` 属性
        """
        # 从机器人状态获取当前值
        if self.robot is None:
            raise ValueError("Robot is not set")

        present_current_dict = self.robot.bus.sync_read("Present_Current")  # type: ignore[attr-defined]
        motor_currents = torch.tensor(
            [present_current_dict[name] for name in self.robot.bus.motors],  # type: ignore[attr-defined]
            dtype=torch.float32,
        ).unsqueeze(0)

        current_state = observation.get(OBS_STATE)
        if current_state is None:
            return observation

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # 创建新的观测字典
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        更新 `observation.state` 特征以反映添加的电机电流。

        此方法将 `observation.state` 形状的第一维度大小增加机器人中的电机数量。

        Args:
            features: 策略特征字典

        Returns:
            更新后的策略特征字典
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # 将电机电流维度添加到原始状态形状中
            num_motors = 0
            if hasattr(self.robot, "bus") and hasattr(self.robot.bus, "motors"):  # type: ignore[attr-defined]
                num_motors = len(self.robot.bus.motors)  # type: ignore[attr-defined]

            if num_motors > 0:
                new_shape = (original_feature.shape[0] + num_motors,) + original_feature.shape[1:]
                features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
        return features
