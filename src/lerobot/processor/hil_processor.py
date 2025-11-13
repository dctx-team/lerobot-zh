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

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    为提供遥操作事件的对象定义的最小协议。

    此协议定义了 `get_teleop_events()` 方法，允许处理器步骤与支持基于事件控制
    (如剧集终止或成功标记) 的遥操作器交互，而无需知道遥操作器的具体类。
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        从遥操作器获取额外的控制事件。

        返回:
            包含控制事件的字典，例如:
            - `is_intervention`: bool - 人类是否正在进行干预。
            - `terminate_episode`: bool - 是否终止当前剧集。
            - `success`: bool - 剧集是否成功。
            - `rerecord_episode`: bool - 是否重新记录剧集。
        """
        ...


# 限制为同时实现事件的 Teleoperator 子类的类型变量
TeleopWithEvents = TypeVar("TeleopWithEvents", bound=Teleoperator)


def _check_teleop_with_events(teleop: Teleoperator) -> None:
    """
    运行时检查遥操作器是否实现了 `HasTeleopEvents` 协议。

    参数:
        teleop: 要检查的遥操作器实例。

    抛出:
        TypeError: 如果遥操作器没有 `get_teleop_events` 方法。
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    将遥操作器的原始动作添加到转换的互补数据中。

    这对于人机协作场景非常有用，在这些场景中，人类的输入需要对下游处理器可用，
    例如，在干预期间覆盖策略的动作。

    属性:
        teleop_device: 从中获取动作的遥操作器实例。
    """

    teleop_device: Teleoperator

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        检索遥操作器的动作并将其添加到互补数据中。

        参数:
            complementary_data: 传入的互补数据字典。

        返回:
            在 `teleop_action` 键下添加了遥操作器动作的新字典。
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    将遥操作器控制事件(例如，终止、成功)添加到转换的 info 中。

    此步骤从支持基于事件交互的遥操作器中提取控制事件，使这些信号可供系统的其他部分使用。

    属性:
        teleop_device: 实现 `HasTeleopEvents` 协议的遥操作器实例。
    """

    teleop_device: TeleopWithEvents

    def __post_init__(self):
        """在初始化后验证提供的遥操作器是否支持事件。"""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        检索遥操作器事件并更新 info 字典。

        参数:
            info: 传入的 info 字典。

        返回:
            包含遥操作器事件的新字典。
        """
        new_info = dict(info)

        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    裁剪和/或调整图像观测的大小。

    此步骤遍历观测字典中的所有图像键并应用指定的变换。它处理设备放置，
    如果加速器(如 MPS)不支持某些操作，则将张量移动到 CPU。

    属性:
        crop_params_dict: 将图像键映射到裁剪参数(顶部、左侧、高度、宽度)的字典。
        resize_size: 用于将所有图像调整为的元组(高度、宽度)。
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        对观测字典中的所有图像应用裁剪和调整大小。

        参数:
            observation: 观测字典，可能包含图像张量。

        返回:
            包含已变换图像的新观测字典。
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # 处理观测中的所有图像键
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # 注意 (maractingi): 裁剪和调整大小没有 mps 内核，因此需要移动到 cpu
            if device.type == "mps":
                image = image.cpu()
            # 如果为此键提供了裁剪参数，则进行裁剪
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        返回步骤的配置以进行序列化。

        返回:
            包含裁剪参数和调整大小尺寸的字典。
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        如果应用了调整大小，则更新策略特征字典中的图像特征形状。

        参数:
            features: 策略特征字典。

        返回:
            具有新图像形状的更新策略特征字典。
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    跟踪剧集步数并通过截断来强制执行时间限制。

    属性:
        max_episode_steps: 每个剧集允许的最大步数。
        current_step: 当前活动剧集的当前步数。
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        递增步数计数器，如果达到时间限制则设置截断标志。

        参数:
            truncated: 传入的截断标志。

        返回:
            如果达到剧集步数限制则为 True，否则为传入值。
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): 是否缺少 else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        返回步骤的配置以进行序列化。

        返回:
            包含 `max_episode_steps` 的字典。
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """重置步数计数器，通常在新剧集开始时调用。"""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ComplementaryDataProcessorStep):
    """
    对低效的夹爪使用施加惩罚。

    此步骤基于位置阈值，对试图关闭已关闭的夹爪或打开已打开的夹爪的动作进行惩罚。

    属性:
        penalty: 要应用的负奖励值。
        max_gripper_pos: 夹爪的最大位置值，用于归一化。
    """

    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        计算夹爪惩罚并将其添加到互补数据中。

        参数:
            complementary_data: 传入的互补数据，应包含原始关节位置。

        返回:
            添加了 `discrete_penalty` 键的新互补数据字典。
        """
        action = self.transition.get(TransitionKey.ACTION)

        raw_joint_positions = complementary_data.get("raw_joint_positions", None)
        if raw_joint_positions is None:
            return complementary_data

        current_gripper_pos = raw_joint_positions.get(GRIPPER_KEY, None)
        if current_gripper_pos is None:
            return complementary_data

        # 在此阶段，夹爪动作是 PolicyAction
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # 归一化夹爪状态和动作
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # 与原始版本一样计算惩罚布尔值
        gripper_penalty_bool = (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # 创建包含惩罚信息的新互补数据
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty

        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        """
        返回步骤的配置以进行序列化。

        返回:
            包含惩罚值和最大夹爪位置的字典。
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        """重置处理器的内部状态。"""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    处理人工干预，覆盖策略动作并管理剧集终止。

    当检测到干预(通过 `info` 字典中的遥操作器事件)时，此步骤将策略的动作
    替换为人类的遥操作动作。它还处理终止剧集或标记成功的信号。

    属性:
        use_gripper: 是否在遥操作动作中包含夹爪。
        terminate_on_success: 如果为 True，则在收到 `success` 事件时自动设置 `done` 标志。
    """

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        处理转换以处理干预。

        参数:
            transition: 传入的环境转换。

        返回:
            修改后的转换，可能包含覆盖的动作、更新的奖励和终止状态。
        """
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # 从互补数据中获取干预信号
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # 如果干预处于活动状态，则覆盖动作
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # 将 teleop_action 字典转换为张量格式
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # 处理剧集终止
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # 使用干预元数据更新 info
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # 使用遥控动作更新互补数据
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回步骤的配置以进行序列化。

        返回:
            包含步骤配置属性的字典。
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    将预训练的奖励分类器应用于图像观测以预测成功。

    此步骤使用模型来确定当前状态是否成功，更新奖励并可能终止剧集。

    属性:
        pretrained_path: 预训练的奖励分类器模型的路径。
        device: 运行分类器的设备。
        success_threshold: 将预测视为成功的概率阈值。
        success_reward: 成功时分配的奖励值。
        terminate_on_success: 如果为 True，则在成功分类后终止剧集。
        reward_classifier: 加载的分类器模型实例。
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """在创建数据类后初始化奖励分类器模型。"""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        处理转换，将奖励分类器应用于其图像观测。

        参数:
            transition: 传入的环境转换。

        返回:
            根据分类器的预测更新了奖励和完成标志的修改后的转换。
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # 从观测中提取图像
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # 运行奖励分类器
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # 计算奖励和终止
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # 更新转换
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # 使用分类器频率更新 info
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回步骤的配置以进行序列化。

        返回:
            包含步骤配置属性的字典。
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
