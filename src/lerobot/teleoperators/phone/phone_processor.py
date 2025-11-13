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

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep
from lerobot.teleoperators.phone.config_phone import PhoneOS


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction(RobotActionProcessorStep):
    """
    将校准后的手机姿态动作映射到标准化的机器人动作输入。

    此处理器步骤充当手机遥操作器输出和机器人期望的动作格式之间的桥梁。
    它将手机的6自由度姿态（位置和旋转）重新映射到机器人的目标末端执行器姿态，
    应用必要的轴反转和交换。它还解释平台特定的按钮按下以生成夹爪命令。

    属性：
        platform: 手机的操作系统（iOS或Android），用于确定夹爪的正确按钮映射。
    """

    # TODO(Steven): Gripper vel could be output of phone_teleop directly
    platform: PhoneOS
    _enabled_prev: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        """
        处理手机动作字典以创建机器人动作字典。

        参数：
            act: 来自手机遥操作器的输入动作字典。

        返回：
            为机器人控制器格式化的新动作字典。

        引发：
            ValueError: 如果输入动作中缺少'pos'或'rot'键。
        """
        # 从动作中弹出它们
        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        rotvec = rot.as_rotvec()  # 绝对方向作为旋转向量

        # 将某些输入映射到某些动作
        if self.platform == PhoneOS.IOS:
            gripper_vel = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = (
                a - b
            )  # 如果按下a则为正，如果按下b则为负，如果都按或都不按则为0

        # 对于某些动作，我们需要反转轴
        action["enabled"] = enabled
        action["target_x"] = -pos[1] if enabled else 0.0
        action["target_y"] = pos[0] if enabled else 0.0
        action["target_z"] = pos[2] if enabled else 0.0
        action["target_wx"] = rotvec[1] if enabled else 0.0
        action["target_wy"] = rotvec[0] if enabled else 0.0
        action["target_wz"] = -rotvec[2] if enabled else 0.0
        action["gripper_vel"] = gripper_vel  # 禁用时仍发送夹爪动作
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"phone.{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
