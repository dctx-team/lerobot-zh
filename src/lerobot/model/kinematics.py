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

import numpy as np


class RobotKinematics:
    """使用 placo 库进行机器人正向和逆向运动学计算"""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] = None,
    ):
        """
        初始化基于 placo 的运动学求解器

        Args:
            urdf_path: 机器人 URDF 文件的路径
            target_frame_name: URDF 中末端执行器框架的名称
            joint_names: 用于运动学求解器的关节名称列表
        """
        try:
            import placo
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # 固定基座

        self.target_frame_name = target_frame_name

        # 设置关节名称
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # 为逆运动学初始化框架任务
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg):
        """
        计算给定关节配置的正向运动学，目标框架名称在构造函数中指定

        Args:
            joint_pos_deg: 关节位置（角度制，numpy 数组）

        Returns:
            末端执行器姿态的 4x4 变换矩阵
        """

        # 将角度转换为弧度
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # 更新 placo 机器人中的关节位置
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # 更新运动学
        self.robot.update_kinematics()

        # 获取变换矩阵
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01
    ):
        """
        使用 placo 求解器计算逆运动学

        Args:
            current_joint_pos: 当前关节位置（角度制，用作初始猜测值）
            desired_ee_pose: 目标末端执行器姿态（4x4 变换矩阵）
            position_weight: 逆运动学中位置约束的权重
            orientation_weight: 逆运动学中方向约束的权重，设为 0.0 则仅约束位置

        Returns:
            达到期望末端执行器姿态的关节位置（角度制）
        """

        # 将当前关节位置转换为弧度作为初始猜测值
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # 将当前关节位置设为初始猜测值
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # 更新框架任务的目标姿态
        self.tip_frame.T_world_frame = desired_ee_pose

        # 根据 position_only 标志配置任务
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # 求解逆运动学
        self.solver.solve(True)
        self.robot.update_kinematics()

        # 提取关节位置
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # 转换回角度制
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # 如果 current_joint_pos 中存在夹爪位置，则保留它
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
