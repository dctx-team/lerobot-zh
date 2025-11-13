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

"""自定义旋转工具，用于替代 scipy.spatial.transform.Rotation。"""

import numpy as np


class Rotation:
    """
    自定义旋转类，提供 scipy.spatial.transform.Rotation 功能的子集。

    支持旋转向量、旋转矩阵和四元数之间的转换。
    """

    def __init__(self, quat: np.ndarray) -> None:
        """从四元数 [x, y, z, w] 初始化旋转。"""
        self._quat = np.asarray(quat, dtype=float)
        # 归一化四元数
        norm = np.linalg.norm(self._quat)
        if norm > 0:
            self._quat = self._quat / norm

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "Rotation":
        """
        使用 Rodrigues 公式从旋转向量创建旋转。

        参数:
            rotvec: 旋转向量 [x, y, z]，其模长为弧度角度

        返回:
            Rotation 实例
        """
        rotvec = np.asarray(rotvec, dtype=float)
        angle = np.linalg.norm(rotvec)

        if angle < 1e-8:
            # 对于非常小的角度，使用单位四元数
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = rotvec / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            # 四元数 [x, y, z, w]
            quat = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])

        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Rotation":
        """
        从 3x3 旋转矩阵创建旋转。

        参数:
            matrix: 3x3 旋转矩阵

        返回:
            Rotation 实例
        """
        matrix = np.asarray(matrix, dtype=float)

        # Shepherd 方法：将旋转矩阵转换为四元数
        trace = np.trace(matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw])
        return cls(quat)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "Rotation":
        """
        从四元数创建旋转。

        参数:
            quat: 四元数 [x, y, z, w] 或 [w, x, y, z]（在文档字符串中指定约定）
                  此实现期望 [x, y, z, w] 格式

        返回:
            Rotation 实例
        """
        return cls(quat)

    def as_matrix(self) -> np.ndarray:
        """
        将旋转转换为 3x3 旋转矩阵。

        返回:
            3x3 旋转矩阵
        """
        qx, qy, qz, qw = self._quat

        # 从四元数计算旋转矩阵
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    def as_rotvec(self) -> np.ndarray:
        """
        将旋转转换为旋转向量。

        返回:
            旋转向量 [x, y, z]，其模长为弧度角度
        """
        qx, qy, qz, qw = self._quat

        # 确保 qw 为正以获得唯一表示
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        # 计算角度和轴
        angle = 2.0 * np.arccos(np.clip(abs(qw), 0.0, 1.0))
        sin_half_angle = np.sqrt(1.0 - qw * qw)

        if sin_half_angle < 1e-8:
            # 对于非常小的角度，使用线性化: rotvec ≈ 2 * [qx, qy, qz]
            return 2.0 * np.array([qx, qy, qz])

        # 提取轴并按角度缩放
        axis = np.array([qx, qy, qz]) / sin_half_angle
        return angle * axis

    def as_quat(self) -> np.ndarray:
        """
        获取四元数表示。

        返回:
            四元数 [x, y, z, w]
        """
        return self._quat.copy()

    def apply(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        将此旋转应用于一组向量。

        这等同于将旋转矩阵应用于向量:
        self.as_matrix() @ vectors (或 self.as_matrix().T @ vectors 如果 inverse=True)。

        参数:
            vectors: 形状为 (3,) 或 (N, 3) 的数组，表示 3D 空间中的向量
            inverse: 如果为 True，则应用旋转的逆。默认为 False。

        返回:
            旋转后的向量，形状为:
            - (3,) 如果输入是形状为 (3,) 的单个向量
            - (N, 3) 在所有其他情况下
        """
        vectors = np.asarray(vectors, dtype=float)
        original_shape = vectors.shape

        # 处理单个向量的情况 - 确保它是 2D 以便矩阵乘法
        if vectors.ndim == 1:
            if len(vectors) != 3:
                raise ValueError("单个向量必须长度为 3")
            vectors = vectors.reshape(1, 3)
            single_vector = True
        elif vectors.ndim == 2:
            if vectors.shape[1] != 3:
                raise ValueError("向量必须具有形状 (N, 3)")
            single_vector = False
        else:
            raise ValueError("向量必须是 1D 或 2D 数组")

        # 获取旋转矩阵
        rotation_matrix = self.as_matrix()

        # 如果需要，应用逆（对于正交旋转矩阵使用转置）
        if inverse:
            rotation_matrix = rotation_matrix.T

        # 应用旋转: (N, 3) @ (3, 3).T -> (N, 3)
        rotated_vectors = vectors @ rotation_matrix.T

        # 对于单个向量情况返回原始形状
        if single_vector and original_shape == (3,):
            return rotated_vectors.flatten()

        return rotated_vectors

    def inv(self) -> "Rotation":
        """
        反转此旋转。

        旋转与其逆的组合会产生单位变换。

        返回:
            包含此旋转的逆的 Rotation 实例
        """
        qx, qy, qz, qw = self._quat

        # 对于单位四元数，逆是共轭: [-x, -y, -z, w]
        inverse_quat = np.array([-qx, -qy, -qz, qw])

        return Rotation(inverse_quat)

    def __mul__(self, other: "Rotation") -> "Rotation":
        """
        使用 * 运算符将此旋转与另一个旋转组合。

        组合 `r2 * r1` 表示"先应用 r1，然后应用 r2"。
        这等同于应用旋转矩阵: r2.as_matrix() @ r1.as_matrix()

        参数:
            other: 要组合的另一个 Rotation 实例

        返回:
            表示旋转组合的 Rotation 实例
        """
        if not isinstance(other, Rotation):
            return NotImplemented

        # 获取四元数 [x, y, z, w]
        x1, y1, z1, w1 = other._quat  # 先应用
        x2, y2, z2, w2 = self._quat  # 后应用

        # 四元数乘法: q2 * q1 (先应用 q1，然后应用 q2)
        composed_quat = np.array(
            [
                w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,  # x 分量
                w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,  # y 分量
                w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,  # z 分量
                w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,  # w 分量
            ]
        )

        return Rotation(composed_quat)
