# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("viperx")
@dataclass
class ViperXConfig(RobotConfig):
    port: str  # 连接机械臂的端口

    disable_torque_on_disconnect: bool = True

    # /!\ 为了安全，请阅读此说明 /!\
    # `max_relative_target` 出于安全目的限制相对位置目标向量的幅度。
    # 将此设置为正标量以对所有电机使用相同的值，或设置为将电机
    # 名称映射到该电机的 max_relative_target 值的字典。
    # 对于 Aloha，默认情况下，对于每个目标位置请求，电机旋转被限制在 5 度。
    # 当你对遥操作或运行策略更有信心时，可以扩展
    # 此安全限制，甚至通过将其设置为 `null` 来移除它。
    # 此外，所有功能预计都能安全地开箱即用，但我们强烈建议
    # 首先尝试仅遥操作夹爪（通过在此 yaml 中注释掉其余电机），
    # 然后逐渐添加更多电机（通过取消注释），直到你可以完全遥操作两个机械臂
    max_relative_target: float | dict[str, float] = 5.0

    # 相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # 故障排除：如果你的 IntelRealSense 相机之一在
    # 数据录制期间因带宽限制而冻结，你可能需要将相机
    # 插入另一个 USB 集线器或 PCIe 卡。
