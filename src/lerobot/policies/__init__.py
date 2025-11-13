# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 有关许可证下特定语言的权限和限制，请参阅许可证。
"""
策略模块

本模块导出各种机器人策略配置类和处理器，包括：
- ACT（Action Chunking with Transformers）
- Diffusion（扩散策略）
- PI0（策略学习）
- SmolVLA（小型视觉-语言-动作模型）
- TDMPC（时序差分模型预测控制）
- VQBeT（矢量量化行为变换器）
"""

from .act.configuration_act import ACTConfig as ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0.processor_pi0 import Pi0NewLineProcessor
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .smolvla.processor_smolvla import SmolVLANewLineProcessor
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "PI0Config",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VQBeTConfig",
]
