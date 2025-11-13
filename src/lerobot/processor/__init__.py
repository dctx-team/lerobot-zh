#!/usr/bin/env python

# 版权所有 2025 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解许可证下的特定语言权限和
# 限制。

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    create_transition,
    transition_to_batch,
)
from .core import (
    EnvAction,
    EnvTransition,
    PolicyAction,
    RobotAction,
    RobotObservation,
    TransitionKey,
)
from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .device_processor import DeviceProcessorStep
from .factory import (
    make_default_processors,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from .gym_action_processor import (
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
)
from .hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
)
from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats
from .observation_processor import VanillaObservationProcessorStep
from .pipeline import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
    DoneProcessorStep,
    IdentityProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    TruncatedProcessorStep,
)
from .policy_robot_bridge import (
    PolicyActionToRobotActionProcessorStep,
    RobotActionToPolicyActionProcessorStep,
)
from .rename_processor import RenameObservationsProcessorStep
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryDataStep",
    "AddTeleopEventsAsInfoStep",
    "ComplementaryDataProcessorStep",
    "batch_to_transition",
    "create_transition",
    "DeviceProcessorStep",
    "DoneProcessorStep",
    "EnvAction",
    "EnvTransition",
    "GripperPenaltyProcessorStep",
    "hotswap_stats",
    "IdentityProcessorStep",
    "ImageCropResizeProcessorStep",
    "InfoProcessorStep",
    "InterventionActionProcessorStep",
    "JointVelocityProcessorStep",
    "make_default_processors",
    "make_default_teleop_action_processor",
    "make_default_robot_action_processor",
    "make_default_robot_observation_processor",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "MotorCurrentProcessorStep",
    "NormalizerProcessorStep",
    "Numpy2TorchActionProcessorStep",
    "ObservationProcessorStep",
    "PolicyAction",
    "PolicyActionProcessorStep",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RobotAction",
    "RobotActionProcessorStep",
    "RobotObservation",
    "RenameObservationsProcessorStep",
    "RewardClassifierProcessorStep",
    "RewardProcessorStep",
    "DataProcessorPipeline",
    "TimeLimitProcessorStep",
    "AddBatchDimensionProcessorStep",
    "RobotProcessorPipeline",
    "TokenizerProcessorStep",
    "Torch2NumpyActionProcessorStep",
    "RobotActionToPolicyActionProcessorStep",
    "PolicyActionToRobotActionProcessorStep",
    "transition_to_batch",
    "TransitionKey",
    "TruncatedProcessorStep",
    "UnnormalizerProcessorStep",
    "VanillaObservationProcessorStep",
]
