# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非遵守许可证，否则你不得使用此文件。
# 你可以在以下地址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解许可证下的特定语言权限和
# 限制。

import abc
from dataclasses import dataclass, field
from typing import Any

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    """环境配置基类。

    Args:
        task: 任务名称，如果为 None 则使用默认任务。
        fps: 环境帧率（每秒帧数）。
        features: 策略特征字典，将特征名称映射到 PolicyFeature 对象。
        features_map: 环境键到策略键的映射。
        max_parallel_tasks: 最大并行任务数。
        disable_env_checker: 是否禁用环境检查器。
    """

    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        """获取环境配置类型。"""
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        """返回创建 Gym 环境所需的参数字典。"""
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    """Aloha 环境配置。"""
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    """PushT 环境配置。"""
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    """Xarm 机械臂环境配置。"""
    task: str | None = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class ImagePreprocessingConfig:
    """图像预处理配置。

    Args:
        crop_params_dict: 每个相机的裁剪参数字典，格式为 {camera_name: (top, left, height, width)}。
        resize_size: 调整大小后的目标尺寸，格式为 (height, width)。
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class RewardClassifierConfig:
    """奖励分类器配置。

    Args:
        pretrained_path: 预训练模型路径。
        success_threshold: 成功阈值。
        success_reward: 成功奖励值。
    """

    pretrained_path: str | None = None
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class InverseKinematicsConfig:
    """逆运动学处理配置。

    Args:
        urdf_path: URDF 模型文件路径。
        target_frame_name: 目标坐标系名称。
        end_effector_bounds: 末端执行器的边界约束。
        end_effector_step_sizes: 末端执行器的步长大小。
    """

    urdf_path: str | None = None
    target_frame_name: str | None = None
    end_effector_bounds: dict[str, list[float]] | None = None
    end_effector_step_sizes: dict[str, float] | None = None


@dataclass
class ObservationConfig:
    """观察值处理配置。

    Args:
        add_joint_velocity_to_observation: 是否将关节速度添加到观察值中。
        add_current_to_observation: 是否将电流值添加到观察值中。
        add_ee_pose_to_observation: 是否将末端执行器位姿添加到观察值中。
        display_cameras: 是否显示相机图像。
    """

    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    display_cameras: bool = False


@dataclass
class GripperConfig:
    """夹爪控制和惩罚配置。

    Args:
        use_gripper: 是否使用夹爪。
        gripper_penalty: 夹爪惩罚值。
        gripper_penalty_in_reward: 是否将夹爪惩罚计入奖励。
    """

    use_gripper: bool = True
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False


@dataclass
class ResetConfig:
    """环境重置行为配置。

    Args:
        fixed_reset_joint_positions: 固定的重置关节位置。
        reset_time_s: 重置时间（秒）。
        control_time_s: 控制时间（秒）。
        terminate_on_success: 是否在成功时终止。
    """

    fixed_reset_joint_positions: Any | None = None
    reset_time_s: float = 5.0
    control_time_s: float = 20.0
    terminate_on_success: bool = True


@dataclass
class HILSerlProcessorConfig:
    """环境处理流水线配置。

    Args:
        control_mode: 控制模式（例如 "gamepad"）。
        observation: 观察值处理配置。
        image_preprocessing: 图像预处理配置。
        gripper: 夹爪配置。
        reset: 重置配置。
        inverse_kinematics: 逆运动学配置。
        reward_classifier: 奖励分类器配置。
        max_gripper_pos: 最大夹爪位置。
    """

    control_mode: str = "gamepad"
    observation: ObservationConfig | None = None
    image_preprocessing: ImagePreprocessingConfig | None = None
    gripper: GripperConfig | None = None
    reset: ResetConfig | None = None
    inverse_kinematics: InverseKinematicsConfig | None = None
    reward_classifier: RewardClassifierConfig | None = None
    max_gripper_pos: float | None = 100.0


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """HILSerl 机器人环境配置。

    Args:
        robot: 机器人配置。
        teleop: 远程操作配置。
        processor: 处理器配置。
        name: 环境名称。
    """

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: HILSerlProcessorConfig = field(default_factory=HILSerlProcessorConfig)

    name: str = "real_robot"

    @property
    def gym_kwargs(self) -> dict:
        """返回创建 Gym 环境所需的参数字典。"""
        return {}


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    """LIBERO 基准测试环境配置。"""
    task: str = "libero_10"  # 也可以选择 libero_spatial、libero_object 等
    fps: int = 30
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = (None,)
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/agentview_image": f"{OBS_IMAGES}.image",
            "pixels/robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        else:
            raise ValueError(f"不支持的观察类型：{self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }
