from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("pi0fast")
@dataclass
class PI0FASTConfig(PreTrainedConfig):
    # 输入/输出结构。
    n_obs_steps: int = 1
    chunk_size: int = 10
    n_action_steps: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 较短的状态和动作向量将被填充
    max_state_dim: int = 32  # 32
    max_action_dim: int = 32  # 32

    # 图像预处理
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    interpolate_like_pi: bool = False

    # 添加空图像。由 pi0_aloha_sim 使用，它除了顶部相机外还添加了空的
    # 左腕和右腕相机。
    empty_cameras: int = 0

    # 将关节和抓手值从标准 Aloha 空间转换为
    # 用于训练基础模型的 pi 内部运行时使用的空间。
    adapt_to_pi_aloha: bool = False

    # 在传递给模型之前，将关节维度转换为相对于当前状态的增量。
    # 抓手维度将保持绝对值。
    use_delta_joint_actions_aloha: bool = False

    # 分词器
    tokenizer_max_length: int = 48

    # 投影器
    proj_width: int = 1024

    # 解码
    max_decoding_steps: int = 256
    fast_skip_tokens: int = 128  # 跳过 PaliGemma 词汇表中的最后 128 个 token，因为它们是特殊 token
    max_input_seq_len: int = 256  # 512

    # 工具
    use_cache: bool = True

    # 冻结参数
    freeze_vision_encoder: bool = True
    freeze_lm_head: bool = True

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    checkpoint_path: str = None

    padding_side: str = "right"

    precision: str = "bfloat16"
    grad_clip_norm: float = 1

    # 允许在去分词化期间对生成的动作 token 进行填充/截断以确保解码。
    # 在原始版本中，如果形状不匹配，会生成全 0 张量以实现稳定解码。
    relaxed_action_decoding: bool = True

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽）。"""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"分块大小是每次模型调用的动作步数的上限。得到 "
                f"{self.n_action_steps} 作为 `n_action_steps`，{self.chunk_size} 作为 `chunk_size`。"
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"尚未处理多个观测步。得到 `nobs_steps={self.n_obs_steps}`")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
