#!/usr/bin/env python

# 版权所有 2024 The HuggingFace Inc. team。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 有关许可证下权限和限制的具体语言，请参阅许可证。

from dataclasses import dataclass, field

from lerobot import (
    policies,  # noqa: F401
)
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # 你可以在此处提供数据集列表。`train.py` 会创建所有数据集并将它们连接起来。
    # 注意：仅保留数据集之间共同的数据键。每个数据集都会获得一个额外的转换，
    # 将 "dataset_index" 插入到返回的项中。索引映射根据提供数据集的顺序进行。
    repo_id: str
    # 存储数据集的根目录（例如 'dataset/path'）。
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False


@dataclass
class WandBConfig:
    enable: bool = False
    # 设置为 true 可禁止保存工件，即使 training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # 允许的值：'online', 'offline' 'disabled'。默认为 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` 指定在 gym.vector.VectorEnv 中使用的环境数量。
    batch_size: int = 50
    # `use_async_envs` 指定是否使用异步环境（多进程）。
    use_async_envs: bool = False

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                f"评估批次大小大于评估轮数 "
                f"({self.batch_size} > {self.n_episodes})。因此，将实例化 {self.batch_size} 个 "
                f"评估环境，但只会使用 {self.n_episodes} 个。"
                "这可能会显著减慢评估速度。要解决此问题，你应该更新命令，将轮数增加到与批次大小匹配 "
                f"（例如 `eval.n_episodes={self.batch_size}`），或降低批次大小 "
                f"（例如 `eval.batch_size={self.n_episodes}`）。"
            )
