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
import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    """训练流程配置类。"""
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # 将 `output_dir` 设置为保存所有运行输出的位置。如果使用相同的 `output_dir` 值运行另一个训练会话，
    # 其内容将被覆盖，除非将 `resume` 设置为 true。
    output_dir: Path | None = None
    job_name: str | None = None
    # 将 `resume` 设置为 true 以恢复之前的运行。为此，您需要确保 `output_dir` 是现有运行的目录，
    # 并且其中至少包含一个检查点。
    # 注意，恢复运行时，默认行为是使用检查点中的配置，而不管恢复时训练命令中提供了什么。
    resume: bool = False
    # `seed` 用于训练（例如：模型初始化、数据集打乱）以及评估环境。
    seed: int | None = 1000
    # 数据加载器的工作进程数。
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # 检查点在每 `save_freq` 次训练迭代后保存，并在最后一次训练步骤后保存。
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        # HACK: 我们在此处再次解析 cli 参数以获取预训练路径（如果有）。
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # 仅加载策略配置
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # 整个训练配置已经加载，我们只需要获取检查点目录
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"恢复运行时需要 config_path。请指定 {TRAIN_CONFIG_NAME} 的路径"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} 应该是本地路径。"
                    "目前不支持从 hub 恢复。"
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"输出目录 {self.output_dir} 已存在且 resume 为 {self.resume}。"
                f"请更改您的输出目录，以免覆盖 {self.output_dir}。"
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset 目前尚未实现。")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("当不使用策略预设时，必须设置 Optimizer 和 Scheduler。")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "缺少 'policy.repo_id' 参数。请指定它以将模型推送到 hub。"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """使解析器能够使用 `--policy.path=local/dir` 从策略加载配置。"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    """强化学习服务器训练流程配置类。"""
    dataset: DatasetConfig | None = None  # 注意：在强化学习中，我们不需要离线数据集。
