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

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class EvalPipelineConfig:
    """评估管道配置。"""
    # Hub 上托管的模型的仓库 ID，或包含使用 `Policy.save_pretrained` 保存的权重的目录路径。
    # 如果未提供，策略将从头开始初始化（用于调试）。此参数与 `--config` 互斥。
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000

    def __post_init__(self):
        # HACK: 我们在此处再次解析 cli 参数以获取预训练路径（如果有）。
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "未提供预训练路径，将从头开始构建评估策略（随机权重）。"
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """使解析器能够使用 `--policy.path=local/dir` 从策略加载配置。"""
        return ["policy"]
