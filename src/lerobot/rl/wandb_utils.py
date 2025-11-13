#!/usr/bin/env python

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
import logging
import os
import re
from glob import glob
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """返回用于日志记录的组名。可选择将组名作为列表返回。"""
    lst = [
        f"policy:{cfg.policy.type}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    """从文件系统获取WandB运行ID。

    Args:
        log_dir: 日志目录路径

    Returns:
        WandB运行ID字符串

    Raises:
        RuntimeError: 如果无法找到WandB运行ID
    """
    # 获取WandB运行ID。
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB工件名称中不接受":"或"/"。"""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """使用wandb记录对象的辅助类。"""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # 设置WandB。
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        # 注意：我们将用wandb运行id覆盖cfg.wandb.run_id。
        # 这是因为我们希望能够从wandb运行id恢复运行。
        cfg.wandb.run_id = run_id
        # 处理用于RL异步训练的自定义步骤键。
        self._wandb_custom_step_key: set[str] | None = None
        print(colored("日志将与wandb同步。", "blue", attrs=["bold"]))
        logging.info(f"追踪此运行 --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """将策略检查点记录到wandb。"""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)
        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        """将字典记录到wandb。

        Args:
            d: 要记录的字典
            step: 当前步骤编号
            mode: 记录模式，可以是"train"或"eval"
            custom_step_key: 自定义步骤键，用于异步RL训练

        Raises:
            ValueError: 如果mode不是"train"或"eval"，或者step和custom_step_key都未提供
        """
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # 注意：这并不简单。Wandb步骤必须始终单调递增，并且每次wandb.log调用都会递增，
        # 但在异步RL的情况下，例如，多个时间步是可能的。例如，与环境交互的步骤、
        # 训练步骤、评估步骤等。因此我们需要定义一个自定义步骤键
        # 以记录每个指标的正确步骤。
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            # 不要记录自定义步骤键本身。
            if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d[custom_step_key]
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._wandb.log(data)
                continue

            self._wandb.log(data={f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """将视频记录到wandb。

        Args:
            video_path: 视频文件路径
            step: 当前步骤编号
            mode: 记录模式，可以是"train"或"eval"

        Raises:
            ValueError: 如果mode不是"train"或"eval"
        """
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
