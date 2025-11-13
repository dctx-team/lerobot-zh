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
from pathlib import Path

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.utils import load_json, write_json
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """返回与步数对应的检查点子目录。"""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """此函数创建以下目录结构:

    005000/  #  检查点时的训练步数
    ├── pretrained_model/
    │   ├── config.json  # 策略配置
    │   ├── model.safetensors  # 策略权重
    │   ├── train_config.json  # 训练配置
    │   ├── processor.json  # 处理器配置（如果提供了预处理器）
    │   └── step_*.safetensors  # 处理器状态文件（如果有）
    └── training_state/
        ├── optimizer_param_groups.json  #  优化器参数组
        ├── optimizer_state.safetensors  # 优化器状态
        ├── rng_state.safetensors  # 随机数生成器状态
        ├── scheduler_state.json  # 调度器状态
        └── training_step.json  # 训练步数

    参数:
        cfg (TrainPipelineConfig): 本次运行使用的训练配置。
        step (int): 该检查点的训练步数。
        policy (PreTrainedPolicy): 要保存的策略。
        optimizer (Optimizer | None, optional): 要保存状态的优化器。默认为 None。
        scheduler (LRScheduler | None, optional): 要保存状态的调度器。默认为 None。
        preprocessor: 要保存的预处理器/管道。默认为 None。
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    保存训练步数、优化器状态、调度器状态和随机数生成器状态。

    参数:
        save_dir (Path): 保存工件的目录。
        train_step (int): 当前训练步数。
        optimizer (Optimizer | None, optional): 要保存 state_dict 的优化器。
            默认为 None。
        scheduler (LRScheduler | None, optional): 要保存 state_dict 的调度器。
            默认为 None。
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    加载训练步数、优化器状态、调度器状态和随机数生成器状态。
    这用于恢复训练运行。

    参数:
        checkpoint_dir (Path): 检查点目录。应包含 'training_state' 目录。
        optimizer (Optimizer): 要加载 state_dict 的优化器。
        scheduler (LRScheduler | None): 要加载 state_dict 的调度器（可以为 None）。

    引发:
        NotADirectoryError: 如果 'checkpoint_dir' 不包含 'training_state' 目录

    返回:
        tuple[int, Optimizer, LRScheduler | None]: 训练步数、优化器和调度器及其
            已加载的 state_dict。
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
