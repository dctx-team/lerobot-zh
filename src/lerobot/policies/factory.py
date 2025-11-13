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

from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    根据注册名称获取策略类。

    该函数使用动态导入以避免一次性将所有策略类加载到内存中，
    从而改善启动时间并减少依赖项。

    参数:
        name: 策略名称。支持的名称有 "tdmpc", "diffusion", "act",
              "vqbet", "pi0", "pi0fast", "sac", "reward_classifier", "smolvla"。

    返回:
        与给定名称对应的策略类。

    异常:
        NotImplementedError: 如果策略名称无法识别。
    """
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    基于策略类型实例化策略配置对象。

    该工厂函数通过将字符串标识符映射到相应的配置类，
    简化了策略配置对象的创建。

    参数:
        policy_type: 策略类型。支持的类型包括 "tdmpc",
                     "diffusion", "act", "vqbet", "pi0", "pi0fast", "sac", "smolvla",
                     "reward_classifier"。
        **kwargs: 传递给配置类构造函数的关键字参数。

    返回:
        一个 `PreTrainedConfig` 子类的实例。

    异常:
        ValueError: 如果 `policy_type` 无法识别。
    """
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    定义处理器配置关键字参数的 TypedDict。

    这为传递给 `make_pre_post_processors` 的可选参数提供了类型提示，
    提高了代码清晰度并支持静态分析。

    属性:
        preprocessor_config_filename: 预处理器配置的文件名。
        postprocessor_config_filename: 后处理器配置的文件名。
        preprocessor_overrides: 预处理器配置的覆盖字典。
        postprocessor_overrides: 后处理器配置的覆盖字典。
        dataset_stats: 用于归一化的数据集统计信息。
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为给定策略创建或加载预处理和后处理管道。

    该函数作为工厂函数。它可以从预训练路径加载现有的处理器管道，
    也可以基于策略配置从头创建新的管道。每种策略类型都有专门的
    处理器工厂函数（例如 `make_tdmpc_pre_post_processors`）。

    参数:
        policy_cfg: 要为其创建处理器的策略配置。
        pretrained_path: 用于加载预训练处理器管道的可选路径。
            如果提供，将从该路径加载管道。
        **kwargs: 处理器配置的关键字参数，如 `ProcessorConfigKwargs` 中定义。

    返回:
        包含输入（预处理器）和输出（后处理器）管道的元组。

    异常:
        NotImplementedError: 如果给定策略配置类型未实现处理器工厂函数。
    """
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # 根据策略类型创建新的处理器
    if isinstance(policy_cfg, TDMPCConfig):
        from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors

        processors = make_tdmpc_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, VQBeTConfig):
        from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors

        processors = make_vqbet_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        processors = make_pi0_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0FASTConfig):
        from lerobot.policies.pi0fast.processor_pi0fast import make_pi0fast_pre_post_processors

        processors = make_pi0fast_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SACConfig):
        from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors

        processors = make_sac_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, RewardClassifierConfig):
        from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor

        processors = make_classifier_processor(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        processors = make_smolvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        raise NotImplementedError(f"Processor for policy type '{policy_cfg.type}' is not implemented.")

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """
    实例化一个策略模型。

    该工厂函数处理创建策略的逻辑，这需要确定输入和输出特征的形状。
    这些形状可以从 `LeRobotDatasetMetadata` 对象或 `EnvConfig` 对象推导。
    该函数可以从头初始化一个新策略，也可以加载一个预训练的策略。

    参数:
        cfg: 要创建的策略配置。如果设置了 `cfg.pretrained_path`，
             则将从该路径加载策略权重。
        ds_meta: 用于推断特征形状和类型的数据集元数据。还提供了
                 归一化层所需的统计信息。
        env_cfg: 用于推断特征形状和类型的环境配置。
                 必须提供 `ds_meta` 或 `env_cfg` 之一。

    返回:
        一个已实例化并放置到设备上的策略模型。

    异常:
        ValueError: 如果同时提供或都不提供 `ds_meta` 和 `env_cfg`。
        NotImplementedError: 如果尝试使用不支持的策略-后端组合
                             (例如 VQBeT 配合 'mps')。
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # 注意：目前，如果你尝试在 mps 后端上运行 vqbet，你将会遇到这个错误。
    # TODO(aliberts, rcadene): 在策略中实现 check_backend_compatibility 检查？
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "你正在从头实例化一个策略，其特征是从环境而非数据集中解析的。"
                "如果没有数据集的统计信息，策略内部的归一化模块默认将具有无限值。"
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # 加载预训练策略，并在需要时覆盖配置（例如，如果有推理时的超参数需要调整）。
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # 创建一个全新的策略。
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
