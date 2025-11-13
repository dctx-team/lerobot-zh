#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
本模块定义了一个通用的顺序数据处理管道框架,主要用于转换机器人数据(观测、动作、奖励等)。

核心组件包括:
- ProcessorStep: 单个数据转换操作的抽象基类。
- ProcessorStepRegistry: 用于按名称注册和检索 ProcessorStep 类的机制。
- DataProcessorPipeline: 将多个 ProcessorStep 实例链接在一起以形成完整数据处理工作流的类。
  它与 Hugging Face Hub 集成,便于管道及其配置和状态的共享和版本控制。
- 专用的 ProcessorStep 抽象子类(例如 ObservationProcessorStep、ActionProcessorStep),
  用于简化针对数据转换特定部分的步骤创建。
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypedDict, TypeVar, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.hub import HubMixin

from .converters import batch_to_transition, create_transition, transition_to_batch
from .core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey

# 管道输入和输出的通用类型变量。
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:
    """ProcessorStep 类的注册表,允许从字符串名称实例化。

    该类提供了将字符串标识符映射到 `ProcessorStep` 类的方法,
    这对于从配置文件反序列化管道非常有用,而无需硬编码类导入。
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """注册 ProcessorStep 的类装饰器。

        参数:
            name: 注册类时使用的名称。如果为 None,则使用类的 `__name__`。

        返回:
            一个装饰器函数,用于注册类并返回它。

        引发:
            ValueError: 如果已注册具有相同名称的步骤。
        """

        def decorator(step_class: type) -> type:
            """执行注册的实际装饰器。"""
            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"处理器步骤 '{registration_name}' 已被注册。"
                    f"请使用不同的名称或先取消注册现有的步骤。"
                )

            cls._registry[registration_name] = step_class
            # 将注册名称存储在类上,以便在序列化期间轻松查找。
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """通过名称从注册表检索处理器步骤类。

        参数:
            name: 要检索的步骤的名称。

        返回:
            与给定名称对应的处理器步骤类。

        引发:
            KeyError: 如果在注册表中找不到该名称。
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"在注册表中未找到处理器步骤 '{name}'。"
                f"可用的步骤: {available}。"
                f"请确保该步骤已使用 @ProcessorStepRegistry.register() 注册"
            )
        return cls._registry[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """从注册表中移除处理器步骤。

        参数:
            name: 要取消注册的步骤的名称。
        """
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:
        """返回所有已注册处理器步骤名称的列表。"""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """清除注册表中的所有处理器步骤。"""
        cls._registry.clear()


class ProcessorStep(ABC):
    """数据处理管道中单个步骤的抽象基类。

    每个步骤必须实现 `__call__` 方法来对数据转换执行其转换,
    以及 `transform_features` 方法来描述它如何改变数据特征的形状或类型。

    子类可以通过实现 `state_dict` 和 `load_state_dict` 来选择性地保持状态。
    """

    _current_transition: EnvTransition | None = None

    @property
    def transition(self) -> EnvTransition:
        """提供对正在处理的最新转换的访问。

        这对于需要访问转换数据的其他部分(除了其主要目标之外)的步骤很有用
        (例如,需要查看观测的动作处理步骤)。

        引发:
            ValueError: 如果在使用转换调用步骤之前访问。
        """
        if self._current_transition is None:
            raise ValueError("转换未设置。请确保先使用转换调用该步骤。")
        return self._current_transition

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """处理环境转换。

        该方法应包含处理步骤的核心逻辑。

        参数:
            transition: 要处理的输入数据转换。

        返回:
            处理后的转换。
        """
        return transition

    def get_config(self) -> dict[str, Any]:
        """返回步骤的配置以进行序列化。

        返回:
            配置参数的 JSON 可序列化字典。
        """
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """返回步骤的状态(例如,学习的参数、运行均值)。

        返回:
            将状态名称映射到张量的字典。
        """
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """从状态字典加载步骤的状态。

        参数:
            state: 状态张量的字典。
        """
        return None

    def reset(self) -> None:
        """重置处理器步骤的内部状态(如果有)。"""
        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """定义此步骤如何修改管道特征的描述。

        该方法用于跟踪数据形状、数据类型或模态的变化,
        因为数据流经管道,而无需处理实际数据。

        参数:
            features: 描述观测、动作等输入特征的字典。

        返回:
            描述此步骤转换后输出特征的字典。
        """
        return features


class ProcessorKwargs(TypedDict, total=False):
    """用于管道构建中可选关键字参数的 TypedDict。"""

    to_transition: Callable[[dict[str, Any]], EnvTransition] | None
    to_output: Callable[[EnvTransition], Any] | None
    name: str | None
    before_step_hooks: list[Callable[[int, EnvTransition], None]] | None
    after_step_hooks: list[Callable[[int, EnvTransition], None]] | None


class ProcessorMigrationError(Exception):
    """当模型需要迁移到处理器格式时引发"""

    def __init__(self, model_path: str | Path, migration_command: str, original_error: str):
        self.model_path = model_path
        self.migration_command = migration_command
        self.original_error = original_error
        super().__init__(
            f"模型 '{model_path}' 需要迁移到处理器格式。"
            f"运行: {migration_command}\n\n原始错误: {original_error}"
        )


@dataclass
class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):
    """用于处理数据的顺序管道,与 Hugging Face Hub 集成。

    该类将多个 `ProcessorStep` 实例链接在一起以形成完整的数据处理工作流。
    它是通用的,允许自定义输入和输出类型,这些类型由 `to_transition` 和 `to_output` 转换器处理。

    属性:
        steps: 构成管道的 `ProcessorStep` 对象序列。
        name: 管道的描述性名称。
        to_transition: 将原始输入数据转换为标准化 `EnvTransition` 格式的函数。
        to_output: 将最终 `EnvTransition` 转换为所需输出格式的函数。
        before_step_hooks: 在执行每个步骤之前要调用的函数列表。
        after_step_hooks: 在执行每个步骤之后要调用的函数列表。
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "DataProcessorPipeline"

    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: TInput) -> TOutput:
        """通过完整管道处理输入数据。

        参数:
            data: 要处理的输入数据。

        返回:
            指定输出格式的处理后数据。
        """
        transition = self.to_transition(data)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

    def _forward(self, transition: EnvTransition) -> EnvTransition:
        """按顺序执行所有处理步骤和钩子。

        参数:
            transition: 初始 `EnvTransition` 对象。

        返回:
            应用所有步骤后的最终 `EnvTransition`。
        """
        for idx, processor_step in enumerate(self.steps):
            # 执行前置钩子
            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)

            # 执行后置钩子
            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    def step_through(self, data: TInput) -> Iterable[EnvTransition]:
        """逐步处理数据,在每个阶段生成转换。

        这是一个生成器方法,对于调试和检查数据在通过管道时的中间状态很有用。

        参数:
            data: 输入数据。

        生成:
            `EnvTransition` 对象,从初始状态开始,然后在每个处理步骤之后。
        """
        transition = self.to_transition(data)

        # 在任何处理之前生成初始状态。
        yield transition

        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """符合 `HubMixin` 保存机制的内部方法。

        该方法执行实际的保存工作,由 HubMixin.save_pretrained 调用。
        """
        config_filename = kwargs.pop("config_filename", None)

        # 清理管道名称以创建有效的文件名前缀。
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        config: dict[str, Any] = {
            "name": self.name,
            "steps": [],
        }

        # 遍历每个步骤以构建其配置条目。
        for step_index, processor_step in enumerate(self.steps):
            registry_name = getattr(processor_step.__class__, "_registry_name", None)

            step_entry: dict[str, Any] = {}
            # 优先使用注册表名称以提高可移植性,否则回退到完整类路径。
            if registry_name:
                step_entry["registry_name"] = registry_name
            else:
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )

            # 如果实现了 `get_config`,则保存步骤配置。
            if hasattr(processor_step, "get_config"):
                step_entry["config"] = processor_step.get_config()

            # 如果实现了 `state_dict` 并返回非空字典,则保存步骤状态。
            if hasattr(processor_step, "state_dict"):
                state = processor_step.state_dict()
                if state:
                    # 克隆张量以避免修改原始状态。
                    cloned_state = {key: tensor.clone() for key, tensor in state.items()}

                    # 为状态文件创建唯一的文件名。
                    if registry_name:
                        state_filename = f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"
                    else:
                        state_filename = f"{sanitized_name}_step_{step_index}.safetensors"

                    save_file(cloned_state, os.path.join(str(save_directory), state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

        # 写入主配置 JSON 文件。
        with open(os.path.join(str(save_directory), config_filename), "w") as file_pointer:
            json.dump(config, file_pointer, indent=2)

    def save_pretrained(
        self,
        save_directory: str | Path | None = None,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        config_filename: str | None = None,
        **push_to_hub_kwargs,
    ):
        """将管道的配置和状态保存到目录。

        该方法创建一个 JSON 配置文件,定义管道的结构(名称和步骤)。
        对于每个有状态的步骤,它还会保存一个包含其状态字典的 `.safetensors` 文件。

        参数:
            save_directory: 保存管道的目录。如果为 None,则保存到
                HF_LEROBOT_HOME/processors/{sanitized_pipeline_name}。
            repo_id: Hub 上仓库的 ID。仅在 `push_to_hub=True` 时使用。
            push_to_hub: 保存后是否将对象推送到 Hugging Face Hub。
            card_kwargs: 传递给卡片模板以自定义卡片的其他参数。
            config_filename: JSON 配置文件的名称。如果为 None,则从管道的 `name` 属性生成名称。
            **push_to_hub_kwargs: 传递给 push_to_hub 方法的其他关键字参数。
        """
        if save_directory is None:
            # 使用 HF_LEROBOT_HOME 中的默认目录
            from lerobot.utils.constants import HF_LEROBOT_HOME

            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
            save_directory = HF_LEROBOT_HOME / "processors" / sanitized_name

        # 对于直接保存(不通过 hub),处理 config_filename
        if not push_to_hub and config_filename is not None:
            # 直接使用 config_filename 调用 _save_pretrained
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self._save_pretrained(save_directory, config_filename=config_filename)
            return None

        # 通过 kwargs 为 _save_pretrained 传递 config_filename(使用 hub 时)
        if config_filename is not None:
            push_to_hub_kwargs["config_filename"] = config_filename

        # 调用父类的 save_pretrained,它将调用我们的 _save_pretrained
        return super().save_pretrained(
            save_directory=save_directory,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            card_kwargs=card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_filename: str,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
        **kwargs,
    ) -> DataProcessorPipeline[TInput, TOutput]:
        """从本地目录、单个文件或 Hugging Face Hub 仓库加载管道。

        该方法实现了具有智能迁移检测的简化加载管道:

        **简化的加载策略**:
        1. **配置加载** (_load_config):
           - **目录**: 从目录加载指定的 config_filename
           - **单个文件**: 直接加载文件(忽略 config_filename)
           - **Hub 仓库**: 从 Hub 下载指定的 config_filename

        2. **配置验证** (_validate_loaded_config):
           - 格式验证: 确保配置是有效的处理器格式
           - 迁移检测: 指导用户迁移旧的 LeRobot 模型
           - 清晰的错误: 提供可操作的错误消息

        3. **步骤构建** (_build_steps_with_overrides):
           - 类解析: 注册表查找或动态导入
           - 覆盖合并: 用户参数覆盖保存的配置
           - 状态加载: 为有状态的步骤加载 .safetensors 文件

        4. **覆盖验证** (_validate_overrides_used):
           - 确保所有用户覆盖都已应用(捕获拼写错误)
           - 提供包含可用键的有用错误消息

        **迁移检测**:
        - **智能检测**: 分析 JSON 文件以检测旧的 LeRobot 模型
        - **精确定位**: 避免对其他 HuggingFace 模型产生误报
        - **清晰指导**: 提供要运行的确切迁移命令
        - **错误模式**: 始终引发 ProcessorMigrationError 以明确用户操作

        **加载示例**:
        ```python
        # 目录加载
        pipeline = DataProcessorPipeline.from_pretrained("/models/my_model", config_filename="processor.json")

        # 单文件加载
        pipeline = DataProcessorPipeline.from_pretrained(
            "/models/my_model/processor.json", config_filename="processor.json"
        )

        # Hub 加载
        pipeline = DataProcessorPipeline.from_pretrained("user/repo", config_filename="processor.json")

        # 多个配置(预处理器/后处理器)
        preprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_preprocessor.json"
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_postprocessor.json"
        )
        ```

        **覆盖系统**:
        - **键匹配**: 使用注册表名称或类名作为覆盖键
        - **配置合并**: 用户覆盖优先于保存的配置
        - **验证**: 确保所有覆盖键匹配实际步骤(捕获拼写错误)
        - **示例**: overrides={"NormalizeStep": {"device": "cuda"}}

        参数:
            pretrained_model_name_or_path: Hugging Face Hub 上仓库的标识符、
                本地目录的路径或单个配置文件的路径。
            config_filename: 管道的 JSON 配置文件的名称。始终必需,
                以防止存在多个配置时出现歧义(例如,预处理器 vs 后处理器)。
            force_download: 是否强制(重新)下载文件。
            resume_download: 是否恢复先前中断的下载。
            proxies: 要使用的代理服务器字典。
            token: 用作私有 Hub 仓库的 HTTP bearer 授权的令牌。
            cache_dir: 用于存储下载文件的特定缓存文件夹的路径。
            local_files_only: 如果为 True,避免从 Hub 下载文件。
            revision: 要使用的特定模型版本(例如,分支名称、标签名称或提交 id)。
            overrides: 用于覆盖特定步骤配置的字典。键应匹配步骤的类名或注册表名称。
            to_transition: 将输入数据转换为 `EnvTransition` 的自定义函数。
            to_output: 将最终 `EnvTransition` 转换为输出格式的自定义函数。
            **kwargs: 其他参数(未使用)。

        返回:
            使用指定配置和状态加载的 `DataProcessorPipeline` 实例。

        引发:
            FileNotFoundError: 如果找不到配置文件。
            ValueError: 如果配置不明确或实例化失败。
            ImportError: 如果无法导入步骤的类。
            KeyError: 如果覆盖键与管道中的任何步骤不匹配。
            ProcessorMigrationError: 如果模型需要迁移到处理器格式。
        """
        model_id = str(pretrained_model_name_or_path)
        hub_download_kwargs = {
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        # 1. 使用简化的 3 路逻辑加载配置
        loaded_config, base_path = cls._load_config(model_id, config_filename, hub_download_kwargs)

        # 2. 验证配置并处理迁移
        cls._validate_loaded_config(model_id, loaded_config, config_filename)

        # 3. 使用覆盖构建步骤
        steps, validated_overrides = cls._build_steps_with_overrides(
            loaded_config, overrides or {}, model_id, base_path, hub_download_kwargs
        )

        # 4. 验证所有覆盖都已使用
        cls._validate_overrides_used(validated_overrides, loaded_config)

        # 5. 构建并返回最终管道实例
        return cls(
            steps=steps,
            name=loaded_config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or cast(Callable[[TInput], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )

    @classmethod
    def _load_config(
        cls,
        model_id: str,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        """从本地文件或 Hugging Face Hub 加载配置。

        此方法实现了超级简化的 3 路加载策略:

        1. **本地目录**: 从目录加载 config_filename
           - 示例: model_id="/models/my_model", config_filename="processor.json"
           - 加载: "/models/my_model/processor.json"

        2. **单个文件**: 直接加载文件(忽略 config_filename)
           - 示例: model_id="/models/my_model/processor.json"
           - 加载: "/models/my_model/processor.json"(忽略 config_filename)

        3. **Hub 仓库**: 从 Hub 下载 config_filename
           - 示例: model_id="user/repo", config_filename="processor.json"
           - 从 Hub 仓库下载并加载: config_filename

        **显式 config_filename 的优势**:
        - 没有自动检测复杂性或边缘情况
        - 没有加载错误配置的风险(预处理器 vs 后处理器)
        - 本地和 Hub 使用之间的一致行为
        - 清晰、可预测的错误

        参数:
            model_id: 模型标识符(Hub 仓库 ID、本地目录或文件路径)
            config_filename: 要加载的显式配置文件名(始终必需)
            hub_download_kwargs: hf_hub_download 的参数(令牌、缓存等)

        返回:
            (loaded_config, base_path) 的元组
            - loaded_config: 解析的 JSON 配置字典(始终已加载,永不为 None)
            - base_path: 包含配置文件的目录(用于状态文件解析)

        引发:
            FileNotFoundError: 如果无法在本地或 Hub 上找到配置文件
        """
        model_path = Path(model_id)

        if model_path.is_dir():
            # 目录: 从目录加载指定的配置
            config_path = model_path / config_filename
            if not config_path.exists():
                # 在给出清晰错误之前检查迁移
                if cls._should_suggest_migration(model_path):
                    cls._suggest_processor_migration(model_id, f"未找到配置文件 '{config_filename}'")
                raise FileNotFoundError(
                    f"在目录 '{model_id}' 中未找到配置文件 '{config_filename}'"
                )

            with open(config_path) as f:
                return json.load(f), model_path

        elif model_path.is_file():
            # 文件: 直接加载文件(对于单个文件忽略 config_filename)
            with open(model_path) as f:
                return json.load(f), model_path.parent

        else:
            # Hub: 下载指定的配置
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=config_filename,
                    repo_type="model",
                    **hub_download_kwargs,
                )

                with open(config_path) as f:
                    return json.load(f), Path(config_path).parent

            except Exception as e:
                raise FileNotFoundError(
                    f"在 HuggingFace Hub 上的 '{model_id}' 中找不到 '{config_filename}'"
                ) from e

    @classmethod
    def _validate_loaded_config(
        cls, model_id: str, loaded_config: dict[str, Any], config_filename: str
    ) -> None:
        """验证配置已加载且是有效的处理器配置。

        此方法通过智能迁移检测验证处理器配置格式:

        **配置格式验证**:
        - 使用 _is_processor_config() 验证结构
          - 必须有包含步骤配置列表的 "steps" 字段
          - 每个步骤需要 "class" 或 "registry_name"
        - 如果验证失败且是本地目录: 检查是否需要迁移
        - 如果需要迁移: 引发包含命令的 ProcessorMigrationError
        - 如果不需要迁移: 引发包含有用信息的 ValueError

        **迁移检测逻辑**:
        - 仅为本地目录触发(不针对 Hub 仓库)
        - 分析目录中的所有 JSON 文件以检测旧的 LeRobot 模型
        - 提供包含模型路径的精确迁移命令

        参数:
            model_id: 模型标识符(用于迁移检测)
            loaded_config: 加载的配置字典(保证非 None)
            config_filename: 已加载的配置文件名(用于错误消息)

        引发:
            ValueError: 如果配置格式无效
            ProcessorMigrationError: 如果模型需要迁移到处理器格式
        """
        # 验证这是否确实是处理器配置
        if not cls._is_processor_config(loaded_config):
            if Path(model_id).is_dir() and cls._should_suggest_migration(Path(model_id)):
                cls._suggest_processor_migration(
                    model_id,
                    f"配置文件 '{config_filename}' 不是有效的处理器配置",
                )
            raise ValueError(
                f"配置文件 '{config_filename}' 不是有效的处理器配置。"
                f"期望包含 'steps' 字段的配置,但得到: {list(loaded_config.keys())}"
            )

    @classmethod
    def _build_steps_with_overrides(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[list[ProcessorStep], set[str]]:
        """使用覆盖和状态加载构建所有处理器步骤。

        此方法编排完整的步骤构建管道:

        **对于 loaded_config["steps"] 中的每个步骤**:

        1. **类解析**(通过 _resolve_step_class):
           - **如果 "registry_name" 存在**: 在 ProcessorStepRegistry 中查找
             示例: {"registry_name": "normalize_step"} -> 获取已注册的类
           - **否则使用 "class" 字段**: 从完整模块路径动态导入
             示例: {"class": "lerobot.processor.normalize.NormalizeStep"}
           - **结果**: (step_class, step_key),其中 step_key 用于覆盖

        2. **步骤实例化**(通过 _instantiate_step):
           - **合并配置**: saved_config + user_overrides
           - **覆盖优先级**: 用户覆盖优先于保存的配置
           - **示例**: saved={"mean": 0.0}, override={"mean": 1.0} -> final={"mean": 1.0}
           - **结果**: 实例化的 ProcessorStep 对象

        3. **状态加载**(通过 _load_step_state):
           - **如果步骤有 "state_file"**: 从 .safetensors 加载张量状态
           - **本地优先**: 检查 base_path/state_file.safetensors
           - **Hub 回退**: 如果本地未找到则下载状态文件
           - **可选**: 仅在步骤有 load_state_dict 方法时加载

        4. **覆盖跟踪**:
           - **跟踪已使用的覆盖**: 从剩余集合中删除 step_key
           - **目的**: 验证所有用户覆盖都已应用(检测拼写错误)

        **错误处理**:
        - 类解析错误 -> 带有有用消息的 ImportError
        - 实例化错误 -> 带有配置详情的 ValueError
        - 状态加载错误 -> 从 load_state_dict 传播

        参数:
            loaded_config: 加载的处理器配置(必须有 "steps" 字段)
            overrides: 用户提供的参数覆盖(按类名/注册表名称键控)
            model_id: 模型标识符(Hub 状态文件下载需要)
            base_path: 查找状态文件的本地目录路径
            hub_download_kwargs: hf_hub_download 的参数(令牌、缓存等)

        返回:
            (instantiated_steps_list, unused_override_keys) 的元组
            - instantiated_steps_list: 即用的 ProcessorStep 实例列表
            - unused_override_keys: 不匹配任何步骤的覆盖键(用于验证)

        引发:
            ImportError: 如果无法导入步骤类或在注册表中找不到
            ValueError: 如果无法使用其配置实例化步骤
        """
        steps: list[ProcessorStep] = []
        override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:
            # 1. 获取步骤类和键
            step_class, step_key = cls._resolve_step_class(step_entry)

            # 2. 使用覆盖实例化步骤
            step_instance = cls._instantiate_step(step_entry, step_class, step_key, overrides)

            # 3. 如果可用,加载步骤状态
            cls._load_step_state(step_instance, step_entry, model_id, base_path, hub_download_kwargs)

            # 4. 跟踪已使用的覆盖
            if step_key in override_keys:
                override_keys.discard(step_key)

            steps.append(step_instance)

        return steps, override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry: dict[str, Any]) -> tuple[type[ProcessorStep], str]:
        """从注册表或导入路径解析步骤类。

        此方法实现两层解析策略:

        **第 1 层: 基于注册表的解析**(首选):
        - **如果 step_entry 中有 "registry_name"**: 在 ProcessorStepRegistry 中查找
          - **优势**: 更快,无需导入,保证兼容性
          - **示例**: {"registry_name": "normalize_step"} -> 获取预注册的类
          - **错误**: 如果未找到 registry_name 则为 KeyError -> 转换为 ImportError

        **第 2 层: 动态导入回退**:
        - **否则使用 "class" 字段**: 完整的 module.ClassName 导入路径
          - **处理**: 将 "module.path.ClassName" 拆分为模块 + 类部分
          - **导入**: 使用 importlib.import_module() + getattr()
          - **示例**: "lerobot.processor.normalize.NormalizeStep"
            a. 导入模块: "lerobot.processor.normalize"
            b. 获取类: getattr(module, "NormalizeStep")
          - **step_key**: 使用 class_name("NormalizeStep")进行覆盖

        **覆盖键策略**:
        - 注册表步骤: 使用 registry_name("normalize_step")
        - 导入步骤: 使用 class_name("NormalizeStep")
        - 这允许用户使用以下方式覆盖: {"normalize_step": {...}} 或 {"NormalizeStep": {...}}

        **错误处理**:
        - 注册表 KeyError -> 带有注册表上下文的 ImportError
        - 导入/属性错误 -> 带有有用建议的 ImportError
        - 所有错误都包含故障排除指导

        参数:
            step_entry: 步骤配置字典(必须有 "registry_name" 或 "class")

        返回:
            (step_class, step_key) 的元组
            - step_class: 解析的 ProcessorStep 类(准备实例化)
            - step_key: 用于用户覆盖的键(registry_name 或 class_name)

        引发:
            ImportError: 如果无法从注册表或导入路径加载步骤类
        """
        if "registry_name" in step_entry:
            try:
                step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                return step_class, step_entry["registry_name"]
            except KeyError as e:
                raise ImportError(f"从注册表加载处理器步骤失败。{str(e)}") from e
        else:
            # 使用完整类路径的动态导入回退
            full_class_path = step_entry["class"]
            module_path, class_name = full_class_path.rsplit(".", 1)

            try:
                module = importlib.import_module(module_path)
                step_class = getattr(module, class_name)
                return step_class, class_name
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"加载处理器步骤 '{full_class_path}' 失败。"
                    f"请确保模块 '{module_path}' 已安装且包含类 '{class_name}'。"
                    f"考虑使用 @ProcessorStepRegistry.register() 注册步骤以获得更好的可移植性。"
                    f"错误: {str(e)}"
                ) from e

    @classmethod
    def _instantiate_step(
        cls,
        step_entry: dict[str, Any],
        step_class: type[ProcessorStep],
        step_key: str,
        overrides: dict[str, Any],
    ) -> ProcessorStep:
        """使用配置覆盖实例化单个处理器步骤。

        此方法处理配置合并和实例化逻辑:

        **配置合并策略**:
        1. **提取保存的配置**: 从保存的管道中获取 step_entry.get("config", {})
           - 示例: {"config": {"mean": 0.0, "std": 1.0}}
        2. **提取用户覆盖**: 为此步骤获取 overrides.get(step_key, {})
           - 示例: overrides = {"NormalizeStep": {"mean": 2.0, "device": "cuda"}}
        3. **优先级合并**: {**saved_cfg, **step_overrides}
           - **覆盖优先级**: 用户值覆盖保存的值
           - **结果**: {"mean": 2.0, "std": 1.0, "device": "cuda"}

        **实例化过程**:
        - **调用构造函数**: step_class(**merged_cfg)
        - **示例**: NormalizeStep(mean=2.0, std=1.0, device="cuda")

        **错误处理**:
        - **实例化期间的任何异常**: 转换为 ValueError
        - **包含上下文**: 步骤名称、尝试的配置、原始错误
        - **目的**: 帮助用户调试配置问题
        - **常见原因**:
          a. 无效的参数类型(str 而不是 float)
          b. 缺少必需的参数
          c. 不兼容的参数组合

        参数:
            step_entry: 保存配置中的步骤配置(包含 "config" 字典)
            step_class: 要实例化的步骤类(已解析)
            step_key: 用于覆盖的键("registry_name" 或类名)
            overrides: 用户提供的参数覆盖(按 step_key 键控)

        返回:
            实例化的处理器步骤(准备使用)

        引发:
            ValueError: 如果无法实例化步骤,带有详细的错误上下文
        """
        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            raise ValueError(
                f"使用配置实例化处理器步骤 '{step_name}' 失败: {step_entry.get('config', {})}。"
                f"错误: {str(e)}"
            ) from e

    @classmethod
    def _load_step_state(
        cls,
        step_instance: ProcessorStep,
        step_entry: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> None:
        """如果可用,为处理器步骤加载状态字典。

        此方法实现条件状态加载与本地/Hub 回退:

        **前提条件检查**(如果不满足则提前返回):
        1. **step_entry 中有 "state_file"**: 步骤配置指定了状态文件
           - **如果缺失**: 步骤没有保存的状态(例如,无状态转换)
        2. **hasattr(step_instance, "load_state_dict")**: 步骤支持状态加载
           - **如果缺失**: 步骤未实现状态加载(罕见)

        **状态文件解析策略**:
        1. **本地文件优先**: 检查 base_path/state_filename 是否存在
           - **优势**: 更快,无需网络调用
           - **示例**: "/models/my_model/normalize_step_0.safetensors"
           - **用例**: 从本地保存的模型目录加载

        2. **Hub 下载回退**: 从仓库下载状态文件
           - **何时触发**: 找不到本地文件或 base_path 为 None
           - **处理**: 使用与配置相同参数的 hf_hub_download
           - **示例**: 从 "user/repo" 下载 "normalize_step_0.safetensors"
           - **结果**: 下载到本地缓存,返回路径

        **状态加载过程**:
        - **加载张量**: 使用 safetensors.torch.load_file()
        - **应用到步骤**: 调用 step_instance.load_state_dict(tensor_dict)
        - **就地修改**: 更新步骤的内部张量状态

        **常见状态文件示例**:
        - "normalize_step_0.safetensors" - 归一化统计信息
        - "custom_step_1.safetensors" - 学习的参数
        - "tokenizer_step_2.safetensors" - 词汇表嵌入

        参数:
            step_instance: 要加载状态的步骤实例(必须有 load_state_dict)
            step_entry: 步骤配置字典(可能包含 "state_file")
            model_id: 模型标识符(如果需要,用于 Hub 下载)
            base_path: 查找状态文件的本地目录路径(Hub-only 为 None)
            hub_download_kwargs: hf_hub_download 的参数(令牌、缓存等)

        注意:
            此方法就地修改 step_instance 并返回 None。
            如果状态加载失败,异常从 load_state_dict 传播。
        """
        if "state_file" not in step_entry or not hasattr(step_instance, "load_state_dict"):
            return

        state_filename = step_entry["state_file"]

        # 首先尝试本地文件
        if base_path and (base_path / state_filename).exists():
            state_path = str(base_path / state_filename)
        else:
            # 从 Hub 下载
            state_path = hf_hub_download(
                repo_id=model_id,
                filename=state_filename,
                repo_type="model",
                **hub_download_kwargs,
            )

        step_instance.load_state_dict(load_file(state_path))

    @classmethod
    def _validate_overrides_used(
        cls, remaining_override_keys: set[str], loaded_config: dict[str, Any]
    ) -> None:
        """验证所有提供的覆盖都已使用。

        此方法确保用户覆盖有效以捕获拼写错误和配置错误:

        **验证逻辑**:
        1. **如果 remaining_override_keys 为空**: 所有覆盖都已使用 -> 成功
           - **提前返回**: 不需要验证
           - **正常情况**: 用户提供了正确的覆盖键

        2. **如果 remaining_override_keys 有条目**: 某些覆盖未使用 -> 错误
           - **根本原因**: 用户提供的键不匹配任何步骤
           - **常见问题**:
             a. 步骤名称中的拼写错误("NormalizStep" vs "NormalizeStep")
             b. 使用错误的键类型(类名 vs 注册表名称)
             c. 步骤在保存的管道中不存在

        **有用的错误生成**:
        - **提取可用键**: 从配置构建有效覆盖键列表
          a. **注册表步骤**: 直接使用 "registry_name"
          b. **导入步骤**: 从 "class" 字段提取类名
          - 示例: "lerobot.processor.normalize.NormalizeStep" -> "NormalizeStep"
        - **错误消息包括**:
          a. 用户提供的无效键
          b. 可使用的有效键列表
          c. 关于注册表 vs 类名的指导

        **覆盖键解析规则**:
        - 带有 "registry_name" 的步骤: 使用 registry_name 进行覆盖
        - 带有 "class" 的步骤: 使用最终类名进行覆盖
        - 用户必须在覆盖字典中匹配这些确切的键

        参数:
            remaining_override_keys: 未匹配到任何步骤的覆盖键
            loaded_config: 加载的处理器配置(包含 "steps" 列表)

        引发:
            KeyError: 如果有任何覆盖键未使用,带有有用的错误消息
        """
        if not remaining_override_keys:
            return

        available_keys = [
            step.get("registry_name") or step["class"].rsplit(".", 1)[1] for step in loaded_config["steps"]
        ]

        raise KeyError(
            f"覆盖键 {list(remaining_override_keys)} 不匹配保存配置中的任何步骤。"
            f"可用的步骤键: {available_keys}。"
            f"确保覆盖键与确切的步骤类名或注册表名称匹配。"
        )

    @classmethod
    def _should_suggest_migration(cls, model_path: Path) -> bool:
        """检查目录是否有 JSON 文件但没有处理器配置。

        此方法实现智能迁移检测以避免误报:

        **决策逻辑**:
        1. **未找到 JSON 文件**: 返回 False
           - **原因**: 空目录或仅有非配置文件
           - **示例**: 仅包含 .safetensors、.md 文件的目录
           - **操作**: 不需要迁移

        2. **存在 JSON 文件**: 分析每个文件
           - **目标**: 确定是否有任何文件是有效的处理器配置
           - **过程**:
             a. 尝试解析每个 .json 文件
             b. 跳过有 JSON 解析错误的文件(格式错误)
             c. 检查解析的配置是否通过 _is_processor_config()
           - **如果找到任何有效处理器**: 返回 False(无需迁移)
           - **如果未找到有效处理器**: 返回 True(需要迁移)

        **示例**:
        - **无需迁移**: ["processor.json", "config.json"],其中 processor.json 有效
        - **需要迁移**: ["config.json", "train.json"],两者都是模型配置
        - **无需迁移**: [](空目录)
        - **需要迁移**: ["old_model_config.json"],旧 LeRobot 格式

        **为什么这有效**:
        - **精确检测**: 仅为实际的旧 LeRobot 模型建议迁移
        - **避免误报**: 不会为其他 HuggingFace 模型类型触发
        - **优雅处理**: 忽略格式错误的 JSON 文件

        参数:
            model_path: 要分析的本地目录路径

        返回:
            如果目录有 JSON 配置但没有处理器配置(需要迁移),返回 True
            如果没有 JSON 文件或至少存在一个有效的处理器配置,返回 False
        """
        json_files = list(model_path.glob("*.json"))
        if len(json_files) == 0:
            return False

        # 检查是否有任何 JSON 文件是处理器配置
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config = json.load(f)

                if cls._is_processor_config(config):
                    return False  # 找到至少一个处理器配置,不需要迁移

            except (json.JSONDecodeError, OSError):
                # 跳过无法解析为 JSON 的文件
                continue

        # 有 JSON 文件但没有处理器配置 - 建议迁移
        return True

    @classmethod
    def _is_processor_config(cls, config: dict) -> bool:
        """检查配置是否遵循 DataProcessorPipeline 格式。

        此方法验证处理器配置结构:

        **必需结构验证**:
        1. **"steps" 字段存在**: 必须有顶层 "steps" 键
           - **如果缺失**: 不是处理器配置(例如,模型配置、训练配置)
           - **无效示例**: {"type": "act", "hidden_dim": 256}

        2. **"steps" 字段类型**: 必须是列表,不能是其他类型
           - **如果不是列表**: 格式无效
           - **无效示例**: {"steps": "some_string"} 或 {"steps": {"key": "value"}}

        3. **空步骤验证**: 空列表是有效的
           - **如果 len(steps) == 0**: 立即返回 True
           - **用例**: 空处理器管道(无操作)
           - **有效示例**: {"name": "EmptyProcessor", "steps": []}

        **单独步骤验证**(针对非空步骤):
        对于步骤列表中的每个步骤:
        1. **步骤类型**: 必须是字典
           - **如果不是字典**: 步骤格式无效
           - **无效示例**: ["string_step", 123, true]

        2. **步骤标识符**: 必须有 "class" 或 "registry_name"
           - **"registry_name"**: 已注册的步骤(首选)
             示例: {"registry_name": "normalize_step", "config": {...}}
           - **"class"**: 完整导入路径
             示例: {"class": "lerobot.processor.normalize.NormalizeStep"}
           - **如果两者都没有**: 步骤无效(无法解析类)
           - **如果两者都有**: 也有效(registry_name 优先)

        **有效处理器配置示例**:
        - {"steps": []} - 空处理器
        - {"steps": [{"registry_name": "normalize"}]} - 注册表步骤
        - {"steps": [{"class": "my.module.Step"}]} - 导入步骤
        - {"name": "MyProcessor", "steps": [...]} - 带名称

        **无效配置示例**:
        - {"type": "act"} - 缺少 "steps"
        - {"steps": "normalize"} - steps 不是列表
        - {"steps": [{}]} - 步骤缺少 class/registry_name
        - {"steps": ["string"]} - 步骤不是字典

        参数:
            config: 要验证的配置字典

        返回:
            如果配置遵循有效的 DataProcessorPipeline 格式,返回 True,否则返回 False
        """
        # 必须有一个 "steps" 字段,包含步骤配置的列表
        if not isinstance(config.get("steps"), list):
            return False

        steps = config["steps"]
        if len(steps) == 0:
            return True  # 空处理器是有效的

        # 每个步骤都必须是一个具有 "class" 或 "registry_name" 的字典
        for step in steps:
            if not isinstance(step, dict):
                return False
            if not ("class" in step or "registry_name" in step):
                return False

        return True

    @classmethod
    def _suggest_processor_migration(cls, model_path: str | Path, original_error: str) -> None:
        """当检测到 JSON 文件但没有处理器配置时引发迁移错误。

        此方法在迁移检测确定模型目录包含配置文件但没有有效处理器配置时调用。
        这通常表示需要迁移的旧 LeRobot 模型。

        **何时调用此方法**:
        - 用户尝试从本地目录加载 DataProcessorPipeline
        - 目录包含 JSON 配置文件
        - 没有 JSON 文件遵循处理器配置格式
        - _should_suggest_migration() 返回 True

        **迁移命令生成**:
        - 构造用户需要运行的确切命令
        - 使用迁移脚本: migrate_policy_normalization.py
        - 自动包含模型路径
        - 示例: "python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path /models/old_model"

        **错误结构**:
        - **始终引发**: ProcessorMigrationError(永不返回)
        - **包含**: model_path、migration_command、original_error
        - **目的**: 强制用户注意迁移需求
        - **用户体验**: 带有要运行的确切命令的清晰可操作错误

        **迁移过程**:
        建议的命令将:
        1. 从旧模型提取归一化统计信息
        2. 创建新的处理器配置(预处理器 + 后处理器)
        3. 从模型中删除归一化层
        4. 保存带有处理器管道的迁移模型

        参数:
            model_path: 需要迁移的模型目录路径
            original_error: 触发迁移检测的错误(用于上下文)

        引发:
            ProcessorMigrationError: 始终引发(此方法永不正常返回)
        """
        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {model_path}"
        )

        raise ProcessorMigrationError(model_path, migration_command, original_error)

    def __len__(self) -> int:
        """返回管道中步骤的数量。"""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | DataProcessorPipeline[TInput, TOutput]:
        """通过索引或切片检索步骤或子管道。

        参数:
            idx: 整数索引或切片对象。

        返回:
            如果 `idx` 是整数,则返回 `ProcessorStep`;
            如果是切片,则返回包含切片步骤的新 `DataProcessorPipeline`。
        """
        if isinstance(idx, slice):
            # 返回包含切片步骤的新管道实例。
            return DataProcessorPipeline(
                steps=self.steps[idx],
                name=self.name,
                to_transition=self.to_transition,
                to_output=self.to_output,
                before_step_hooks=self.before_step_hooks.copy(),
                after_step_hooks=self.after_step_hooks.copy(),
            )
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注册在每个步骤之前调用的函数。

        参数:
            fn: 接受步骤索引和当前转换的可调用对象。
        """
        self.before_step_hooks.append(fn)

    def unregister_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """取消注册 'before_step' 钩子。

        参数:
            fn: 之前注册的确切函数对象。

        引发:
            ValueError: 如果在列表中找不到钩子。
        """
        try:
            self.before_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"在 before_step_hooks 中未找到钩子 {fn}。请确保传递完全相同的函数引用。"
            ) from None

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注册在每个步骤之后调用的函数。

        参数:
            fn: 接受步骤索引和当前转换的可调用对象。
        """
        self.after_step_hooks.append(fn)

    def unregister_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """取消注册 'after_step' 钩子。

        参数:
            fn: 之前注册的确切函数对象。

        引发:
            ValueError: 如果在列表中找不到钩子。
        """
        try:
            self.after_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"在 after_step_hooks 中未找到钩子 {fn}。请确保传递完全相同的函数引用。"
            ) from None

    def reset(self):
        """重置管道中所有有状态步骤的状态。"""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:
        """提供管道的简洁字符串表示。"""
        step_names = [step.__class__.__name__ for step in self.steps]

        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:
            # 对于长管道,显示第一个、第二个和最后一个步骤。
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"

        parts = [f"name='{self.name}'", steps_repr]

        return f"DataProcessorPipeline({', '.join(parts)})"

    def __post_init__(self):
        """验证所有提供的步骤都是 `ProcessorStep` 的实例。"""
        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"步骤 {i} ({type(step).__name__}) 必须继承自 ProcessorStep")

    def transform_features(
        self, initial_features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """按顺序应用所有步骤的特征转换。

        该方法通过每个步骤的 `transform_features` 方法传播特征描述字典,
        允许管道静态确定输出特征规范,而无需处理任何实际数据。

        参数:
            initial_features: 描述初始特征的字典。

        返回:
            所有转换后的最终特征描述。
        """
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = deepcopy(initial_features)

        for _, step in enumerate(self.steps):
            out = step.transform_features(features)
            features = out
        return features

    # 用于处理转换的各个部分的便捷方法。
    def process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """仅通过管道处理转换的观测部分。

        参数:
            observation: 观测字典。

        返回:
            处理后的观测字典。
        """
        transition: EnvTransition = create_transition(observation=observation)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.OBSERVATION]

    def process_action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """仅通过管道处理转换的动作部分。

        参数:
            action: 动作数据。

        返回:
            处理后的动作。
        """
        transition: EnvTransition = create_transition(action=action)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.ACTION]

    def process_reward(self, reward: float | torch.Tensor) -> float | torch.Tensor:
        """仅通过管道处理转换的奖励部分。

        参数:
            reward: 奖励值。

        返回:
            处理后的奖励。
        """
        transition: EnvTransition = create_transition(reward=reward)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.REWARD]

    def process_done(self, done: bool | torch.Tensor) -> bool | torch.Tensor:
        """仅通过管道处理转换的 done 标志。

        参数:
            done: done 标志。

        返回:
            处理后的 done 标志。
        """
        transition: EnvTransition = create_transition(done=done)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.DONE]

    def process_truncated(self, truncated: bool | torch.Tensor) -> bool | torch.Tensor:
        """仅通过管道处理转换的 truncated 标志。

        参数:
            truncated: truncated 标志。

        返回:
            处理后的 truncated 标志。
        """
        transition: EnvTransition = create_transition(truncated=truncated)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.TRUNCATED]

    def process_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """仅通过管道处理转换的 info 字典。

        参数:
            info: info 字典。

        返回:
            处理后的 info 字典。
        """
        transition: EnvTransition = create_transition(info=info)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.INFO]

    def process_complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """仅通过管道处理转换的互补数据部分。

        参数:
            complementary_data: 互补数据字典。

        返回:
            处理后的互补数据字典。
        """
        transition: EnvTransition = create_transition(complementary_data=complementary_data)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.COMPLEMENTARY_DATA]


# 用于语义清晰的类型别名。
RobotProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]
PolicyProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]


class ObservationProcessorStep(ProcessorStep, ABC):
    """专门针对转换中观测的抽象 `ProcessorStep`。"""

    @abstractmethod
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """处理观测字典。子类必须实现此方法。

        参数:
            observation: 来自转换的输入观测字典。

        返回:
            处理后的观测字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `observation` 方法应用于转换的观测。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ObservationProcessorStep 需要转换中的观测。")

        processed_observation = self.observation(observation.copy())
        new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition


class ActionProcessorStep(ProcessorStep, ABC):
    """专门针对转换中动作的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """处理动作。子类必须实现此方法。

        参数:
            action: 来自转换的输入动作。

        返回:
            处理后的动作。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于转换的动作。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("ActionProcessorStep 需要转换中的动作。")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RobotActionProcessorStep(ProcessorStep, ABC):
    """用于处理 `RobotAction`(字典)的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(self, action: RobotAction) -> RobotAction:
        """处理 `RobotAction`。子类必须实现此方法。

        参数:
            action: 输入 `RobotAction` 字典。

        返回:
            处理后的 `RobotAction`。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于转换的动作,确保它是 `RobotAction`。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"动作应该是 RobotAction 类型(字典),但得到了 {type(action)}")

        processed_action = self.action(action.copy())
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class PolicyActionProcessorStep(ProcessorStep, ABC):
    """用于处理 `PolicyAction`(张量或张量字典)的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:
        """处理 `PolicyAction`。子类必须实现此方法。

        参数:
            action: 输入 `PolicyAction`。

        返回:
            处理后的 `PolicyAction`。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于转换的动作,确保它是 `PolicyAction`。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"动作应该是 PolicyAction 类型(张量),但得到了 {type(action)}")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RewardProcessorStep(ProcessorStep, ABC):
    """专门针对转换中奖励的抽象 `ProcessorStep`。"""

    @abstractmethod
    def reward(self, reward) -> float | torch.Tensor:
        """处理奖励。子类必须实现此方法。

        参数:
            reward: 来自转换的输入奖励。

        返回:
            处理后的奖励。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `reward` 方法应用于转换的奖励。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        reward = new_transition.get(TransitionKey.REWARD)
        if reward is None:
            raise ValueError("RewardProcessorStep 需要转换中的奖励。")

        processed_reward = self.reward(reward)
        new_transition[TransitionKey.REWARD] = processed_reward
        return new_transition


class DoneProcessorStep(ProcessorStep, ABC):
    """专门针对转换中 'done' 标志的抽象 `ProcessorStep`。"""

    @abstractmethod
    def done(self, done) -> bool | torch.Tensor:
        """处理 'done' 标志。子类必须实现此方法。

        参数:
            done: 来自转换的输入 'done' 标志。

        返回:
            处理后的 'done' 标志。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `done` 方法应用于转换的 'done' 标志。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        done = new_transition.get(TransitionKey.DONE)
        if done is None:
            raise ValueError("DoneProcessorStep 需要转换中的 done 标志。")

        processed_done = self.done(done)
        new_transition[TransitionKey.DONE] = processed_done
        return new_transition


class TruncatedProcessorStep(ProcessorStep, ABC):
    """专门针对转换中 'truncated' 标志的抽象 `ProcessorStep`。"""

    @abstractmethod
    def truncated(self, truncated) -> bool | torch.Tensor:
        """处理 'truncated' 标志。子类必须实现此方法。

        参数:
            truncated: 来自转换的输入 'truncated' 标志。

        返回:
            处理后的 'truncated' 标志。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `truncated` 方法应用于转换的 'truncated' 标志。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        truncated = new_transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            raise ValueError("TruncatedProcessorStep 需要转换中的 truncated 标志。")

        processed_truncated = self.truncated(truncated)
        new_transition[TransitionKey.TRUNCATED] = processed_truncated
        return new_transition


class InfoProcessorStep(ProcessorStep, ABC):
    """专门针对转换中 'info' 字典的抽象 `ProcessorStep`。"""

    @abstractmethod
    def info(self, info) -> dict[str, Any]:
        """处理 'info' 字典。子类必须实现此方法。

        参数:
            info: 来自转换的输入 'info' 字典。

        返回:
            处理后的 'info' 字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `info` 方法应用于转换的 'info' 字典。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        info = new_transition.get(TransitionKey.INFO)
        if info is None or not isinstance(info, dict):
            raise ValueError("InfoProcessorStep 需要转换中的 info 字典。")

        processed_info = self.info(info.copy())
        new_transition[TransitionKey.INFO] = processed_info
        return new_transition


class ComplementaryDataProcessorStep(ProcessorStep, ABC):
    """针对转换中 'complementary_data' 的抽象 `ProcessorStep`。"""

    @abstractmethod
    def complementary_data(self, complementary_data) -> dict[str, Any]:
        """处理 'complementary_data' 字典。子类必须实现此方法。

        参数:
            complementary_data: 来自转换的输入 'complementary_data'。

        返回:
            处理后的 'complementary_data' 字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `complementary_data` 方法应用于转换的数据。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep 需要转换中的互补数据。")

        processed_complementary_data = self.complementary_data(complementary_data.copy())
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition


class IdentityProcessorStep(ProcessorStep):
    """返回未更改的输入转换和特征的无操作处理器步骤。

    这对于占位符或调试目的很有用。
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """返回未修改的转换。"""
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """返回未修改的特征。"""
        return features
