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
import abc
import builtins
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    策略模型的基础配置类。

    参数:
        n_obs_steps: 传递给策略的观察值所对应的环境步数(包括当前步和之前的额外步)。
        input_shapes: 定义策略输入数据形状的字典。
        output_shapes: 定义策略输出数据形状的字典。
        input_normalization_modes: 键表示模态,值指定要应用的归一化模式的字典。
        output_normalization_modes: 类似于 `input_normalization_modes` 的字典,但用于反归一化到原始尺度。
    """

    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mp
    # `use_amp` 决定是否在训练和评估中使用自动混合精度 (AMP)。使用 AMP 时,会使用自动梯度缩放。
    use_amp: bool = False

    push_to_hub: bool = True
    repo_id: str | None = None

    # 在 Hugging Face hub 上传到私有仓库。
    private: bool | None = None
    # 为您在 hub 上的策略添加标签。
    tags: list[str] | None = None
    # 为您在 hub 上的策略添加许可证。
    license: str | None = None

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"设备 '{self.device}' 不可用。切换到 '{auto_device}'。")
            self.device = auto_device.type

        # 如有必要,自动停用 AMP
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"自动混合精度 (amp) 在设备 '{self.device}' 上不可用。正在停用 AMP。"
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
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
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: 解析原始配置以获取配置子类,以便我们可以应用 cli 覆盖。
        # 这非常不美观,理想情况下我们希望能够使用 draccus 原生支持这一点
        # 类似于 --policy.path (除了 --policy.type 之外)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        with open(config_file) as f:
            config = json.load(f)

        config.pop("type")
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)
