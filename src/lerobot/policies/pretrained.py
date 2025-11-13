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
import logging
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import packaging
import safetensors
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    策略模型的基类。
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"`{self.__class__.__name__}(config)` 中的参数 config 应该是 "
                "`PreTrainedConfig` 类的实例。要从预训练模型创建模型，请使用 "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"类 {cls.__name__} 必须定义 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"类 {cls.__name__} 必须定义 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        默认情况下，策略被设置为评估模式，使用 `policy.eval()`（停用 dropout 模块）。
        要训练它，您应该首先使用 `policy.train()` 将其设置回训练模式。
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("从本地目录加载权重")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"在 HuggingFace Hub 的 {model_id} 中未找到 {SAFETENSORS_SINGLE_FILE}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # 创建基础参数字典
        kwargs = {"strict": strict}

        # 为支持设备参数的新版本添加设备参数
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # 使用适当的参数加载模型
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # 对于旧版本，如果需要，手动移动到设备
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "在您的 safetensors 版本中，不原生支持在 'cpu' 以外的设备上加载模型权重。"
                "这意味着模型首先在 'cpu' 上加载，然后复制到设备。"
                "这会导致加载时间变慢。"
                "请将 safetensors 更新到 0.4.3 或更高版本以提高性能。"
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        返回要传递给优化器的策略特定参数字典。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """每当环境重置时调用。

        执行诸如清除缓存之类的操作。
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): 拆分为 'forward' 和 'compute_loss'？
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """前向传播方法。

        Args:
            batch (dict[str, Tensor]): 输入批次数据。

        Returns:
            tuple[Tensor, dict | None]: 损失以及可能的其他信息。除了损失是 Tensor 之外，
                所有其他项应该是便于日志记录的原生 Python 类型。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """返回给定观察的动作块（用于动作分块策略），可能是批处理模式。

        使用动作分块的子类应在 `select_action` 中使用此方法来形成缓存以供选择的动作块。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """返回要在环境中运行的一个动作（可能是批处理模式）。

        当模型使用观察历史或输出动作序列时，此方法负责缓存。
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
    ):
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        # 在单次提交中将文件推送到仓库
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id

            self.save_pretrained(saved_path)  # 调用 _save_pretrained 并存储模型张量

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            cfg.save_pretrained(saved_path)  # 调用 _save_pretrained 并存储训练配置

            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="上传策略权重、训练配置和 readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"模型已推送到 {commit_info.repo_url.url}")

    def generate_model_card(
        self, dataset_repo_id: str, model_type: str, license: str | None, tags: list[str] | None
    ) -> ModelCard:
        base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None  # 设置基础模型

        card_data = ModelCardData(
            license=license or "apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
            model_name=model_type,
            datasets=dataset_repo_id,
            base_model=base_model,
        )

        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
        )
        card = ModelCard.from_template(card_data, template_str=template_card)
        card.validate()
        return card
