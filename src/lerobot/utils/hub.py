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

import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args

T = TypeVar("T", bound="HubMixin")


class HubMixin:
    """
    一个 Mixin，包含将对象推送到 Hub 的功能。

    这类似于 huggingface_hub.ModelHubMixin，但更轻量级，对其子类的假设更少
    （特别是它不一定是模型）。

    继承类必须实现 '_save_pretrained' 和 'from_pretrained'。
    """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        将对象保存到本地目录。

        参数：
            save_directory (`str` 或 `Path`):
                对象将保存到的目录路径。
            push_to_hub (`bool`，*可选*，默认为 `False`):
                保存后是否将对象推送到 Huggingface Hub。
            repo_id (`str`，*可选*):
                Hub 上仓库的 ID。仅在 `push_to_hub=True` 时使用。如果未提供，将默认为文件夹名称。
            card_kwargs (`Dict[str, Any]`，*可选*):
                传递给卡片模板以自定义卡片的额外参数。
            push_to_hub_kwargs:
                传递给 [`~HubMixin.push_to_hub`] 方法的额外关键字参数。
        返回：
            `str` 或 `None`：如果 `push_to_hub=True`，返回 Hub 上提交的 url，否则返回 `None`。
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # 保存对象（权重、文件等）
        self._save_pretrained(save_directory)

        # 如果需要，推送到 Hub
        if push_to_hub:
            if repo_id is None:
                repo_id = save_directory.name  # 默认为 `save_directory` 名称
            return self.push_to_hub(repo_id=repo_id, card_kwargs=card_kwargs, **push_to_hub_kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        在子类中重写此方法以定义如何保存对象。

        参数：
            save_directory (`str` 或 `Path`):
                对象文件将保存到的目录路径。
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """
        从 Huggingface Hub 下载对象并实例化它。

        参数：
            pretrained_name_or_path (`str`，`Path`):
                - Hub 上托管的对象的 `repo_id`（字符串），例如 `lerobot/diffusion_pusht`。
                - 或者是包含使用 `.save_pretrained` 保存的对象文件的 `目录` 路径，
                    例如 `../path/to/my_model_directory/`。
            revision (`str`，*可选*):
                Hub 上的修订版本。可以是分支名称、git 标签或任何提交 ID。
                默认为 `main` 分支上的最新提交。
            force_download (`bool`，*可选*，默认为 `False`):
                是否强制（重新）从 Hub 下载文件，覆盖现有缓存。
            proxies (`Dict[str, str]`，*可选*):
                按协议或端点使用的代理服务器字典，例如 `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`。代理将用于每个请求。
            token (`str` 或 `bool`，*可选*):
                用作远程文件的 HTTP bearer 授权的令牌。默认情况下，它将使用
                运行 `huggingface-cli login` 时缓存的令牌。
            cache_dir (`str`，`Path`，*可选*):
                缓存文件存储的文件夹路径。
            local_files_only (`bool`，*可选*，默认为 `False`):
                如果为 `True`，避免下载文件，如果本地缓存文件存在则返回其路径。
            kwargs (`Dict`，*可选*):
                在初始化期间传递给对象的额外 kwargs。
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        将模型检查点上传到 Hub。

        使用 `allow_patterns` 和 `ignore_patterns` 精确过滤应推送到 Hub 的文件。使用
        `delete_patterns` 在同一提交中删除现有的远程文件。有关更多详细信息，请参阅 [`upload_folder`] 参考。

        参数：
            repo_id (`str`):
                要推送到的仓库 ID（例如：`"username/my-model"`）。
            commit_message (`str`，*可选*):
                推送时提交的消息。
            private (`bool`，*可选*):
                创建的仓库是否应该是私有的。
                如果为 `None`（默认），除非组织的默认设置是私有，否则仓库将是公开的。
            token (`str`，*可选*):
                用作远程文件的 HTTP bearer 授权的令牌。默认情况下，它将使用
                运行 `huggingface-cli login` 时缓存的令牌。
            branch (`str`，*可选*):
                推送模型的 git 分支。默认为 `"main"`。
            create_pr (`boolean`，*可选*):
                是否从带有该提交的 `branch` 创建 Pull Request。默认为 `False`。
            allow_patterns (`List[str]` 或 `str`，*可选*):
                如果提供，只推送至少匹配一个模式的文件。
            ignore_patterns (`List[str]` 或 `str`，*可选*):
                如果提供，不推送匹配任何模式的文件。
            delete_patterns (`List[str]` 或 `str`，*可选*):
                如果提供，将从仓库中删除匹配任何模式的远程文件。
            card_kwargs (`Dict[str, Any]`，*可选*):
                传递给卡片模板以自定义卡片的额外参数。

        返回：
            给定仓库中对象提交的 url。
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        if commit_message is None:
            if "Policy" in self.__class__.__name__:
                commit_message = "上传策略"
            elif "Config" in self.__class__.__name__:
                commit_message = "上传配置"
            else:
                commit_message = f"上传 {self.__class__.__name__}"

        # 在单个提交中将文件推送到仓库
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, card_kwargs=card_kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )
