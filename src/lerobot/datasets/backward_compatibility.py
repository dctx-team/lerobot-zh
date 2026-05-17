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

import packaging.version

V30_MESSAGE = """
您请求的数据集 ({repo_id}) 采用 {version} 格式。

我们从 v3.0 版本引入了新格式，该格式与 v2.1 版本不向后兼容。
请使用以下命令将您的数据集更新到新格式：
```
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

如果您遇到问题，请在 [Discord](https://discord.com/invite/s3KuuzsPFb) 上联系 LeRobot 维护者
或在 [GitHub](https://github.com/huggingface/lerobot/issues/new/choose) 上提交问题。
"""

FUTURE_MESSAGE = """
您请求的数据集 ({repo_id}) 仅提供 {version} 格式版本。
由于我们无法确保向前兼容，请更新您当前的 lerobot 版本。
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError("请在 [Discord](https://discord.com/invite/s3KuuzsPFb) 上联系维护者。")
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)
