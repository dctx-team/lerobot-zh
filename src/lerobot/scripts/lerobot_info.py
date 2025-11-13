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

"""
使用此脚本可以快速获取系统配置的摘要。
它应该能够在不安装任何 LeRobot 依赖项或 LeRobot 本身的情况下运行。

示例:

```shell
lerobot-info
```
"""

import importlib
import platform


def get_package_version(package_name: str) -> str:
    """获取包的版本（如果存在），否则返回 'N/A'。"""
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "Installed (version not found)")
    except ImportError:
        return "N/A"


def get_sys_info() -> dict:
    """运行此函数以获取基本系统信息，帮助跟踪问题和错误。"""
    # 通用包版本
    info = {
        "lerobot version": get_package_version("lerobot"),
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Huggingface Hub version": get_package_version("huggingface_hub"),
        "Datasets version": get_package_version("datasets"),
        "Numpy version": get_package_version("numpy"),
    }

    # PyTorch 和 GPU 特定信息
    torch_version = "N/A"
    torch_cuda_available = "N/A"
    cuda_version = "N/A"
    gpu_model = "N/A"
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            cuda_version = torch.version.cuda
            # 获取第一个可用 GPU 的名称
            gpu_model = torch.cuda.get_device_name(0)
    except ImportError:
        # 如果未安装 torch，将使用默认的 "N/A" 值。
        pass

    info.update(
        {
            "PyTorch version": torch_version,
            "Is PyTorch built with CUDA support?": torch_cuda_available,
            "Cuda version": cuda_version,
            "GPU model": gpu_model,
            "Using GPU in script?": "<fill in>",
        }
    )

    return info


def format_dict_for_markdown(d: dict) -> str:
    """将字典格式化为 Markdown 友好的项目符号列表。"""
    return "\n".join([f"- {prop}: {val}" for prop, val in d.items()])


def main():
    system_info = get_sys_info()
    print("\n请将以下文本复制粘贴到您的 GitHub 问题中，并填写最后一项。\n")
    print(format_dict_for_markdown(system_info))


if __name__ == "__main__":
    main()
