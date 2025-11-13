#!/usr/bin/env python

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
import importlib
import logging


def is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """复制自 https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    检查包规范是否存在并获取其版本，以避免导入本地目录。
    **注意：** 这并非适用于所有包。
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # 获取包版本的主要方法
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # 备用方法：仅适用于 "torch" 和包含 "dev" 的版本
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # 检查版本是否包含 "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # 如果无法导入包，则不可用
                    package_exists = False
            elif pkg_name == "grpc":
                package = importlib.import_module(pkg_name)
                package_version = getattr(package, "__version__", "N/A")
            else:
                # 对于 "torch" 以外的包，不尝试备用方法，并设置为不可用
                package_exists = False
        logging.debug(f"检测到 {pkg_name} 版本：{package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_transformers_available = is_package_available("transformers")
