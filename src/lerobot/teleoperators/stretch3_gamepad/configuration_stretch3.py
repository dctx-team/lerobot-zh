#!/usr/bin/env python

# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 有关许可证下特定权限和限制的详细信息，请参阅许可证。

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("stretch3")
@dataclass
class Stretch3GamePadConfig(TeleoperatorConfig):
    mock: bool = False
