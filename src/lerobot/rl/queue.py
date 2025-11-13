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

import platform
from contextlib import suppress
from queue import Empty
from typing import Any

from torch.multiprocessing import Queue


def get_last_item_from_queue(queue: Queue, block=True, timeout: float = 0.1) -> Any:
    if block:
        try:
            item = queue.get(timeout=timeout)
        except Empty:
            return None
    else:
        item = None

    # 清空队列并只保留最新的参数
    if platform.system() == "Darwin":
        # 在 Mac 上，避免使用 `qsize`，因为其实现不可靠。
        # Python 源代码中的 `qsize` 代码有一条注释：
        # 在 Mac OSX 上引发 NotImplementedError，因为 sem_getvalue() 有问题
        try:
            while True:
                item = queue.get_nowait()
        except Empty:
            pass

        return item

    # 关于使用 qsize 的详细信息请参阅 https://github.com/huggingface/lerobot/issues/1523
    while queue.qsize() > 0:
        with suppress(Empty):
            item = queue.get_nowait()

    return item
