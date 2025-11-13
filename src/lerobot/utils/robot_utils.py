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

import platform
import time


def busy_wait(seconds):
    """
    执行精确的时间等待，在需要时使用忙等待。

    在 Mac 和 Windows 上，`time.sleep` 不够精确，因此使用忙等待循环。
    在 Linux 上，使用标准的 `time.sleep`，因为它足够精确。

    参数:
        seconds: 要等待的秒数。

    注意:
        在 Mac 和 Windows 上，此函数会消耗 CPU 周期以保持精确性。
    """
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # 在 Mac 和 Windows 上，`time.sleep` 不准确，我们需要使用这个 while 循环技巧，
        # 但它会消耗 CPU 周期。
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # 在 Linux 上，time.sleep 是准确的
        if seconds > 0:
            time.sleep(seconds)
