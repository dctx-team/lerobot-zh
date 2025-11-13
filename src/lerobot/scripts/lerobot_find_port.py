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
帮助查找与你的 MotorsBus 关联的 USB 端口的工具。

示例：

```shell
lerobot-find-port
```
"""

import platform
import time
from pathlib import Path


def find_available_ports():
    from serial.tools import list_ports  # pyserial 库的一部分

    if platform.system() == "Windows":
        # 使用 pyserial 列出 COM 端口
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # 在基于 Unix 的系统上列出 /dev/tty* 端口
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_port():
    print("正在查找 MotorsBus 的所有可用端口。")
    ports_before = find_available_ports()
    print("断开连接前的端口:", ports_before)

    print("从你的 MotorsBus 上拔下 USB 线缆，完成后按 Enter 键。")
    input()  # 等待用户断开设备连接

    time.sleep(0.5)  # 留出一些时间以释放端口
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"此 MotorsBus 的端口是 '{port}'")
        print("重新连接 USB 线缆。")
    elif len(ports_diff) == 0:
        raise OSError(f"无法检测到端口。未发现差异 ({ports_diff})。")
    else:
        raise OSError(f"无法检测到端口。发现了多个端口 ({ports_diff})。")


def main():
    find_port()


if __name__ == "__main__":
    main()
