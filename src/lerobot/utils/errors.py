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


class DeviceNotConnectedError(ConnectionError):
    """设备未连接时引发的异常。"""

    def __init__(self, message="该设备未连接。请先尝试调用 `connect()` 方法。"):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """设备已连接时引发的异常。"""

    def __init__(
        self,
        message="该设备已经连接。请不要重复调用 `connect()` 方法。",
    ):
        self.message = message
        super().__init__(self.message)


class InvalidActionError(ValueError):
    """动作无效时引发的异常。"""

    def __init__(
        self,
        message="该动作无效。请检查数值是否符合动作空间的预期。",
    ):
        self.message = message
        super().__init__(self.message)
