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

"""客户端侧：环境以等于 1/fps 的时间分辨率演化"""

DEFAULT_FPS = 30

"""服务器端：运行推理的最大频率为 1/fps"""
DEFAULT_INFERENCE_LATENCY = 1 / DEFAULT_FPS

"""服务器端：观测队列的超时时间（秒）"""
DEFAULT_OBS_QUEUE_TIMEOUT = 2

# 所有动作分块策略
SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "pi0", "tdmpc", "vqbet"]

# TODO: 添加所有其他机器人
SUPPORTED_ROBOTS = ["so100_follower", "so101_follower"]
