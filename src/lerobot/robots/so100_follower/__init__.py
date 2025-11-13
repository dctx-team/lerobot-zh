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

"""
SO100从动机器人模块

该模块提供了SO100从动机器人的配置类和控制接口。
SO100Follower是一款单臂从动机器人，用于遥操作和数据收集任务。
"""

from .config_so100_follower import SO100FollowerConfig
from .so100_follower import SO100Follower
