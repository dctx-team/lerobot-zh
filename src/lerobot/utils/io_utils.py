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
import json
import warnings
from pathlib import Path
from typing import TypeVar

import imageio

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def write_video(video_path, stacked_frames, fps):
    # 过滤掉来自 pkg_resources 的弃用警告
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    从 `fpath` 加载 JSON 数据,并递归地用相应的值填充 `obj`(严格匹配结构和类型)。
    `obj` 中的元组在 JSON 数据中应为列表,会被转换回元组。
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        递归地用来自 `source` 的数据覆写 `target` 中的结构,
        对结构和类型执行严格检查。
        返回更新后的 `target` 版本(对于元组尤为重要)。
        """

        # 如果目标是字典,源也必须是字典。
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # 检查它们是否拥有完全相同的键集合。
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # 递归更新每个键。
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # 如果目标是列表,源也必须是列表。
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # 检查长度
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # 递归更新每个元素。
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # 如果目标是元组,源在 JSON 中必须是列表,
        # 我们会将其转换回元组。
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # 转换每个元素,形成新的元组。
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # 返回一个全新的元组(Python 中元组是不可变的)。
            return tuple(converted_items)

        # 否则,我们处理的是"原始类型"(int, float, str, bool, None)。
        else:
            # 检查确切类型。如果这些必须一一匹配,则执行:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # 执行就地/递归反序列化
    updated_obj = _deserialize(obj, data)
    return updated_obj
