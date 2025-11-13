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
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import numpy
import PIL
import torch

from lerobot.datasets.video_utils import encode_video_frames


def concatenate_episodes(ep_dicts):
    """
    连接多个回合字典。

    参数:
    - ep_dicts: 回合字典列表

    返回:
    - data_dict: 连接后的数据字典
    """
    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def save_images_concurrently(imgs_array: numpy.array, out_dir: Path, max_workers: int = 4):
    """
    并发保存图像数组到指定目录。

    参数:
    - imgs_array: numpy 图像数组
    - out_dir: 输出目录路径
    - max_workers: 最大工作线程数，默认为 4
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img_array, i, out_dir):
        """保存单张图像"""
        img = PIL.Image.fromarray(img_array)
        img.save(str(out_dir / f"frame_{i:06d}.png"), quality=100)

    num_images = len(imgs_array)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(save_image, imgs_array[i], i, out_dir) for i in range(num_images)]


def get_default_encoding() -> dict:
    """返回 `encode_video_frames` 使用的默认 ffmpeg 编码参数。"""
    signature = inspect.signature(encode_video_frames)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k in ["vcodec", "pix_fmt", "g", "crf"]
    }


def check_repo_id(repo_id: str) -> None:
    """
    检查仓库 ID 的格式是否正确。

    参数:
    - repo_id: 仓库 ID，格式应为 "用户名/数据集名称"

    异常:
    - ValueError: 如果 repo_id 格式不正确
    """
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"""`repo_id` 应包含社区或用户 ID `/` 数据集名称
            (例如 'lerobot/pusht')，但包含的是 '{repo_id}'。"""
        )


# TODO(aliberts): 移除
def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> dict[str, torch.Tensor]:
    """
    为提供的 HuggingFace 数据集计算回合数据索引。依赖于 hf_dataset 的 episode_index 列。

    参数:
    - hf_dataset (datasets.Dataset): 包含回合索引的 HuggingFace 数据集。

    返回:
    - episode_data_index: 包含每个回合的数据索引的字典。字典有两个键:
        - "from": 包含每个回合起始索引的张量。
        - "to": 包含每个回合结束索引的张量。
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    episode_index 是一个整数列表，每个整数代表对应样本的回合索引。
    例如，以下是一个有效的 episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    下面，我们遍历 episode_index 并用每个回合的起始和结束索引填充 episode_data_index 字典。
    对于上面的 episode_index，episode_data_index 字典将如下所示:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # 我们遇到了一个新回合，所以将其起始位置追加到 "from" 列表
            episode_data_index["from"].append(idx)
            # 如果这不是第一个回合，我们将上一个回合的结束位置追加到 "to" 列表
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # 让我们跟踪当前的回合索引
            current_episode = episode_idx
        else:
            # 我们仍在同一个回合中，所以这里没有什么需要做的
            pass
    # 我们已经到达数据集的末尾，所以将最后一个回合的结束位置追加到 "to" 列表
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index
