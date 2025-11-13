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
"""train.py 中在线训练循环的在线缓冲区

维护者注意：这里重复了 LeRobotDataset 和 EpisodeAwareSampler 的一些逻辑。我们应该考虑统一为一种方法。
这里我们选择使用 numpy.memmap 作为数据缓冲区的后端。它比使用 HuggingFace Datasets 快得多，因为不需要
转换为中间的非 Python 对象。此外，它支持原地切片和修改，这对于动态缓冲区非常方便。
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _make_memmap_safe(**kwargs) -> np.memmap:
    """创建 numpy memmap，首先检查可用磁盘空间。

    预期的 kwargs 参数为："filename"、"dtype"（必须为 np.dtype）、"mode" 和 "shape"

    有关 dtypes 的信息：
    https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
    """
    if kwargs["mode"].startswith("w"):
        required_space = kwargs["dtype"].itemsize * np.prod(kwargs["shape"])  # 字节
        stats = os.statvfs(Path(kwargs["filename"]).parent)
        available_space = stats.f_bavail * stats.f_frsize  # 字节
        if required_space >= available_space * 0.8:
            raise RuntimeError(
                f"You're about to take up {required_space} of {available_space} bytes available."
            )
    return np.memmap(**kwargs)


class OnlineBuffer(torch.utils.data.Dataset):
    """train.py 中在线训练循环的 FIFO 数据缓冲区。

    尽可能遵循 LeRobotDataset 的协议，以便在线训练循环可以像使用 LeRobotDataset 一样使用它。

    底层数据结构将以循环方式插入数据。始终在最后一个索引之后插入，当到达末尾时，回绕到开始位置。

    数据存储在 numpy memmap 中。
    """

    NEXT_INDEX_KEY = "_next_index"
    OCCUPANCY_MASK_KEY = "_occupancy_mask"
    INDEX_KEY = "index"
    FRAME_INDEX_KEY = "frame_index"
    EPISODE_INDEX_KEY = "episode_index"
    TIMESTAMP_KEY = "timestamp"
    IS_PAD_POSTFIX = "_is_pad"

    def __init__(
        self,
        write_dir: str | Path,
        data_spec: dict[str, Any] | None,
        buffer_capacity: int | None,
        fps: float | None = None,
        delta_timestamps: dict[str, list[float]] | dict[str, np.ndarray] | None = None,
    ):
        """
        在线缓冲区可以从头创建，也可以通过传递与现有缓冲区关联的 `write_dir` 来加载现有的在线缓冲区。

        参数：
            write_dir: 保存 numpy memmap 文件的位置。每个数据键将存储一个 memmap 文件。
                注意，如果文件已存在，它们将以读写模式打开（用于训练恢复）。
            data_spec: 从数据键到数据规范的映射，如 {data_key: {"shape": tuple[int],
                "dtype": np.dtype}}。这应包括你希望记录到缓冲区的所有数据，
                但请注意 "index"、"frame_index" 和 "episode_index" 已由此类处理，
                因此你不需要包含它们。
            buffer_capacity: 缓冲区中最多应存储多少帧。选择时请注意系统的可用磁盘空间。
            fps: 与 LeRobot 数据集中的 fps 概念相同。这里需要为 delta_timestamps 逻辑提供它。
                 如果不使用 delta_timestamps，可以传递 None。
            delta_timestamps: 与 LeRobotDataset 中的 delta_timestamps 概念相同。为了优化目的，
                这在内部转换为 dict[str, np.ndarray]。

        """
        self.set_delta_timestamps(delta_timestamps)
        self._fps = fps
        # 用于在加载帧的时间戳与请求帧的时间戳不够接近时丢弃这些帧的容差（秒）。
        # 仅在提供 `delta_timestamps` 时使用。
        # 减去 1e-4 以考虑可能的数值误差
        self.tolerance_s = 1 / self.fps - 1e-4 if fps is not None else None
        self._buffer_capacity = buffer_capacity
        data_spec = self._make_data_spec(data_spec, buffer_capacity)
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        self._data = {}
        for k, v in data_spec.items():
            self._data[k] = _make_memmap_safe(
                filename=Path(write_dir) / k,
                dtype=v["dtype"] if v is not None else None,
                mode="r+" if (Path(write_dir) / k).exists() else "w+",
                shape=tuple(v["shape"]) if v is not None else None,
            )

    @property
    def delta_timestamps(self) -> dict[str, np.ndarray] | None:
        return self._delta_timestamps

    def set_delta_timestamps(self, value: dict[str, list[float]] | None):
        """设置 delta_timestamps，将值转换为 numpy 数组。

        转换是为了优化 __getitem__。如果数组需要转换为 numpy 数组，循环会慢得多。
        """
        if value is not None:
            self._delta_timestamps = {k: np.array(v) for k, v in value.items()}
        else:
            self._delta_timestamps = None

    def _make_data_spec(self, data_spec: dict[str, Any], buffer_capacity: int) -> dict[str, dict[str, Any]]:
        """为 np.memmap 创建数据规范。"""
        if any(k.startswith("_") for k in data_spec):
            raise ValueError(
                "data_spec keys should not start with '_'. This prefix is reserved for internal logic."
            )
        preset_keys = {
            OnlineBuffer.INDEX_KEY,
            OnlineBuffer.FRAME_INDEX_KEY,
            OnlineBuffer.EPISODE_INDEX_KEY,
            OnlineBuffer.TIMESTAMP_KEY,
        }
        if len(intersection := set(data_spec).intersection(preset_keys)) > 0:
            raise ValueError(
                f"data_spec should not contain any of {preset_keys} as these are handled internally. "
                f"The provided data_spec has {intersection}."
            )
        complete_data_spec = {
            # _next_index 将是一个指针，指向当我们添加更多数据时应该开始填充的下一个索引。
            OnlineBuffer.NEXT_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": ()},
            # 由于 memmap 初始化为全零，这会跟踪哪些索引被真实数据占用，而不是虚拟初始化。
            OnlineBuffer.OCCUPANCY_MASK_KEY: {"dtype": np.dtype("?"), "shape": (buffer_capacity,)},
            OnlineBuffer.INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.FRAME_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.EPISODE_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.TIMESTAMP_KEY: {"dtype": np.dtype("float64"), "shape": (buffer_capacity,)},
        }
        for k, v in data_spec.items():
            complete_data_spec[k] = {"dtype": v["dtype"], "shape": (buffer_capacity, *v["shape"])}
        return complete_data_spec

    def add_data(self, data: dict[str, np.ndarray]):
        """向缓冲区添加新数据，这可能意味着将旧数据移出。

        新数据应包含任意数量的 episode 的所有帧（按顺序）。索引应从 0 开始
        （开发者注意：这可以很容易地泛化）。有关如何构造数据的更多信息，请参阅 `eval.py` 中的
        `rollout` 和 `eval_policy` 函数。

        将传入数据的 index 和 episode_index 移位，以从最后一帧继续。注意这将原地完成！
        """
        if len(missing_keys := (set(self.data_keys).difference(set(data)))) > 0:
            raise ValueError(f"Missing data keys: {missing_keys}")
        new_data_length = len(data[self.data_keys[0]])
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")

        next_index = self._data[OnlineBuffer.NEXT_INDEX_KEY]

        # 健全性检查，确保新数据索引从 0 开始。
        assert data[OnlineBuffer.EPISODE_INDEX_KEY][0].item() == 0
        assert data[OnlineBuffer.INDEX_KEY][0].item() == 0

        # 如有必要，移位传入的索引。
        if self.num_frames > 0:
            last_episode_index = self._data[OnlineBuffer.EPISODE_INDEX_KEY][next_index - 1]
            last_data_index = self._data[OnlineBuffer.INDEX_KEY][next_index - 1]
            data[OnlineBuffer.EPISODE_INDEX_KEY] += last_episode_index + 1
            data[OnlineBuffer.INDEX_KEY] += last_data_index + 1

        # 从 next_index 开始插入新数据。可能需要回绕到开始位置。
        n_surplus = max(0, new_data_length - (self._buffer_capacity - next_index))
        for k in self.data_keys:
            if n_surplus == 0:
                slc = slice(next_index, next_index + new_data_length)
                self._data[k][slc] = data[k]
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY][slc] = True
            else:
                self._data[k][next_index:] = data[k][:-n_surplus]
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY][next_index:] = True
                self._data[k][:n_surplus] = data[k][-n_surplus:]
        if n_surplus == 0:
            self._data[OnlineBuffer.NEXT_INDEX_KEY] = next_index + new_data_length
        else:
            self._data[OnlineBuffer.NEXT_INDEX_KEY] = n_surplus

    @property
    def data_keys(self) -> list[str]:
        keys = set(self._data)
        keys.remove(OnlineBuffer.OCCUPANCY_MASK_KEY)
        keys.remove(OnlineBuffer.NEXT_INDEX_KEY)
        return sorted(keys)

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def num_episodes(self) -> int:
        return len(
            np.unique(self._data[OnlineBuffer.EPISODE_INDEX_KEY][self._data[OnlineBuffer.OCCUPANCY_MASK_KEY]])
        )

    @property
    def num_frames(self) -> int:
        return np.count_nonzero(self._data[OnlineBuffer.OCCUPANCY_MASK_KEY])

    def __len__(self):
        return self.num_frames

    def _item_to_tensors(self, item: dict) -> dict:
        item_ = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                item_[k] = v
            elif isinstance(v, np.ndarray):
                item_[k] = torch.from_numpy(v)
            else:
                item_[k] = torch.tensor(v)
        return item_

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self) or idx < -len(self):
            raise IndexError

        item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}

        if self.delta_timestamps is None:
            return self._item_to_tensors(item)

        episode_index = item[OnlineBuffer.EPISODE_INDEX_KEY]
        current_ts = item[OnlineBuffer.TIMESTAMP_KEY]
        episode_data_indices = np.where(
            np.bitwise_and(
                self._data[OnlineBuffer.EPISODE_INDEX_KEY] == episode_index,
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY],
            )
        )[0]
        episode_timestamps = self._data[OnlineBuffer.TIMESTAMP_KEY][episode_data_indices]

        for data_key in self.delta_timestamps:
            # 注意：此循环中的逻辑从 `load_previous_and_future_frames` 复制而来。
            # 获取用作查询的时间戳以检索前后帧的数据。
            query_ts = current_ts + self.delta_timestamps[data_key]

            # 计算每个查询时间戳与属于该 episode 的所有帧的所有时间戳之间的距离。
            dist = np.abs(query_ts[:, None] - episode_timestamps[None, :])
            argmin_ = np.argmin(dist, axis=1)
            min_ = dist[np.arange(dist.shape[0]), argmin_]

            is_pad = min_ > self.tolerance_s

            # 检查违反的查询时间戳是否都在 episode 范围之外。
            assert (
                (query_ts[is_pad] < episode_timestamps[0]) | (episode_timestamps[-1] < query_ts[is_pad])
            ).all(), (
                f"One or several timestamps unexpectedly violate the tolerance ({min_} > {self.tolerance_s=}"
                ") inside the episode range."
            )

            # 加载此数据键的帧。
            item[data_key] = self._data[data_key][episode_data_indices[argmin_]]

            item[f"{data_key}{OnlineBuffer.IS_PAD_POSTFIX}"] = is_pad

        return self._item_to_tensors(item)

    def get_data_by_key(self, key: str) -> torch.Tensor:
        """返回给定数据键的所有数据作为张量。"""
        return torch.from_numpy(self._data[key][self._data[OnlineBuffer.OCCUPANCY_MASK_KEY]])


def compute_sampler_weights(
    offline_dataset: LeRobotDataset,
    offline_drop_n_last_frames: int = 0,
    online_dataset: OnlineBuffer | None = None,
    online_sampling_ratio: float | None = None,
    online_drop_n_last_frames: int = 0,
) -> torch.Tensor:
    """计算 train.py 中在线训练数据加载器的采样权重。

    参数：
        offline_dataset: 用于离线预训练的 LeRobotDataset。
        online_drop_n_last_frames: 从每个离线数据集 episode 末尾丢弃的帧数。
        online_dataset: 在线训练中使用的 OnlineBuffer。
        online_sampling_ratio: 应从在线数据集采样的数据比例。如果提供了在线数据集，
            则也必须提供此值。
        online_drop_n_first_frames: 参见 `offline_drop_n_last_frames`。这对于在线数据集是相同的。
    返回：
        [offline_dataset; online_dataset] 的权重张量，归一化为 1。

    维护者注意：
        - 这里重复了 EpisodeAwareSampler 的一些逻辑。我们应该考虑统一为一种方法。
        - 当与 `torch.utils.data.WeightedRandomSampler` 一起使用时，它可以完全替代
          `EpisodeAwareSampler`，因为在线数据集相关参数是可选的。唯一缺少的功能是关闭
          shuffling 的能力。
        - 可以轻松添加 `drop_first_n_frames` 和 `episode_indices_to_use` 选项。
          这里没有包括它们是为了避免增加复杂性。
    """
    if len(offline_dataset) == 0 and (online_dataset is None or len(online_dataset) == 0):
        raise ValueError("At least one of `offline_dataset` or `online_dataset` should be contain data.")
    if (online_dataset is None) ^ (online_sampling_ratio is None):
        raise ValueError(
            "`online_dataset` and `online_sampling_ratio` must be provided together or not at all."
        )
    offline_sampling_ratio = 0 if online_sampling_ratio is None else 1 - online_sampling_ratio

    weights = []

    if len(offline_dataset) > 0:
        offline_data_mask_indices = []
        for start_index, end_index in zip(
            offline_dataset.meta.episodes["dataset_from_index"],
            offline_dataset.meta.episodes["dataset_to_index"],
            strict=True,
        ):
            offline_data_mask_indices.extend(range(start_index, end_index - offline_drop_n_last_frames))
        offline_data_mask = torch.zeros(len(offline_dataset), dtype=torch.bool)
        offline_data_mask[torch.tensor(offline_data_mask_indices)] = True
        weights.append(
            torch.full(
                size=(len(offline_dataset),),
                fill_value=offline_sampling_ratio / offline_data_mask.sum(),
            )
            * offline_data_mask
        )

    if online_dataset is not None and len(online_dataset) > 0:
        online_data_mask_indices = []
        episode_indices = online_dataset.get_data_by_key("episode_index")
        for episode_idx in torch.unique(episode_indices):
            where_episode = torch.where(episode_indices == episode_idx)
            start_index = where_episode[0][0]
            end_index = where_episode[0][-1] + 1
            online_data_mask_indices.extend(
                range(start_index.item(), end_index.item() - online_drop_n_last_frames)
            )
        online_data_mask = torch.zeros(len(online_dataset), dtype=torch.bool)
        online_data_mask[torch.tensor(online_data_mask_indices)] = True
        weights.append(
            torch.full(
                size=(len(online_dataset),),
                fill_value=online_sampling_ratio / online_data_mask.sum(),
            )
            * online_data_mask
        )

    weights = torch.cat(weights)

    if weights.sum() == 0:
        weights += 1 / len(weights)
    else:
        weights /= weights.sum()

    return weights
