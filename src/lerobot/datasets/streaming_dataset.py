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
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    Backtrackable,
    LookAheadError,
    LookBackError,
    check_version_compatibility,
    find_float_index,
    get_delta_indices,
    is_float_in_list,
    item_to_torch,
    safe_shard,
)
from lerobot.datasets.video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME, LOOKAHEAD_BACKTRACKTABLE, LOOKBACK_BACKTRACKTABLE


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """具有流式传输功能的LeRobotDataset。

    该类扩展了LeRobotDataset以添加流式传输功能，允许数据以流的方式传输而不是完全加载到内存中。
    这对于可能无法完全加载到内存中的大型数据集，或者当您希望快速浏览数据集而不完全下载它时特别有用。

    关键创新是使用Backtrackable迭代器，该迭代器维护最近项的有界缓冲区，
    允许我们访问delta时间戳的先前帧而无需将整个数据集加载到内存中。

    示例:
        基本用法:
        ```python
        from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

        # 创建带有delta时间戳的流式数据集
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1秒前、0.5秒前、当前
            "action": [0.0, 0.1, 0.2],  # 当前、0.1秒后、0.2秒后
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=1000,
        )

        # 遍历数据集
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item将包含根据delta_timestamps堆叠的帧
            if i >= 10:
                break
        ```
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
    ):
        """初始化StreamingLeRobotDataset。

        参数:
            repo_id (str): 用于获取数据集的仓库ID。
            root (Path | None, optional): 用于下载/写入文件的本地目录。
            episodes (list[int] | None, optional): 如果指定，将仅加载此列表中按episode_index指定的episode。
            image_transforms (Callable | None, optional): 应用于图像数据的转换。
            tolerance_s (float, optional): 时间戳匹配的容差（秒）。
            revision (str, optional): Git修订ID（分支名称、标签或提交哈希）。
            force_cache_sync (bool, optional): 首先同步和刷新本地文件的标志。
            streaming (bool, optional): 是否流式传输数据集或全部加载。默认为True。
            buffer_size (int, optional): 流式传输时用于洗牌的缓冲区大小。默认为1000。
            max_num_shards (int, optional): 将输入数据集重新分片的分片数。默认为16。
            seed (int, optional): 可重现性的随机种子。
            rng (np.random.Generator | None, optional): 随机数生成器。
            shuffle (bool, optional): 是否在遍历之间洗牌数据集。默认为True。
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.streaming = streaming
        self.buffer_size = buffer_size

        # 我们缓存视频解码器以避免在每一帧时重新初始化它们（避免约10倍的速度减慢）
        self.video_decoder_cache = None

        self.root.mkdir(exist_ok=True, parents=True)

        # 加载元数据
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # 检查版本
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # 如果无效则引发ValueError
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.hf_dataset: datasets.IterableDataset = load_dataset(
            self.repo_id if not self.streaming_from_local else str(self.root),
            split="train",
            streaming=self.streaming,
            data_files="data/*/*.parquet",
            revision=self.revision,
        )

        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=100
    ) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    # TODO(fracapuano): 实现多线程预取以加速数据加载。
    # 当前的顺序迭代是一个瓶颈。可以使用生产者-消费者模式
    # 配合ThreadPoolExecutor并行运行`make_frame`（特别是视频解码），
    # 将处理后的项填充到队列中，此迭代器从队列中产出。
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.video_decoder_cache is None:
            self.video_decoder_cache = VideoDecoderCache()

        # 如果shuffle为False则在遍历之间保持相同的种子，否则在遍历之间洗牌数据
        rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng

        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

        idx_to_backtrack_dataset = {
            idx: self._make_backtrackable_dataset(safe_shard(self.hf_dataset, idx, self.num_shards))
            for idx in range(self.num_shards)
        }

        # 在迭代数据集的分片时填充此缓冲区
        # 逻辑是添加2个级别的随机性：
        # (1) 从可用的分片中随机采样一个分片，以及
        # (2) 从(1)采样的分片中采样一帧
        frames_buffer = []
        while available_shards := list(idx_to_backtrack_dataset.keys()):
            shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
            backtrack_dataset = idx_to_backtrack_dataset[shard_key]  # 选择要迭代的分片

            try:
                for frame in self.make_frame(backtrack_dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)  # 从缓冲区采样一个元素
                        yield frames_buffer[i]
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break  # 已采样随机分片，切换分片
            except (
                RuntimeError,
                StopIteration,
            ):  # 注意：从Python 3.7开始，生成器内的StopIteration会抛出RuntimeError
                del idx_to_backtrack_dataset[shard_key]  # 移除已耗尽的分片，继续下一个分片

        # 一旦所有分片都耗尽，洗牌缓冲区并产出剩余的帧
        rng.shuffle(frames_buffer)
        yield from frames_buffer

    def _get_window_steps(
        self, delta_timestamps: dict[str, list[float]] | None = None, dynamic_bounds: bool = False
    ) -> tuple[int, int]:
        if delta_timestamps is None:
            return 1, 1

        if not dynamic_bounds:
            # 固定窗口
            lookback = LOOKBACK_BACKTRACKTABLE
            lookahead = LOOKAHEAD_BACKTRACKTABLE
        else:
            # 根据给定的delta_timesteps动态调整窗口
            all_timestamps = sum(delta_timestamps.values(), [])
            lookback = min(all_timestamps) * self.fps
            lookahead = max(all_timestamps) * self.fps

            # 当lookback>=0时，表示没有提供负时间步
            lookback = 0 if lookback >= 0 else (lookback * -1)

        return lookback, lookahead

    def _make_backtrackable_dataset(self, dataset: datasets.IterableDataset) -> Backtrackable:
        lookback, lookahead = self._get_window_steps(self.delta_timestamps)
        return Backtrackable(dataset, history=lookback, lookahead=lookahead)

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (
                    start_ts + torch.tensor(indices[key]) / self.fps
                ).tolist()  # 注意：为什么不直接使用delta_timestamps？
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.meta.video_keys, [start_ts])

    def _make_padding_camera_frame(self, camera_key: str):
        """给定相机键的可变形状填充帧，以(H, W, C)格式给出"""
        return torch.zeros(self.meta.info["features"][camera_key]["shape"]).permute(-1, 0, 1)

    def _get_video_frame_padding_mask(
        self,
        video_frames: dict[str, torch.Tensor],
        query_timestamps: dict[str, list[float]],
        original_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.BoolTensor]:
        padding_mask = {}

        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue  # 只对可用的视频键进行填充
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    idx = find_float_index(ts, query_timestamps[video_key])
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)

            padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)

        return padding_mask

    def make_frame(
        self, dataset_iterator: Backtrackable, previous_dataset_iterator: Backtrackable | None = None
    ) -> Generator:
        """从数据集迭代器生成一帧"""
        item = next(dataset_iterator)
        item = item_to_torch(item)

        updates = []  # 要应用到从hf_dataset检索的项的"更新"列表（不包括相机特征）

        # 从项中获取episode索引
        ep_idx = item["episode_index"]

        # "timestamp"在每个episode中从0重新开始，而我们需要单个.mp4文件内的全局时间步（由index/fps给出）
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # 如果需要，应用delta查询逻辑
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # 需要时加载视频帧
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)

            # 考虑到episode的边界，某些时间戳可能不可用
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)

            if self.delta_indices is not None:
                # 我们总是返回相同数量的帧。不可用的帧会被填充。
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        yield result

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        episode_boundaries_ts: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = keys_to_timestamps[key]
                # 限制超出episode边界的时间步
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps), *episode_boundaries_ts[key]
                ).tolist()

            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """注意：当使用数据工作器时（例如num_workers>0的DataLoader），不要在主进程中调用此函数
        （例如通过使用num_workers=0的第二个Dataloader）。这会导致段错误。
        这可能是因为在主进程中创建了对视频加载器的内存引用，而子进程无法访问它。
        """

        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if self.streaming and not self.streaming_from_local else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"
            frames = decode_video_frames_torchcodec(
                video_path, query_ts, self.tolerance_s, decoder_cache=self.video_decoder_cache
            )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(self, dataset_iterator: Backtrackable, current_item: dict):
        # TODO(fracapuano): 模块化此函数，重构代码
        """使用可回溯迭代器获取具有delta偏移的帧。

        参数:
            current_item (dict): 来自迭代器的当前项。
            ep_idx (int): Episode索引。

        返回:
            tuple: (query_result, padding) - delta偏移处的帧和填充信息。
        """
        current_episode_idx = current_item["episode_index"]

        # 准备结果
        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # 视觉帧单独解码

            target_frames = []
            is_pad = []

            # 创建一个结果字典，按处理顺序存储帧，然后重构原始顺序进行堆叠
            delta_results = {}

            # 按难度分离和排序delta（先执行较容易的操作）
            negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)  # [-1, -2, -3, ...]
            positive_deltas = sorted([d for d in delta_indices if d > 0])  # [1, 2, 3, ...]
            zero_deltas = [d for d in delta_indices if d == 0]

            # 处理零delta（当前帧）
            for delta in zero_deltas:
                delta_results[delta] = (
                    current_item[key],
                    False,
                )

            # 按难度递增顺序处理负delta
            lookback_failed = False

            last_successful_frame = current_item[key]

            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)

                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]

                        else:
                            raise LookBackError("检索的帧来自不同的episode！")
                    else:
                        raise LookBackError("无法回溯超过历史缓冲区！")

                except LookBackError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookback_failed = True  # 所有后续的负delta也将失败

            # 按难度递增顺序处理正delta
            lookahead_failed = False
            last_successful_frame = current_item[key]

            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)

                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]

                        else:
                            raise LookAheadError("检索的帧来自不同的episode！")
                    else:
                        raise LookAheadError("无法前瞻超过前瞻缓冲区！")

                except LookAheadError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookahead_failed = True  # 所有后续的正delta也将失败

            # 重构原始顺序进行堆叠
            for delta in delta_indices:
                frame, is_padded = delta_results[delta]

                # 添加批次维度以进行堆叠
                target_frames.append(frame)  # frame.unsqueeze(0))
                is_pad.append(is_padded)

            # 堆叠帧并添加到结果中
            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[list[float]]) -> None:
        """
        验证delta_timestamps中的所有键是否对应于数据集中的实际特征。

        引发:
            ValueError: 如果任何delta时间戳键不对应于数据集特征。
        """
        if delta_timestamps is None:
            return

        # 从数据集元数据获取所有可用的特征键
        available_features = set(self.meta.features.keys())

        # 从delta_timestamps获取所有键
        delta_keys = set(delta_timestamps.keys())

        # 查找任何不对应于特征的键
        invalid_keys = delta_keys - available_features

        if invalid_keys:
            raise ValueError(
                f"以下delta_timestamp键不对应于数据集特征：{invalid_keys}。"
                f"可用特征为：{sorted(available_features)}"
            )
