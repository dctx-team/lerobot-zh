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
import contextlib
import gc
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import pandas as pd
import PIL.Image
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    _validate_feature_names,
    check_delta_timestamps,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    flatten_dict,
    get_delta_indices,
    get_hf_dataset_cache_dir,
    get_hf_dataset_size_in_mb,
    get_hf_features_from_features,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    get_safe_version,
    get_video_size_in_mb,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_tasks,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    validate_episode_buffer,
    validate_frame,
    write_info,
    write_json,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import (
    VideoFrame,
    concatenate_video_files,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_duration_in_s,
    get_video_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

CODEBASE_VERSION = "v3.0"


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def url_root(self) -> str:
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """用于创建此数据集的代码库版本。"""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    @property
    def data_path(self) -> str:
        """parquet 文件的格式化字符串。"""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """视频文件的格式化字符串。"""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """录制此数据集时使用的机器人类型。"""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """数据采集期间使用的每秒帧数。"""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """数据集中包含的所有特征。"""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """访问以图像形式存储的视觉模态的键。"""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """访问以视频形式存储的视觉模态的键。"""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """访问视觉模态的键（无论其存储方法如何）。"""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """向量模态的各个维度的名称。"""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """不同特征的形状。"""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """可用的总情节数。"""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """此数据集中保存的总帧数。"""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """此数据集中执行的不同任务的总数。"""
        return self.info["total_tasks"]

    @property
    def chunks_size(self) -> int:
        """每个块的最大文件数。"""
        return self.info["chunks_size"]

    @property
    def data_files_size_in_mb(self) -> int:
        """数据文件的最大大小（以兆字节为单位）。"""
        return self.info["data_files_size_in_mb"]

    @property
    def video_files_size_in_mb(self) -> int:
        """视频文件的最大大小（以兆字节为单位）。"""
        return self.info["video_files_size_in_mb"]

    def get_task_index(self, task: str) -> int | None:
        """
        给定一个自然语言的任务，如果该任务已存在于数据集中，则返回其 task_index，
        否则返回 None。
        """
        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=tasks)
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            # 更新磁盘上的数据
            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """将情节元数据保存到 parquet 文件并更新情节元数据的 Hugging Face 数据集。

        此函数从字典中处理情节元数据，将其转换为 Hugging Face 数据集，
        并将其保存为 parquet 文件。它根据大小约束处理新 parquet 文件的创建和
        现有文件的更新。保存元数据后，它会重新加载 Hugging Face 数据集以确保它是最新的。

        注意：我们需要同时更新 parquet 文件和 HF 数据集：
        - `pandas` 将 parquet 文件加载到 RAM 中
        - `datasets` 依赖于 pyarrow 的内存映射（无 RAM）。它要么将 parquet 文件转换为磁盘上的 pyarrow 缓存，
          要么直接从 pyarrow 缓存加载。
        """
        # 将缓冲区转换为 HF 数据集
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        ep_dataset = datasets.Dataset.from_dict(episode_dict)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        df = pd.DataFrame(ep_dataset)
        num_frames = episode_dict["length"][0]

        if self.episodes is None:
            # 为由第一个情节数据组成的新数据集初始化索引和帧计数
            chunk_idx, file_idx = 0, 0
            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [0]
            df["dataset_to_index"] = [num_frames]
        else:
            # 从最新的 parquet 文件检索信息
            latest_ep = self.episodes[-1]
            chunk_idx = latest_ep["meta/episodes/chunk_index"]
            file_idx = latest_ep["meta/episodes/file_index"]

            latest_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.data_files_size_in_mb:
                # 达到大小限制，准备新的 parquet 文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)

            # 用新行更新现有的 pandas 数据框
            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [latest_ep["dataset_to_index"]]
            df["dataset_to_index"] = [latest_ep["dataset_to_index"] + num_frames]

            if latest_size_in_mb + ep_size_in_mb < self.data_files_size_in_mb:
                # 未达到大小限制，将最新的数据框与新的数据框连接
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)

                # 内存优化
                del latest_df
                gc.collect()

        # 将结果数据框从 RAM 写入磁盘
        path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        if self.episodes is not None:
            # 删除情节缓存目录，这是避免缓存膨胀所必需的
            cached_dir = get_hf_dataset_cache_dir(self.episodes)
            if cached_dir is not None:
                shutil.rmtree(cached_dir)

        self.episodes = load_episodes(self.root)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # 更新信息
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        write_stats(self.stats, self.root)

    def update_video_info(self, video_key: str | None = None) -> None:
        """
        警告：此函数从第一个情节视频中写入信息，隐式假设所有视频都以相同的方式编码。
        此外，这意味着它假设第一个情节存在。
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.video_path.format(
                    video_key=video_key, chunk_index=0, file_index=0
                )
                self.info["features"][key]["info"] = get_video_info(video_path)

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """在数据集创建后更新块和文件大小设置。

        这允许用户自定义存储组织而无需修改构造函数。
        这些设置控制情节如何分块以及文件在创建新文件之前可以增长多大。

        Args:
            chunks_size: 每个块目录的最大文件数。如果为 None，则保留当前值。
            data_files_size_in_mb: 数据 parquet 文件的最大大小（以 MB 为单位）。如果为 None，则保留当前值。
            video_files_size_in_mb: 视频文件的最大大小（以 MB 为单位）。如果为 None，则保留当前值。
        """
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info["chunks_size"] = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info["data_files_size_in_mb"] = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info["video_files_size_in_mb"] = video_files_size_in_mb

        # 更新磁盘上的信息文件
        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """获取当前的块和文件大小设置。

        Returns:
            包含 chunks_size、data_files_size_in_mb 和 video_files_size_in_mb 的字典。
        """
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """创建 LeRobotDataset 的元数据。"""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        根据 2 个不同的用例，有 2 种模式可用于实例化此类：

        1. 您的数据集已经存在：
            - 在本地磁盘的 'root' 文件夹中。当您在本地录制数据集并且可能尚未将其推送到 hub 时，
              通常会出现这种情况。使用 'root' 实例化此类将直接从磁盘加载您的数据集。
              这可以在您离线时发生（没有互联网连接）。

            - 在 Hugging Face Hub 上，地址为 https://huggingface.co/datasets/{repo_id}，
              而不在本地磁盘的 'root' 文件夹中。使用此 'repo_id' 实例化此类将从该地址下载数据集并加载它，
              前提是您的数据集符合 codebase_version v3.0。如果您的数据集是在此新格式之前创建的，
              系统将提示您使用我们的转换脚本从 v2.1 转换到 v3.0，
              您可以在 lerobot/datasets/v30/convert_dataset_v21_to_v30.py 中找到。


        2. 您的数据集尚不存在（无论是在本地磁盘上还是在 Hub 上）：您可以使用 'create' 类方法
           创建一个空的 LeRobotDataset。这可用于录制数据集或将现有数据集移植到 LeRobotDataset 格式。


        就文件而言，LeRobotDataset 封装了 3 个主要内容：
            - 元数据：
                - info 包含有关数据集的各种信息，如形状、键、fps 等。
                - stats 存储不同模态的数据集统计信息以进行归一化
                - tasks 包含数据集每个任务的提示，可用于任务条件训练。
            - hf_dataset（来自 datasets.Dataset），它将从 parquet 文件读取任何值。
            - 视频（可选），从中加载帧以与 parquet 文件中的数据同步。

        从其根路径来看，典型的 LeRobotDataset 如下所示：
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        请注意，这种基于文件的结构旨在尽可能通用。多个情节被合并到分块文件中，
        这提高了存储效率和加载性能。数据集的结构完全在 info.json 文件中描述，
        可以在下载任何实际数据之前轻松下载或直接在 hub 上查看。使用的文件类型非常简单，
        不需要复杂的工具来读取，它只使用 .parquet、.json 和 .mp4 文件（以及用于 README 的 .md）。

        Args:
            repo_id (str): 这是将用于获取数据集的 repo id。在本地，数据集将存储在 root/repo_id 下。
            root (Path | None, optional): 用于下载/写入文件的本地目录。您也可以设置 LEROBOT_HOME
                环境变量以指向不同的位置。默认为 '~/.cache/huggingface/lerobot'。
            episodes (list[int] | None, optional): 如果指定，这将仅加载由其 episode_index
                在此列表中指定的情节。默认为 None。
            image_transforms (Callable | None, optional): 您可以在此处传递来自
                torchvision.transforms.v2 的标准 v2 图像转换，这些转换将应用于视觉模态
                （无论它们来自视频还是图像）。默认为 None。
            delta_timestamps (dict[list[float]] | None, optional): _description_。默认为 None。
            tolerance_s (float, optional): 以秒为单位的容差，用于确保数据时间戳实际上与 fps 值同步。
                它在数据集初始化时用于确保每个时间戳与下一个时间戳的间隔为 1/fps +/- tolerance_s。
                这也适用于从视频文件解码的帧。它还用于检查 `delta_timestamps`（当提供时）是否为 1/fps 的倍数。
                默认为 1e-4。
            revision (str, optional): 可选的 Git 修订 id，可以是分支名称、标签或提交哈希。
                默认为当前代码库版本标签。
            force_cache_sync (bool, optional): 首先同步和刷新本地文件的标志。如果为 True 且文件已存在于本地缓存中，
                则速度会更快。但是，加载的文件可能与 hub 上的版本不同步，特别是如果您指定了 'revision'。
                默认为 False。
            download_videos (bool, optional): 下载视频的标志。请注意，当设置为 True 但视频文件已存在于本地磁盘上时，
                它们不会再次下载。默认为 True。
            video_backend (str | None, optional): 用于解码视频的视频后端。在平台上可用时默认为 torchcodec；
                否则，默认为 'pyav'。您还可以使用 Torchvision 使用的 'pyav' 解码器（以前是默认选项），
                或 'video_reader'，这是 Torchvision 的另一个解码器。
            batch_encoding_size (int, optional): 在批量编码视频之前要累积的情节数。
                设置为 1 表示立即编码（默认），或设置为更高的值表示批量编码。默认为 1。
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # 未使用的属性
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # 加载元数据
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # 加载实际数据
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            # 检查缓存的数据集是否包含所有请求的情节
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        # 设置 delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download(self, download_videos: bool = True) -> None:
        """从给定的 'repo_id' 下载提供版本的数据集。如果给出了 'episodes'，
        这将仅下载这些情节（由其 episode_index 选择）。如果 'episodes' 为 None，
        则将下载整个数据集。由于 snapshot_download 的行为，如果文件已存在于 'local_dir' 中，
        它们将不会再次下载。
        """
        # TODO(rcadene, aliberts): 实现更快的传输
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # 情节存储在相同的文件中，因此我们只返回唯一的路径
        fpaths = list(set(fpaths))
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset 包含所有观测、状态、动作、奖励等。"""
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """检查缓存的数据集是否包含所有请求的情节。"""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        # 从缓存的数据集中获取可用的情节索引
        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset["episode_index"]
        }

        # 确定请求的情节
        if self.episodes is None:
            # 请求所有情节 - 检查我们是否拥有元数据中的所有情节
            requested_episodes = set(range(self.meta.total_episodes))
        else:
            # 请求特定情节
            requested_episodes = set(self.episodes)

        # 检查所有请求的情节是否在缓存数据中可用
        return requested_episodes.issubset(available_episodes)

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """数据采集期间使用的每秒帧数。"""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """选定情节中的帧数。"""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """选定的情节数。"""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """hf_dataset 的特征。"""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # 在当前情节范围外填充值
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """注意：当使用数据工作线程时（例如，DataLoader 的 num_workers>0），
        不要在主进程中调用此函数（例如，使用 num_workers=0 的第二个 Dataloader）。
        这将导致段错误。这可能是因为在主进程中创建了对视频加载器的内存引用，
        而子进程无法访问它。
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # 情节按顺序存储在单个 mp4 上以减少文件数量。
            # 因此，我们加载此 mp4 上情节的起始时间戳，
            # 并相应地移动查询时间戳。
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # 添加任务作为字符串
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size 和 task 是特殊情况，不在 self.features 中
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        此函数仅将帧添加到 episode_buffer。除了图像 —— 它们被写入临时目录 —— 之外，
        没有任何内容被写入磁盘。要保存这些帧，然后需要调用 'save_episode()' 方法。
        """
        # 如果需要，将 torch 转换为 numpy
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # 自动将 frame_index 和 timestamp 添加到情节缓冲区
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))  # 处理后从帧中删除任务

        # 将帧特征添加到 episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        这将把 self.episode_buffer 中的当前情节保存到磁盘。

        视频编码根据 batch_encoding_size 自动处理：
        - 如果 batch_encoding_size == 1：每个情节后立即编码视频
        - 如果 batch_encoding_size > 1：视频以批处理方式编码。

        Args:
            episode_data (dict | None, optional): 包含要保存的情节数据的字典。如果为 None，
                这将保存 self.episode_buffer 中的当前情节，该情节使用 'add_frame' 填充。
                默认为 None。
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size 和 task 是特殊情况，不会添加到 hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # 如果有新任务，则更新任务和任务索引
        self.meta.save_episode_tasks(episode_tasks)

        # 给定自然语言的任务，找到它们对应的任务索引
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index、episode_index、task_index 已在上面处理，image 和 video
            # 通过将图像路径和帧信息存储为元数据来单独处理
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # 等待图像写入器结束，以便可以计算图像的情节统计信息
        self._wait_image_writer()
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` 需要在编码视频后执行
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # 检查是否应该触发批量编码
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # 重置情节缓冲区并清理临时图像（如果在视频编码期间尚未删除）
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None):
        """
        批量保存多个情节的视频。

        Args:
            start_episode: 起始情节索引（包含）
            end_episode: 结束情节索引（不包含）。如果为 None，则编码从 start_episode 到当前情节的所有情节。
        """
        if end_episode is None:
            end_episode = self.num_episodes

        logging.info(
            f"Batch encoding {self.batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self.meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self.meta.episodes[start_episode]["data/file_index"]
        episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logging.info(f"Encoding videos for episode {ep_idx}")

            if (
                self.meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self.meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                # 当前情节在新的块或文件中。
                # 保存先前的情节数据框并通过重新加载来更新 Hugging Face 数据集。
                episode_df.to_parquet(episode_df_path)
                self.meta.episodes = load_episodes(self.root)

                # 加载新的情节数据框
                chunk_idx = self.meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self.meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            # 将当前情节的视频元数据保存到数据框
            video_ep_metadata = {}
            for video_key in self.meta.video_keys:
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )  # 允许 NaN 值与整数一起使用

            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            self.meta.episodes = load_episodes(self.root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """将情节数据保存到 parquet 文件并更新帧数据的 Hugging Face 数据集。

        此函数从缓冲区处理情节数据，将其转换为 Hugging Face 数据集，
        并将其保存为 parquet 文件。它根据大小约束处理新 parquet 文件的创建和
        现有文件的更新。保存数据后，它会重新加载 Hugging Face 数据集以确保它是最新的。

        注意：我们需要同时更新 parquet 文件和 HF 数据集：
        - `pandas` 将 parquet 文件加载到 RAM 中
        - `datasets` 依赖于 pyarrow 的内存映射（无 RAM）。它要么将 parquet 文件转换为磁盘上的 pyarrow 缓存，
          要么直接从 pyarrow 缓存加载。
        """
        # 将缓冲区转换为 HF 数据集
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        ep_num_frames = len(ep_dataset)
        df = pd.DataFrame(ep_dataset)

        if self.meta.episodes is None:
            # 为由第一个情节数据组成的新数据集初始化索引和帧计数
            chunk_idx, file_idx = 0, 0
            latest_num_frames = 0
        else:
            # 从最新的 parquet 文件检索信息
            latest_ep = self.meta.episodes[-1]
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)
            latest_num_frames = get_parquet_num_frames(latest_path)

            # 确定是否需要新的 parquet 文件
            if latest_size_in_mb + ep_size_in_mb >= self.meta.data_files_size_in_mb:
                # 达到大小限制，准备新的 parquet 文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                latest_num_frames = 0
            else:
                # 用新行更新现有的 parquet 文件
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)

                # 内存优化
                del latest_df
                gc.collect()

        # 将结果数据框从 RAM 写入磁盘
        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.meta.image_keys) > 0:
            to_parquet_with_hf_images(df, path)
        else:
            df.to_parquet(path)

        if self.hf_dataset is not None:
            # 删除 hf 数据集缓存目录，这是避免缓存膨胀所必需的
            cached_dir = get_hf_dataset_cache_dir(self.hf_dataset)
            if cached_dir is not None:
                shutil.rmtree(cached_dir)

        self.hf_dataset = self.load_hf_dataset()

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": latest_num_frames,
            "dataset_to_index": latest_num_frames + ep_num_frames,
        }
        return metadata

    def _save_episode_video(self, video_key: str, episode_index: int):
        # 将情节帧编码为临时视频
        ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        ep_size_in_mb = get_video_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if self.meta.episodes is None or (
            f"videos/{video_key}/chunk_index" not in self.meta.episodes.column_names
            or f"videos/{video_key}/file_index" not in self.meta.episodes.column_names
        ):
            # 为由第一个情节数据组成的新数据集初始化索引
            chunk_idx, file_idx = 0, 0
            latest_duration_in_s = 0.0
            new_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            # 从最新更新的视频文件中检索信息（可能是几个情节之前）
            latest_ep = self.meta.episodes[episode_index - 1]
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"]
            file_idx = latest_ep[f"videos/{video_key}/file_index"]

            latest_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_video_size_in_mb(latest_path)
            latest_duration_in_s = get_video_duration_in_s(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.meta.video_files_size_in_mb:
                # 将临时情节视频移动到数据集中的新视频文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                new_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                # 更新最新的视频文件
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # 删除临时目录
        shutil.rmtree(str(ep_path.parent))

        # 更新视频信息（仅在第一个情节编码时需要，因为它从情节 0 读取）
        if episode_index == 0:
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)  # 确保视频信息始终正确写入

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        # 清理当前情节缓冲区的图像文件
        if delete_images:
            # 等待异步图像写入器完成
            if self.image_writer is not None:
                self._wait_image_writer()
            episode_index = self.episode_buffer["episode_index"]
            if isinstance(episode_index, np.ndarray):
                episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # 重置缓冲区
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        每当在并行化的 DataLoader 中包装此数据集时，都需要首先调用此方法以删除 image_writer，
        以便 LeRobotDataset 对象可以被 pickle 化和并行化。
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """等待异步图像写入器完成。"""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> dict:
        """
        使用 ffmpeg 将存储为 png 的帧转换为 mp4 视频。
        注意：`encode_video_frames` 是一个阻塞调用。使其异步不应加快编码速度，
        因为使用 ffmpeg 的视频编码已经在使用多线程。
        """
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_video_frames(img_dir, temp_path, self.fps, overwrite=True)
        shutil.rmtree(img_dir)
        return temp_path

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> "LeRobotDataset":
        """从头创建 LeRobot 数据集以记录数据。"""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """由多个底层 `LeRobotDataset` 组成的数据集。

    底层的 `LeRobotDataset` 实际上被连接在一起，此类采用 `LeRobotDataset` 的大部分 API 结构。
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # 构造底层数据集，传递除 `transform` 和 `delta_timestamps` 之外的所有内容，
        # 这些由此类处理。
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # 禁用所有数据集中不通用的任何数据键。注意：我们可能会在此类的未来版本中放宽此限制。
        # 目前，这至少是为了能够使用 PyTorch 的默认 DataLoader 整理函数所必需的。
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): 我们不应该对具有不同范围的多个机器人的数据集执行此聚合。
        # 相反，我们应该为每个机器人进行一次归一化。
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """返回从数据集 repo_id 到此类自动创建的数据集索引的映射。

        此索引作为数据键合并在 `__getitem__` 返回的字典中。
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """返回 repo_id_to_index 的逆映射。"""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """数据采集期间使用的每秒帧数。

        注意：目前，这依赖于 __init__ 中的检查以确保所有子数据集具有相同的信息。
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """如果此数据集从 mp4 文件加载视频帧，则返回 True。

        如果它只从 png 文件加载图像，则返回 False。

        注意：目前，这依赖于 __init__ 中的检查以确保所有子数据集具有相同的信息。
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """访问相机的图像和视频流的键。"""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """访问需要解码为图像的视频帧的键。

        注意：如果数据集仅包含图像，则为空，
        如果数据集仅包含视频，则等于 `self.cameras`，
        或者在混合图像/视频数据集的情况下，甚至可以是 `self.cameras` 的子集。
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """样本/帧数。"""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """情节数。"""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """以秒为单位的容差，用于在时间戳与请求的帧不够接近时丢弃加载的帧。
        仅在提供 `delta_timestamps` 或从 mp4 文件加载视频帧时使用。
        """
        # 1e-4 用于考虑可能的数值误差
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # 根据索引确定要从哪个数据集获取项目。
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
