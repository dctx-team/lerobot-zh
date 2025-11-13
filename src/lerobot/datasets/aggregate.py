#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_parquet_file_size_in_mb,
    get_video_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """验证所有数据集元数据具有一致的属性。

    确保所有数据集具有相同的 fps、robot_type 和 features，以保证
    将它们聚合为单个数据集时的兼容性。

    Args:
        all_metadata: 要验证的 LeRobotDatasetMetadata 对象列表。

    Returns:
        tuple: 包含来自第一个元数据的 (fps, robot_type, features) 的元组。

    Raises:
        ValueError: 如果任何元数据的 fps、robot_type 或 features
                   与列表中第一个元数据不同。
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """使用新的索引和任务映射更新数据 DataFrame 以进行聚合。

    调整情节索引、帧索引和任务索引，以考虑
    目标数据集中先前聚合的数据。

    Args:
        df: 包含要更新的数据的 DataFrame。
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。

    Returns:
        pd.DataFrame: 索引已调整的更新后的 DataFrame。
    """

    def _update(row):
        row["episode_index"] = row["episode_index"] + dst_meta.info["total_episodes"]
        row["index"] = row["index"] + dst_meta.info["total_frames"]
        task = src_meta.tasks.iloc[row["task_index"]].name
        row["task_index"] = dst_meta.tasks.loc[task].task_index.item()
        return row

    return df.apply(_update, axis=1)


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
):
    """使用新的块、文件和时间戳索引更新元数据 DataFrame。

    调整所有索引和时间戳，以考虑目标数据集中
    先前聚合的数据和视频。

    Args:
        df: 包含要更新的元数据的 DataFrame。
        dst_meta: 目标数据集元数据。
        meta_idx: 包含当前元数据块和文件索引的字典。
        data_idx: 包含当前数据块和文件索引的字典。
        videos_idx: 包含当前视频索引和时间戳的字典。

    Returns:
        pd.DataFrame: 索引和时间戳已调整的更新后的 DataFrame。
    """

    def _update(row):
        row["meta/episodes/chunk_index"] = row["meta/episodes/chunk_index"] + meta_idx["chunk"]
        row["meta/episodes/file_index"] = row["meta/episodes/file_index"] + meta_idx["file"]
        row["data/chunk_index"] = row["data/chunk_index"] + data_idx["chunk"]
        row["data/file_index"] = row["data/file_index"] + data_idx["file"]
        for key, video_idx in videos_idx.items():
            row[f"videos/{key}/chunk_index"] = row[f"videos/{key}/chunk_index"] + video_idx["chunk"]
            row[f"videos/{key}/file_index"] = row[f"videos/{key}/file_index"] + video_idx["file"]
            row[f"videos/{key}/from_timestamp"] = (
                row[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            row[f"videos/{key}/to_timestamp"] = (
                row[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]
            )

        row["dataset_from_index"] = row["dataset_from_index"] + dst_meta.info["total_frames"]
        row["dataset_to_index"] = row["dataset_to_index"] + dst_meta.info["total_frames"]
        row["episode_index"] = row["episode_index"] + dst_meta.info["total_episodes"]
        return row

    return df.apply(_update, axis=1)


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """将多个 LeRobot 数据集聚合为单个统一数据集。

    这是协调聚合过程的主要函数，通过以下步骤：
    1. 加载并验证所有源数据集元数据
    2. 创建具有统一任务的新目标数据集
    3. 从所有源数据集聚合视频、数据和元数据
    4. 使用适当的统计信息完成聚合数据集

    Args:
        repo_ids: 要聚合的数据集的仓库 ID 列表。
        aggr_repo_id: 聚合输出数据集的仓库 ID。
        roots: 源数据集的根路径列表（可选）。
        aggr_root: 聚合数据集的根路径（可选）。
        data_files_size_in_mb: 数据文件的最大大小（MB）（默认为 DEFAULT_DATA_FILE_SIZE_IN_MB）
        video_files_size_in_mb: 视频文件的最大大小（MB）（默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB）
        chunk_size: 每个块的最大文件数（默认为 DEFAULT_CHUNK_SIZE）
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size)
        data_idx = aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size)

        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """将源数据集的视频块聚合到目标数据集中。

    根据文件大小限制处理视频文件拼接和轮换。
    当超过大小限制时创建新的视频文件。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        videos_idx: 跟踪视频块和文件索引的字典。
        video_files_size_in_mb: 视频文件的最大大小（MB）（默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB）
        chunk_size: 每个块的最大文件数（默认为 DEFAULT_CHUNK_SIZE）

    Returns:
        dict: 包含当前块和文件索引的更新后的 videos_idx。
    """
    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            # 如果创建了新文件，我们不想增加 latest_duration
            update_latest_duration = False

            if not dst_path.exists():
                # 首次写入到此目标文件
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                continue  # 不再继续累积，文件已复制到位

            # 在追加前检查文件大小
            src_size = get_video_size_in_mb(src_path)
            dst_size = get_video_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                # 轮换到新的块/文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
            else:
                # 获取此视频的时间戳偏移
                timestamps_shift_s = dst_meta.info["total_frames"] / dst_meta.info["fps"]

                # 追加到现有视频文件
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                # 追加时更新 latest_duration（会偏移时间戳！）
                update_latest_duration = not update_latest_duration

        # 使用此键的最终块和文件索引更新 videos_idx
        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

        if update_latest_duration:
            videos_idx[key]["latest_duration"] += timestamps_shift_s

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """将源数据集的数据块聚合到目标数据集中。

    读取源数据文件，更新索引以匹配聚合数据集，
    并使用适当的文件轮换将它们写入目标。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        data_idx: 跟踪数据块和文件索引的字典。

    Returns:
        dict: 包含当前块和文件索引的更新后的 data_idx。
    """
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        data_idx = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )

    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """将源数据集的元数据聚合到目标数据集中。

    读取源元数据文件，更新所有索引和时间戳，
    并使用适当的文件轮换将它们写入目标。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        meta_idx: 跟踪元数据块和文件索引的字典。
        data_idx: 跟踪数据块和文件索引的字典。
        videos_idx: 跟踪视频索引和时间戳的字典。

    Returns:
        dict: 包含当前块和文件索引的更新后的 meta_idx。
    """
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        for k in videos_idx:
            videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

        meta_idx = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    return meta_idx


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """根据大小约束将数据追加到现有 parquet 文件或创建新文件。

    当超过大小限制时管理文件轮换，以防止单个文件
    变得过大。处理常规 parquet 文件和包含图像的文件。

    Args:
        df: 要写入 parquet 文件的 DataFrame。
        src_path: 源文件路径（用于大小估算）。
        idx: 包含当前 'chunk' 和 'file' 索引的字典。
        max_mb: 轮换前允许的最大文件大小（MB）。
        chunk_size: 递增块索引前每个块的最大文件数。
        default_path: 生成文件路径的格式字符串。
        contains_images: 数据是否包含需要特殊处理的图像。
        aggr_root: 聚合数据集的根路径。

    Returns:
        dict: 包含当前块和文件索引的更新后的索引字典。
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    """通过写入摘要文件和统计信息完成数据集聚合。

    写入任务文件、包含总计数和拆分的信息文件，以及
    来自所有源数据集的聚合统计信息。

    Args:
        aggr_meta: 聚合数据集元数据。
        all_metadata: 所有源数据集元数据对象的列表。
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
