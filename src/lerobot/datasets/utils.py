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
import importlib.resources
import json
import logging
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from pprint import pformat
from typing import Any, Deque, Generic, TypeVar

import datasets
import numpy as np
import packaging.version
import pandas
import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import Dataset, concatenate_datasets
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.backward_compatibility import (
    FUTURE_MESSAGE,
    BackwardCompatibilityError,
    ForwardCompatibilityError,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from lerobot.utils.utils import is_valid_numpy_dtype_string

DEFAULT_CHUNK_SIZE = 1000  # 每个块的最大文件数
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # 每个数据文件的最大大小（MB）
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500  # 每个视频文件的最大大小（MB）

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"
LEGACY_DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
LEGACY_DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## {}

"""

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

T = TypeVar("T")


def get_parquet_file_size_in_mb(parquet_path: str | Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            col_metadata = rg_metadata.column(column)
            total_uncompressed_size += col_metadata.total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def get_hf_dataset_size_in_mb(hf_ds: Dataset) -> int:
    return hf_ds.data.nbytes // (1024**2)


def get_hf_dataset_cache_dir(hf_ds: Dataset) -> Path | None:
    if hf_ds.cache_files is None or len(hf_ds.cache_files) == 0:
        return None
    return Path(hf_ds.cache_files[0]["filename"]).parents[2]


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def load_nested_dataset(pq_dir: Path, features: datasets.Features | None = None) -> Dataset:
    """在提供的目录中查找 parquet 文件 {pq_dir}/chunk-xxx/file-xxx.parquet
    将 parquet 文件转换为内存映射的 pyarrow 格式并缓存，以高效使用 RAM
    拼接所有 pyarrow 引用以返回 HF 数据集格式

    Args:
        pq_dir: 包含 parquet 文件的目录
        features: 可选的特征模式，用于确保复杂类型（如图像）的一致性加载
    """
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"提供的目录不包含任何 parquet 文件: {pq_dir}")

    # TODO(rcadene): 设置 num_proc 以加速转换为 pyarrow
    datasets = [Dataset.from_parquet(str(path), features=features) for path in paths]
    return concatenate_datasets(datasets)


def get_parquet_num_frames(parquet_path: str | Path) -> int:
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def get_video_size_in_mb(mp4_path: Path) -> float:
    file_size_bytes = mp4_path.stat().st_size
    file_size_mb = file_size_bytes / (1024**2)
    return file_size_mb


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """通过使用分隔符连接键来扁平化嵌套字典。

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): 要扁平化的字典。
        parent_key (str): 要添加到当前层级键之前的基础键。
        sep (str): 键之间使用的分隔符。

    Returns:
        dict: 扁平化的字典。
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """将带有分隔键的字典展开为嵌套字典。

    Example:
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): 带有扁平化键的字典。
        sep (str): 键中使用的分隔符。

    Returns:
        dict: 嵌套字典。
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """将包含张量或 numpy 数组的字典序列化为 JSON 兼容格式。

    将 torch.Tensor、np.ndarray 和 np.generic 类型转换为列表或 Python 原生类型。

    Args:
        stats (dict): 可能包含不可序列化数值类型的字典。

    Returns:
        dict: 所有值都转换为 JSON 可序列化类型的字典。

    Raises:
        NotImplementedError: 如果值的类型不受支持。
    """
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int, float, list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    """在保存为 Parquet 前将图像字节嵌入到数据集表中。

    此函数通过将图像对象转换为可以存储在 Arrow/Parquet 中的嵌入格式，
    来准备 Hugging Face 数据集以进行序列化。

    Args:
        dataset (datasets.Dataset): 输入数据集，可能包含图像特征。

    Returns:
        datasets.Dataset: 图像已嵌入到表存储中的数据集。
    """
    # 在保存到 parquet 之前将图像字节嵌入表中
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def load_json(fpath: Path) -> Any:
    """从 JSON 文件加载数据。

    Args:
        fpath (Path): JSON 文件的路径。

    Returns:
        Any: 从 JSON 文件加载的数据。
    """
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    """将数据写入 JSON 文件。

    如果父目录不存在则创建它们。

    Args:
        data (dict): 要写入的字典。
        fpath (Path): 输出 JSON 文件的路径。
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_info(info: dict, local_dir: Path) -> None:
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    """从标准文件路径加载数据集信息元数据。

    同时将形状列表转换为元组以保持一致性。

    Args:
        local_dir (Path): 数据集的根目录。

    Returns:
        dict: 数据集信息字典。
    """
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path) -> None:
    """序列化并将数据集统计信息写入标准文件路径。

    Args:
        stats (dict): 统计信息字典（可以包含张量/numpy 数组）。
        local_dir (Path): 数据集的根目录。
    """
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    """递归地将统计信息字典中的数值转换为 numpy 数组。

    Args:
        stats (dict): 统计信息字典。

    Returns:
        dict: 值已转换为 numpy 数组的统计信息字典。
    """
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    """加载数据集统计信息并将数值转换为 numpy 数组。

    如果统计信息文件不存在则返回 None。

    Args:
        local_dir (Path): 数据集的根目录。

    Returns:
        统计信息字典，如果文件未找到则返回 None。
    """
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_tasks(tasks: pandas.DataFrame, local_dir: Path) -> None:
    path = local_dir / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def load_tasks(local_dir: Path) -> pandas.DataFrame:
    tasks = pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
    return tasks


def write_episodes(episodes: Dataset, local_dir: Path) -> None:
    """以 LeRobot v3.0 格式将回合元数据写入 parquet 文件。
    此函数将回合级别的元数据写入单个 parquet 文件。
    主要用于数据集转换（v2.1 → v3.0）和测试夹具。

    Args:
        episodes: 包含回合元数据的 HuggingFace 数据集
        local_dir: 数据集存储的根目录
    """
    episode_size_mb = get_hf_dataset_size_in_mb(episodes)
    if episode_size_mb > DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise NotImplementedError(
            f"Episodes dataset is too large ({episode_size_mb} MB) to write to a single file. "
            f"The current limit is {DEFAULT_DATA_FILE_SIZE_IN_MB} MB. "
            "This function only supports single-file episode metadata. "
        )

    fpath = local_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(fpath)


def load_episodes(local_dir: Path) -> datasets.Dataset:
    episodes = load_nested_dataset(local_dir / EPISODES_DIR)
    # 选择包含回合数据和视频引用的回合特征/列
    # （例如 tasks, dataset_from_index, dataset_to_index, data/chunk_index, data/file_index 等）
    # 这是为了加速对这些数据的访问，而不必加载回合统计信息
    episodes = episodes.select_columns([key for key in episodes.features if not key.startswith("stats/")])
    return episodes


def backward_compatible_episodes_stats(
    stats: dict[str, dict[str, np.ndarray]], episodes: list[int]
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    return dict.fromkeys(episodes, stats)


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    """从文件加载图像为 numpy 数组。

    Args:
        fpath (str | Path): 图像文件路径。
        dtype (np.dtype): 输出数组的期望数据类型。如果是浮点型，
            像素值将缩放到 [0, 1] 范围。
        channel_first (bool): 如果为 True，将图像转换为 (C, H, W) 格式。
            否则保持 (H, W, C) 格式。

    Returns:
        np.ndarray: 图像的 numpy 数组表示。
    """
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:
    """将 Hugging Face 数据集的批次转换为 torch 张量。

    此转换函数将 Hugging Face 数据集格式（pyarrow）的项转换为 torch 张量。
    重要的是，图像从 PIL 对象（H, W, C, uint8）转换为 torch 图像表示
    （C, H, W, float32），范围为 [0, 1]。其他类型转换为 torch.tensor。

    Args:
        items_dict (dict): 表示 Hugging Face 数据集批次的字典。

    Returns:
        dict: 项已转换为 torch 张量的批次。
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def is_valid_version(version: str) -> bool:
    """检查字符串是否为有效的 PEP 440 版本。

    Args:
        version (str): 要检查的版本字符串。

    Returns:
        bool: 如果版本字符串有效则返回 True，否则返回 False。
    """
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    """检查数据集与当前代码库之间的版本兼容性。

    Args:
        repo_id (str): 用于日志记录的仓库 ID。
        version_to_check (str | packaging.version.Version): 数据集的版本。
        current_version (str | packaging.version.Version): 代码库的当前版本。
        enforce_breaking_major (bool): 如果为 True，在主版本不匹配时抛出错误。

    Raises:
        BackwardCompatibilityError: 如果数据集版本来自更新的、不兼容的
            代码库主版本。
    """
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    """返回给定 Hub 仓库上可用的有效版本（分支和标签）。

    Args:
        repo_id (str): Hugging Face Hub 上的仓库 ID。

    Returns:
        list[packaging.version.Version]: 找到的有效版本列表。
    """
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    """返回仓库上可用的指定版本，或最新的兼容版本。

    如果找不到精确版本，它会查找具有相同主版本号且小于或等于
    目标次版本号的最新版本。

    Args:
        repo_id (str): Hugging Face Hub 上的仓库 ID。
        version (str | packaging.version.Version): 目标版本。

    Returns:
        str: 要用作修订版的安全版本字符串（例如 "v1.2.3"）。

    Raises:
        RevisionNotFoundError: 如果仓库没有版本标签。
        BackwardCompatibilityError: 如果只有较旧的主版本可用。
        ForwardCompatibilityError: 如果只有较新的主版本可用。
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""您的数据集必须用代码库版本标记。
            假设 _version_ 是 info.json 中的 codebase_version 值，您可以运行此命令：
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def get_hf_features_from_features(features: dict) -> datasets.Features:
    """将 LeRobot 特征字典转换为 `datasets.Features` 对象。

    Args:
        features (dict): LeRobot 风格的特征字典。

    Returns:
        datasets.Features: 对应的 Hugging Face `datasets.Features` 对象。

    Raises:
        ValueError: 如果特征的形状不受支持。
    """
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def _validate_feature_names(features: dict[str, dict]) -> None:
    """验证特征名称不包含无效字符。

    Args:
        features (dict): LeRobot 特征字典。

    Raises:
        ValueError: 如果任何特征名称包含 '/'。
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """将硬件特定特征转换为 LeRobot 数据集特征字典。

    此函数接收描述硬件输出（如关节状态或相机图像形状）的字典，
    并将其格式化为标准的 LeRobot 特征规范。

    Args:
        hw_features (dict): 将特征名称映射到其类型（关节为 float）或
            形状（图像为元组）的字典。
        prefix (str): 要添加到特征键的前缀（例如 "observation"
            或 "action"）。
        use_video (bool): 如果为 True，图像特征标记为 "video"，否则为 "image"。

    Returns:
        dict: LeRobot 特征字典。
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """基于数据集特征从原始值构造单个数据帧。

    "帧"是包含单个时间步所有数据的字典，
    根据特征规范格式化为 numpy 数组。

    Args:
        ds_features (dict): LeRobot 数据集特征字典。
        values (dict): 来自硬件/环境的原始值字典。
        prefix (str): 用于过滤特征的前缀（例如 "observation"
            或 "action"）。

    Returns:
        dict: 表示单个数据帧的字典。
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """将数据集特征转换为策略特征。

    此函数将数据集的特征规范转换为策略可以使用的格式，
    按类型（例如视觉、状态、动作）对特征进行分类，
    并确保正确的形状（例如图像的通道优先）。

    Args:
        features (dict): LeRobot 数据集特征字典。

    Returns:
        dict: 将特征键映射到 `PolicyFeature` 对象的字典。

    Raises:
        ValueError: 如果图像特征没有 3D 形状。
    """
    # TODO(aliberts): 实现数据集特征中的 "type" 并简化此逻辑
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            # 向后兼容性：针对 LeRobotDataset v2.0 中移植数据集引入的错误 "channel"
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def combine_feature_dicts(*dicts: dict) -> dict:
    """合并 LeRobot 分组特征字典。

    - 对于带有 "names" 的 1D 数值规范（dtype 不是 image/video/string）：
      我们合并名称并重新计算形状。
    - 对于其他（例如 `observation.images.*`），最后一个生效（如果它们相同）。

    Args:
        *dicts: 可变数量的要合并的 LeRobot 特征字典。

    Returns:
        dict: 单个合并的特征字典。

    Raises:
        ValueError: 如果要合并的特征的 dtype 不匹配。
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # 为此特征键初始化或检索累积字典
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # 确保合并条目之间数据类型的一致性
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")

                # 合并特征名称：仅追加新名称以保持顺序且无重复
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # 重新计算形状以反映更新后的特征数量
                target["shape"] = (len(target["names"]),)
            else:
                # 对于图像/视频和非 1D 条目：使用最新的定义覆盖
                out[key] = value
    return out


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
) -> dict:
    """为新数据集的 `info.json` 创建模板字典。

    Args:
        codebase_version (str): LeRobot 代码库的版本。
        fps (int): 数据的帧率（每秒帧数）。
        features (dict): 数据集的 LeRobot 特征字典。
        use_videos (bool): 数据集是否存储视频。
        robot_type (str | None): 使用的机器人类型（如果有）。

    Returns:
        dict: 包含初始数据集元数据的字典。
    """
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": chunks_size or DEFAULT_CHUNK_SIZE,
        "data_files_size_in_mb": data_files_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": video_files_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """检查 delta 时间戳是否为 1/fps 的倍数 +/- 容差。

    这确保将这些 delta 时间戳添加到数据集中的任何现有时间戳时，
    将产生与数据集帧率对齐的值。

    Args:
        delta_timestamps (dict): 字典，其值为以秒为单位的时间增量列表。
        fps (int): 数据集的帧率（每秒帧数）。
        tolerance_s (float): 允许的容差（以秒为单位）。
        raise_value_error (bool): 如果为 True，在失败时抛出错误。

    Returns:
        bool: 如果所有增量都有效则返回 True，否则返回 False。

    Raises:
        ValueError: 如果任何增量超出容差且 `raise_value_error` 为 True。
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                以下 delta_timestamps 超出容差范围。
                请确保它们是 1/{fps} 的倍数 +/- 容差，并相应地
                调整它们的值。
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    """将以秒为单位的 delta 时间戳转换为以帧为单位的 delta 索引。

    Args:
        delta_timestamps (dict): 以秒为单位的时间增量字典。
        fps (int): 数据集的帧率（每秒帧数）。

    Returns:
        dict: 帧 delta 索引的字典。
    """
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def cycle(iterable: Any) -> Iterator[Any]:
    """创建一个数据加载器安全的循环迭代器。

    这相当于 `itertools.cycle`，但可以安全地与具有多个工作进程的
    PyTorch DataLoader 一起使用。
    详见 https://github.com/pytorch/pytorch/issues/23900。

    Args:
        iterable: 要循环的可迭代对象。

    Yields:
        来自可迭代对象的项，在耗尽时从头开始重新启动。
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:
    """在现有的 Hugging Face 仓库上创建分支。

    如果分支已存在，则在创建前将其删除。

    Args:
        repo_id (str): 仓库的 ID。
        branch (str): 要创建的分支名称。
        repo_type (str | None): 仓库的类型（例如 "dataset"）。
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """为 LeRobot 数据集创建 `DatasetCard`。

    关键字参数用于替换卡片模板中的值。
    注意：如果指定，`license` 必须是来自
    https://huggingface.co/docs/hub/repositories-licenses 的有效许可证标识符。

    Args:
        tags (list | None): 要添加到数据集卡片的标签列表。
        dataset_info (dict | None): 数据集的信息字典，将
            显示在卡片上。
        **kwargs: 用于填充卡片模板的其他关键字参数。

    Returns:
        DatasetCard: 生成的数据集卡片对象。
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


def validate_frame(frame: dict, features: dict) -> None:
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)

    # task 是一个特殊的必需字段，不属于常规特征
    if "task" not in actual_features:
        raise ValueError("特征不匹配在 `frame` 字典中:\n缺少特征: {'task'}\n")

    # 从 actual_features 中移除 task 以进行常规特征验证
    actual_features_for_validation = actual_features - {"task"}

    error_message = validate_features_presence(actual_features_for_validation, expected_features)

    common_features = actual_features_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:
    """检查帧中缺失或多余的特征。

    Args:
        actual_features (set[str]): 帧中存在的特征名称集合。
        expected_features (set[str]): 帧中期望的特征名称集合。

    Returns:
        str: 如果存在不匹配则返回错误消息字符串，否则返回空字符串。
    """
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features

    if missing_features or extra_features:
        error_message += "`frame` 字典中的特征不匹配:\n"
        if missing_features:
            error_message += f"缺少的特征: {missing_features}\n"
        if extra_features:
            error_message += f"多余的特征: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:
    """验证单个特征值的 dtype 和形状。

    Args:
        name (str): 特征的名称。
        feature (dict): LeRobot 特征字典中的特征规范。
        value: 要验证的特征值。

    Returns:
        str: 如果验证失败则返回错误消息，否则返回空字符串。

    Raises:
        NotImplementedError: 如果特征 dtype 不支持验证。
    """
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"特征 dtype '{expected_dtype}' 尚未实现。")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:
    """验证期望为 numpy 数组的特征。

    Args:
        name (str): 特征的名称。
        expected_dtype (str): 期望的 numpy dtype 字符串。
        expected_shape (list[int]): 期望的形状。
        value (np.ndarray): 要验证的 numpy 数组。

    Returns:
        str: 如果验证失败则返回错误消息，否则返回空字符串。
    """
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"特征 '{name}' 的 dtype '{actual_dtype}' 不是期望的 dtype '{expected_dtype}'。\n"

        if actual_shape != expected_shape:
            error_message += f"特征 '{name}' 的形状 '{actual_shape}' 不是期望的形状 '{expected_shape}'。\n"
    else:
        error_message += f"特征 '{name}' 不是 'np.ndarray'。期望类型是 '{expected_dtype}'，但提供的类型是 '{type(value)}'。\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:
    """验证期望为图像或视频帧的特征。

    接受 `np.ndarray`（通道优先或通道后置）或 `PIL.Image.Image`。

    Args:
        name (str): 特征的名称。
        expected_shape (list[str]): 期望的形状 (C, H, W)。
        value: 要验证的图像数据。

    Returns:
        str: 如果验证失败则返回错误消息，否则返回空字符串。
    """
    # 注意：像素范围的检查（浮点型为 [0,1]，uint8 为 [0,255]）由图像写入线程完成
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"特征 '{name}' 的形状 '{actual_shape}' 不是期望的形状 '{(c, h, w)}' 或 '{(h, w, c)}'。\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"特征 '{name}' 期望是 'PIL.Image' 或通道优先/通道后置的 'np.ndarray' 类型，但提供的类型是 '{type(value)}'。\n"

    return error_message


def validate_feature_string(name: str, value: str) -> str:
    """验证期望为字符串的特征。

    Args:
        name (str): 特征的名称。
        value (str): 要验证的值。

    Returns:
        str: 如果验证失败则返回错误消息，否则返回空字符串。
    """
    if not isinstance(value, str):
        return f"特征 '{name}' 期望是 'str' 类型，但提供的类型是 '{type(value)}'。\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    """在回合缓冲区写入磁盘之前对其进行验证。

    确保缓冲区具有所需的键，包含至少一个帧，并且
    特征与数据集的规范一致。

    Args:
        episode_buffer (dict): 包含单个回合数据的缓冲区。
        total_episodes (int): 数据集中当前的回合总数。
        features (dict): 数据集的 LeRobot 特征字典。

    Raises:
        ValueError: 如果缓冲区无效。
        NotImplementedError: 如果手动设置了回合索引且不匹配。
    """
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        # TODO(aliberts): 添加使用现有 episode_index 的选项
        raise NotImplementedError(
            "您可能手动为 episode_buffer 提供了与数据集中已有回合总数不匹配的 episode_index。"
            "目前不支持此操作。"
        )

    if episode_buffer["size"] == 0:
        raise ValueError("在调用 `add_episode` 之前，您必须使用 `add_frame` 添加一个或多个帧。")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"来自 `episode_buffer` 的特征与 `features` 中的特征不匹配。"
            f"在 episode_buffer 中但不在 features 中: {buffer_keys - set(features)}"
            f"在 features 中但不在 episode_buffer 中: {set(features) - buffer_keys}"
        )


def to_parquet_with_hf_images(df: pandas.DataFrame, path: Path) -> None:
    """此函数正确地将包含由 HF 数据集编码的图像的 panda DataFrame 写入 parquet。
    这样，它可以被 HF 数据集加载并返回正确格式的图像。
    """
    # TODO(qlhoest): 仅用 `df.to_parquet(path)` 替换这个奇怪的语法
    datasets.Dataset.from_dict(df.to_dict(orient="list")).to_parquet(path)


def item_to_torch(item: dict) -> dict:
    """将字典中的所有项转换为 PyTorch 张量（如果适用）。

    此函数用于将流式数据集中的项转换为 PyTorch 张量。

    Args:
        item (dict): 来自数据集的项字典。

    Returns:
        dict: 所有类张量项都转换为 torch.Tensor 的字典。
    """
    for key, val in item.items():
        if isinstance(val, (np.ndarray, list)) and key not in ["task"]:
            # 将 numpy 数组和列表转换为 torch 张量
            item[key] = torch.tensor(val)
    return item


def is_float_in_list(target, float_list, threshold=1e-6):
    return any(abs(target - x) <= threshold for x in float_list)


def find_float_index(target, float_list, threshold=1e-6):
    for i, x in enumerate(float_list):
        if abs(target - x) <= threshold:
            return i
    return -1


class LookBackError(Exception):
    """
    尝试回溯 Backtrackable 对象的历史记录超出范围时引发的异常。
    """

    pass


class LookAheadError(Exception):
    """
    尝试前瞻 Backtrackable 对象的未来超出范围时引发的异常。
    """

    pass


class Backtrackable(Generic[T]):
    """
    包装任何迭代器/可迭代对象，使您可以向后回溯最多 `history` 个项，
    并向前查看最多 `lookahead` 个项。

    这对于需要访问前后项的流式数据集很有用，
    但无法将整个数据集加载到内存中。

    Example:
    -------
    ```python
    ds = load_dataset("c4", "en", streaming=True, split="train")
    rev = Backtrackable(ds, history=3, lookahead=2)

    x0 = next(rev)  # 向前
    x1 = next(rev)
    x2 = next(rev)

    # 向前查看
    x3_peek = rev.peek_ahead(1)  # 下一项，不移动光标
    x4_peek = rev.peek_ahead(2)  # 向前两项

    # 向后查看
    x1_again = rev.peek_back(1)  # 上一项，不移动光标
    x0_again = rev.peek_back(2)  # 向后两项

    # 向后移动
    x1_back = rev.prev()  # 后退一步
    next(rev)  # 返回 x2，从我们所在的位置继续前进
    ```
    """

    __slots__ = ("_source", "_back_buf", "_ahead_buf", "_cursor", "_history", "_lookahead")

    def __init__(self, iterable: Iterable[T], *, history: int = 1, lookahead: int = 0):
        if history < 1:
            raise ValueError("history must be >= 1")
        if lookahead <= 0:
            raise ValueError("lookahead must be > 0")

        self._source: Iterator[T] = iter(iterable)
        self._back_buf: Deque[T] = deque(maxlen=history)
        self._ahead_buf: Deque[T] = deque(maxlen=lookahead) if lookahead > 0 else deque()
        self._cursor: int = 0
        self._history = history
        self._lookahead = lookahead

    def __iter__(self) -> "Backtrackable[T]":
        return self

    def __next__(self) -> T:
        # 如果我们后退了，先从回退缓冲区消费
        if self._cursor < 0:  # -1 表示"最后一项"，等等
            self._cursor += 1
            return self._back_buf[self._cursor]

        # 如果前瞻缓冲区中有项，优先使用它们
        item = self._ahead_buf.popleft() if self._ahead_buf else next(self._source)

        # 将当前项添加到回退缓冲区并重置光标
        self._back_buf.append(item)
        self._cursor = 0
        return item

    def prev(self) -> T:
        """
        在历史记录中后退一项并返回它。
        如果已经在最旧的缓冲项，则引发 LookBackError。
        """
        if len(self._back_buf) + self._cursor <= 1:
            raise LookBackError("At start of history")

        self._cursor -= 1
        return self._back_buf[self._cursor]

    def peek_back(self, n: int = 1) -> T:
        """
        向后查看 `n` 项（n=1 表示上一项），不移动光标。
        """
        if n < 0 or n + 1 > len(self._back_buf) + self._cursor:
            raise LookBackError("peek_back distance out of range")

        return self._back_buf[self._cursor - (n + 1)]

    def peek_ahead(self, n: int = 1) -> T:
        """
        向前查看 `n` 项（n=1 表示下一项），不移动光标。
        如有必要会填充前瞻缓冲区。
        """
        if n < 1:
            raise LookAheadError("peek_ahead distance must be 1 or more")
        elif n > self._lookahead:
            raise LookAheadError("peek_ahead distance exceeds lookahead limit")

        # 如有必要填充前瞻缓冲区
        while len(self._ahead_buf) < n:
            try:
                item = next(self._source)
                self._ahead_buf.append(item)

            except StopIteration as err:
                raise LookAheadError("peek_ahead: not enough items in source") from err

        return self._ahead_buf[n - 1]

    def history(self) -> list[T]:
        """
        返回缓冲历史记录的副本（最新的在最后）。
        列表长度 ≤ 构造时传递的 `history` 参数。
        """
        if self._cursor == 0:
            return list(self._back_buf)

        # 当光标<0时，切片以保持时间顺序
        return list(self._back_buf)[: self._cursor or None]

    def lookahead_buffer(self) -> list[T]:
        """
        返回当前前瞻缓冲区的副本。
        """
        return list(self._ahead_buf)

    def can_peek_back(self, steps: int = 1) -> bool:
        """
        检查是否可以向后回溯 `steps` 项而不引发 LookBackError。
        """
        return steps <= len(self._back_buf) + self._cursor

    def can_peek_ahead(self, steps: int = 1) -> bool:
        """
        检查是否可以向前查看 `steps` 项。
        这可能涉及尝试填充前瞻缓冲区。
        """
        if self._lookahead > 0 and steps > self._lookahead:
            return False

        # 尝试填充前瞻缓冲区以检查是否可以向前查看那么远
        try:
            while len(self._ahead_buf) < steps:
                if self._lookahead > 0 and len(self._ahead_buf) >= self._lookahead:
                    return False
                item = next(self._source)
                self._ahead_buf.append(item)
            return True
        except StopIteration:
            return False

    def reset_cursor(self) -> None:
        """
        将光标重置到最新位置（相当于调用 next() 直到返回到最新项）。
        """
        self._cursor = 0

    def clear_ahead_buffer(self) -> None:
        """
        清除前瞻缓冲区，丢弃任何预取的项。
        """
        self._ahead_buf.clear()

    def switch_source_iterable(self, new_source: Iterable[T]) -> None:
        """
        将 backtrackable 的源切换到新的可迭代对象，同时保留历史记录。

        这在迭代一系列数据集时很有用。保留前一个源的历史记录，
        但清除前瞻缓冲区。光标重置到当前位置。
        """
        self._source = iter(new_source)
        self.clear_ahead_buffer()
        self.reset_cursor()


def safe_shard(dataset: datasets.IterableDataset, index: int, num_shards: int) -> datasets.Dataset:
    """
    安全地分片数据集。
    """
    shard_idx = min(dataset.num_shards, index + 1) - 1

    return dataset.shard(num_shards, index=shard_idx)
