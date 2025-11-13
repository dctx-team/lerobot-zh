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
""" 可视化 LeRobotDataset 类型数据集中任意回合的 **所有** 帧的数据。

注意：回合的最后一帧并不总是对应最终状态。
这是因为我们的数据集由状态到状态的转换组成，直到倒数第三个状态关联到到达最终状态的最后一个动作。
然而，可能不存在从最终状态到另一个状态的转换。

注意：此脚本旨在可视化用于训练神经网络的数据。
~所见即所得~。在可视化图像模态时，通常会观察到有损压缩伪影，因为这些图像是从压缩的 mp4 视频中解码的，
以节省磁盘空间。应用的压缩因子已经过调整，不会影响成功率。

示例：

- 可视化存储在本地机器上的数据：
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- 使用本地查看器可视化存储在远程机器上的数据：
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- 通过流式传输可视化存储在远程机器上的数据：
（您需要将 websocket 端口转发到远程机器，使用
`ssh -L 9087:localhost:9087 username@remote-host`）
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # 在 `rr.init` 之后手动调用 python 垃圾回收器，以避免在使用 `num_workers` > 0 的数据加载器迭代时挂起阻塞刷新
    # TODO(rcadene): 当 rerun 0.16 版本发布时移除 `gc.collect`，该版本包含修复
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # 遍历批次
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # 显示每个相机图像
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): 添加 `.compress()`? 它是无损的吗？
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # 显示动作空间的每个维度（例如执行器命令）
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalar(val.item()))

            # 显示观测状态空间的每个维度（例如关节空间中的智能体位置）
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalar(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalar(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

    if mode == "local" and save:
        # 本地保存 .rrd
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # 防止进程退出，因为它正在服务 websocket 连接
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="包含 LeRobotDataset 数据集的 Hugging Face 仓库名称（例如 `lerobot/pusht`）。",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="要可视化的回合。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="本地存储数据集的根目录（例如 `--root data`）。默认情况下，数据集将从 Hugging Face 缓存文件夹加载，或者如果可用，从 hub 下载。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="设置 `--save 1` 时写入 .rrd 文件的目录路径。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader 加载的批次大小。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 用于加载数据的进程数。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "查看模式，可选 'local' 或 'distant'。"
            "'local' 要求数据位于本地机器上。它会生成一个查看器以在本地可视化数据。"
            "'distant' 在存储数据的远程机器上创建服务器。"
            "通过在本地机器上使用 `rerun ws://localhost:PORT` 连接到服务器来可视化数据。"
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="当 `--mode distant` 设置时 rerun.io 的 Web 端口。",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="当 `--mode distant` 设置时 rerun.io 的 WebSocket 端口。",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "在 `--output-dir` 提供的目录中保存 .rrd 文件。"
            "它还会停用查看器的生成。"
            "通过在本地机器上运行 `rerun path/to/file.rrd` 来可视化数据。"
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "用于确保数据时间戳符合数据集 fps 值的容差（秒）。"
            "此参数传递给 LeRobotDataset 的构造函数，映射到其 tolerance_s 构造函数参数。"
            "如果未给出，默认为 1e-4。"
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
