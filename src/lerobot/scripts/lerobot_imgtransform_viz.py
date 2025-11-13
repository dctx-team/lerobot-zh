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
""" 为给定配置可视化图像变换的效果。

此脚本将生成经过变换的图像示例，这些图像由 LeRobot 数据集输出。
此外，每个单独的变换可以单独可视化，以及组合变换的示例

示例：
```bash
lerobot-imgtransform-viz \
  --repo_id=lerobot/pusht \
  --episodes='[0]' \
  --image_transforms.enable=True
```
"""

import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import draccus
from torchvision.transforms import ToPILImage

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransforms,
    ImageTransformsConfig,
    make_transform_from_config,
)

OUTPUT_DIR = Path("outputs/image_transforms")
to_pil = ToPILImage()


def save_all_transforms(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    """保存所有组合变换的示例。"""
    output_dir_all = output_dir / "all"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    tfs = ImageTransforms(cfg)
    for i in range(1, n_examples + 1):
        transformed_frame = tfs(original_frame)
        to_pil(transformed_frame).save(output_dir_all / f"{i}.png", quality=100)

    print("组合变换示例已保存至：")
    print(f"    {output_dir_all}")


def save_each_transform(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    """分别保存每个单独变换的示例。"""
    if not cfg.enable:
        logging.warning(
            "不会保存单个变换，因为 `image_transforms.enable=False`。要启用，请在 `ImageTransformsConfig` 中将 `enable` 设置为 True，或在命令行中使用 `--image_transforms.enable=True`。"
        )
        return

    print("单个变换示例已保存至：")
    for tf_name, tf_cfg in cfg.tfs.items():
        # 应用一些在 min_max 范围内具有随机值的变换
        output_dir_single = output_dir / tf_name
        output_dir_single.mkdir(parents=True, exist_ok=True)

        tf = make_transform_from_config(tf_cfg)
        for i in range(1, n_examples + 1):
            transformed_frame = tf(original_frame)
            to_pil(transformed_frame).save(output_dir_single / f"{i}.png", quality=100)

        # 应用最小、最大、平均变换
        tf_cfg_kwgs_min = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_max = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_avg = deepcopy(tf_cfg.kwargs)

        for key, (min_, max_) in tf_cfg.kwargs.items():
            avg = (min_ + max_) / 2
            tf_cfg_kwgs_min[key] = [min_, min_]
            tf_cfg_kwgs_max[key] = [max_, max_]
            tf_cfg_kwgs_avg[key] = [avg, avg]

        tf_min = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_min}))
        tf_max = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_max}))
        tf_avg = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_avg}))

        tf_frame_min = tf_min(original_frame)
        tf_frame_max = tf_max(original_frame)
        tf_frame_avg = tf_avg(original_frame)

        to_pil(tf_frame_min).save(output_dir_single / "min.png", quality=100)
        to_pil(tf_frame_max).save(output_dir_single / "max.png", quality=100)
        to_pil(tf_frame_avg).save(output_dir_single / "mean.png", quality=100)

        print(f"    {output_dir_single}")


@draccus.wrap()
def visualize_image_transforms(cfg: DatasetConfig, output_dir: Path = OUTPUT_DIR, n_examples: int = 5):
    """可视化给定配置的图像变换效果。

    Args:
        cfg: 数据集配置，包含图像变换设置。
        output_dir: 保存可视化结果的输出目录。
        n_examples: 要生成的变换示例数量。
    """
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        episodes=cfg.episodes,
        revision=cfg.revision,
        video_backend=cfg.video_backend,
    )

    output_dir = output_dir / cfg.repo_id.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从第 1 个片段的第 1 个相机获取第 1 帧
    original_frame = dataset[0][dataset.meta.camera_keys[0]]
    to_pil(original_frame).save(output_dir / "original_frame.png", quality=100)
    print("\n原始帧已保存至：")
    print(f"    {output_dir / 'original_frame.png'}.")

    save_all_transforms(cfg.image_transforms, original_frame, output_dir, n_examples)
    save_each_transform(cfg.image_transforms, original_frame, output_dir, n_examples)


def main():
    visualize_image_transforms()


if __name__ == "__main__":
    main()
