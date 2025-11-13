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

"""
一个通用脚本，用于将具有内置归一化层的 LeRobot 策略迁移到新的基于管道的处理器系统。

此脚本执行以下步骤：
1.  从本地路径或 Hugging Face Hub 加载预训练的策略模型及其配置。
2.  扫描模型的状态字典以提取所有特征的归一化统计信息（例如，均值、标准差、最小值、最大值）。
3.  创建两个新的处理器管道：
    - 预处理器：对输入（观察）和输出（动作）进行归一化。
    - 后处理器：对输出（动作）进行反归一化以供推理使用。
4.  从模型的状态字典中移除原始的归一化层，创建一个"干净的"模型。
5.  将新的干净模型、预处理器、后处理器和生成的模型卡保存到新目录。
6.  可选地将所有新的工件推送到 Hugging Face Hub。

用法：
    python src/lerobot/processor/migrate_policy_normalization.py \
        --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
        --push-to-hub \
        --branch main

注意：此脚本现在使用 `lerobot.policies.factory` 中的现代 `make_pre_post_processors` 和
`make_policy_config` 工厂函数来创建处理器和配置，确保与当前代码库的一致性。

该脚本从旧模型的 state_dict 中提取归一化统计信息，使用工厂函数创建干净的处理器管道，
并保存与新 PolicyProcessorPipeline 架构兼容的迁移模型。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION


def extract_normalization_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """
    扫描模型的 state_dict 以查找并提取归一化统计信息。

    此函数基于一组预定义的模式识别对应归一化层的键（例如，均值、标准差、最小值、最大值），
    并将它们组织到嵌套字典中。

    参数：
        state_dict: 预训练策略模型的状态字典。

    返回：
        嵌套字典，外层键是特征名称（例如，'observation.state'），内层键是统计类型（'mean'、'std'），
        映射到它们对应的张量值。
    """
    stats = {}

    # 定义要匹配的模式及其要移除的前缀
    normalization_patterns = [
        "normalize_inputs.buffer_",
        "unnormalize_outputs.buffer_",
        "normalize_targets.buffer_",
        "normalize.",  # 必须在 normalize_* 模式之后
        "unnormalize.",  # 必须在 unnormalize_* 模式之后
        "input_normalizer.",
        "output_normalizer.",
        "normalalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",
        "unnormalize_targets.",
    ]

    # 处理 state_dict 中的每个键
    for key, tensor in state_dict.items():
        # 尝试每个模式
        for pattern in normalization_patterns:
            if key.startswith(pattern):
                # 提取模式后的剩余部分
                remaining = key[len(pattern) :]
                parts = remaining.split(".")

                # 至少需要特征名称和统计类型
                if len(parts) >= 2:
                    # 最后一部分是统计类型（mean、std、min、max 等）
                    stat_type = parts[-1]
                    # 其他所有部分是特征名称
                    feature_name = ".".join(parts[:-1]).replace("_", ".")

                    # 添加到 stats
                    if feature_name not in stats:
                        stats[feature_name] = {}
                    stats[feature_name][stat_type] = tensor.clone()

                # 仅处理第一个匹配的模式
                break

    return stats


def detect_features_and_norm_modes(
    config: dict[str, Any], stats: dict[str, dict[str, torch.Tensor]]
) -> tuple[dict[str, PolicyFeature], dict[FeatureType, NormalizationMode]]:
    """
    从模型配置和统计信息推断策略特征和归一化模式。

    此函数首先尝试直接从策略的配置文件中查找特征定义和归一化映射。
    如果此信息不存在，它将从提取的归一化统计信息中推断，使用张量形状
    来确定特征形状，并根据特定统计键（例如，'mean'/'std' 与 'min'/'max'）
    的存在来确定归一化模式。如果无法推断，它将应用合理的默认值。

    参数：
        config: 来自 `config.json` 的策略配置字典。
        stats: 从模型的 state_dict 中提取的归一化统计信息。

    返回：
        包含以下内容的元组：
        - 将特征名称映射到 `PolicyFeature` 对象的字典。
        - 将 `FeatureType` 枚举映射到 `NormalizationMode` 枚举的字典。
    """
    features = {}
    norm_modes = {}

    # 首先，检查配置中是否有 normalization_mapping
    if "normalization_mapping" in config:
        print(f"Found normalization_mapping in config: {config['normalization_mapping']}")
        # 从配置中提取归一化模式
        for feature_type_str, mode_str in config["normalization_mapping"].items():
            # 将字符串转换为 FeatureType 枚举
            try:
                if feature_type_str == "VISUAL":
                    feature_type = FeatureType.VISUAL
                elif feature_type_str == "STATE":
                    feature_type = FeatureType.STATE
                elif feature_type_str == "ACTION":
                    feature_type = FeatureType.ACTION
                else:
                    print(f"Warning: Unknown feature type '{feature_type_str}', skipping")
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse feature type '{feature_type_str}', skipping")
                continue

            # 将字符串转换为 NormalizationMode 枚举
            try:
                if mode_str == "MEAN_STD":
                    mode = NormalizationMode.MEAN_STD
                elif mode_str == "MIN_MAX":
                    mode = NormalizationMode.MIN_MAX
                elif mode_str == "IDENTITY":
                    mode = NormalizationMode.IDENTITY
                else:
                    print(
                        f"Warning: Unknown normalization mode '{mode_str}' for feature type '{feature_type_str}'"
                    )
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse normalization mode '{mode_str}', skipping")
                continue

            norm_modes[feature_type] = mode

    # 尝试从配置中提取
    if "features" in config:
        for key, feature_config in config["features"].items():
            shape = feature_config.get("shape", feature_config.get("dim"))
            shape = (shape,) if isinstance(shape, int) else tuple(shape)

            # 确定特征类型
            if "image" in key or "visual" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key:
                feature_type = FeatureType.STATE
            elif ACTION in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE  # 默认

            features[key] = PolicyFeature(feature_type, shape)

    # 如果配置中没有特征，则从 stats 推断
    if not features:
        for key, stat_dict in stats.items():
            # 从任何统计张量获取形状
            tensor = next(iter(stat_dict.values()))
            shape = tuple(tensor.shape)

            # 根据键确定特征类型
            if "image" in key or "visual" in key or "pixels" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key or "joint" in key or "position" in key:
                feature_type = FeatureType.STATE
            elif ACTION in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE

            features[key] = PolicyFeature(feature_type, shape)

    # 如果归一化模式不在配置中，则根据可用的统计信息确定
    if not norm_modes:
        for key, stat_dict in stats.items():
            if key in features:
                if "mean" in stat_dict and "std" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MEAN_STD
                elif "min" in stat_dict and "max" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MIN_MAX

    # 如果未检测到，使用默认归一化模式
    if FeatureType.VISUAL not in norm_modes:
        norm_modes[FeatureType.VISUAL] = NormalizationMode.MEAN_STD
    if FeatureType.STATE not in norm_modes:
        norm_modes[FeatureType.STATE] = NormalizationMode.MIN_MAX
    if FeatureType.ACTION not in norm_modes:
        norm_modes[FeatureType.ACTION] = NormalizationMode.MEAN_STD

    return features, norm_modes


def remove_normalization_layers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    创建一个新的 state_dict，移除所有与归一化相关的层。

    此函数过滤原始状态字典，排除与一组预定义的与归一化模块相关的模式匹配的任何键。

    参数：
        state_dict: 原始模型状态字典。

    返回：
        仅包含核心模型权重的新状态字典，不包含任何归一化参数。
    """
    new_state_dict = {}

    # 要移除的模式
    remove_patterns = [
        "normalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",  # 为目标归一化添加的模式
        "normalize.",
        "unnormalize.",
        "input_normalizer.",
        "output_normalizer.",
        "normalizer.",
    ]

    for key, tensor in state_dict.items():
        should_remove = any(pattern in key for pattern in remove_patterns)
        if not should_remove:
            new_state_dict[key] = tensor

    return new_state_dict


def clean_state_dict(
    state_dict: dict[str, torch.Tensor], remove_str: str = "._orig_mod"
) -> dict[str, torch.Tensor]:
    """
    从状态字典中的所有键移除子字符串（例如 '._orig_mod'）。

    参数：
        state_dict (dict): 原始状态字典。
        remove_str (str): 要从键中移除的子字符串。

    返回：
        dict: 具有清理后的键的新状态字典。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace(remove_str, "")
        new_state_dict[new_k] = v
    return new_state_dict


def convert_features_to_policy_features(features_dict: dict[str, dict]) -> dict[str, PolicyFeature]:
    """
    将特征字典从旧配置格式转换为新的 `PolicyFeature` 格式。

    参数：
        features_dict: 旧格式的特征字典，其中值是简单字典（例如，`{"shape": [7]}`）。

    返回：
        将特征名称映射到 `PolicyFeature` 数据类对象的字典。
    """
    converted_features = {}

    for key, feature_dict in features_dict.items():
        # 根据键确定特征类型
        if "image" in key or "visual" in key:
            feature_type = FeatureType.VISUAL
        elif "state" in key:
            feature_type = FeatureType.STATE
        elif ACTION in key:
            feature_type = FeatureType.ACTION
        else:
            feature_type = FeatureType.STATE

        # 从特征字典获取形状
        shape = feature_dict.get("shape", feature_dict.get("dim"))
        shape = (shape,) if isinstance(shape, int) else tuple(shape) if shape is not None else ()

        converted_features[key] = PolicyFeature(feature_type, shape)

    return converted_features


def load_model_from_hub(
    repo_id: str, revision: str | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """
    从 Hugging Face Hub 下载并加载模型的 state_dict 和配置。

    参数：
        repo_id: Hub 上的仓库 ID（例如，'lerobot/aloha'）。
        revision: 要使用的特定 git 修订版本（分支、标签或提交哈希）。

    返回：
        包含模型的状态字典、策略配置和训练配置的元组。
    """
    # 下载文件。
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    train_config_path = hf_hub_download(repo_id=repo_id, filename="train_config.json", revision=revision)

    # 加载 state_dict
    state_dict = load_safetensors(safetensors_path)

    # 加载配置
    with open(config_path) as f:
        config = json.load(f)

    with open(train_config_path) as f:
        train_config = json.load(f)

    return state_dict, config, train_config


def main():
    parser = argparse.ArgumentParser(
        description="将具有归一化层的策略模型迁移到新的管道系统"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        help="预训练模型的路径（hub 仓库或本地目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="迁移模型的输出目录（默认：与 pretrained-path 相同）",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="将迁移的模型推送到 hub")
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="用于推送的 Hub 仓库 ID（默认：与 pretrained-path 相同）",
    )
    parser.add_argument("--revision", type=str, default=None, help="要加载的模型修订版本")
    parser.add_argument("--private", action="store_true", help="将 hub 仓库设为私有")
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="推送到 hub 时使用的 Git 分支。如果指定，将自动创建 PR（默认：直接推送到 main）",
    )

    args = parser.parse_args()

    # 加载模型和配置
    print(f"Loading model from {args.pretrained_path}...")
    if os.path.isdir(args.pretrained_path):
        # 本地目录
        state_dict = load_safetensors(os.path.join(args.pretrained_path, "model.safetensors"))
        with open(os.path.join(args.pretrained_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(args.pretrained_path, "train_config.json")) as f:
            train_config = json.load(f)
    else:
        # Hub 仓库
        state_dict, config, train_config = load_model_from_hub(args.pretrained_path, args.revision)

    # 提取归一化统计信息
    print("Extracting normalization statistics...")
    stats = extract_normalization_stats(state_dict)

    print(f"Found normalization statistics for: {list(stats.keys())}")

    # 检测输入特征和归一化模式
    print("Detecting features and normalization modes...")
    features, norm_map = detect_features_and_norm_modes(config, stats)

    print(f"Detected features: {list(features.keys())}")
    print(f"Normalization modes: {norm_map}")

    # 从 state_dict 中移除归一化层
    print("Removing normalization layers from model...")
    new_state_dict = remove_normalization_layers(state_dict)
    new_state_dict = clean_state_dict(new_state_dict, remove_str="._orig_mod")

    removed_keys = set(state_dict.keys()) - set(new_state_dict.keys())
    if removed_keys:
        print(f"Removed {len(removed_keys)} normalization layer keys")

    # 确定输出路径
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if os.path.isdir(args.pretrained_path):
            output_dir = Path(args.pretrained_path).parent / f"{Path(args.pretrained_path).name}_migrated"
        else:
            output_dir = Path(f"./{args.pretrained_path.replace('/', '_')}_migrated")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 从配置中提取策略类型
    if "type" not in config:
        raise ValueError("Policy type not found in config.json. The config must contain a 'type' field.")

    policy_type = config["type"]
    print(f"Detected policy type: {policy_type}")

    # 清理配置 - 移除不应传递给配置构造函数的字段
    cleaned_config = dict(config)

    # 移除不属于配置类构造函数的字段
    fields_to_remove = ["normalization_mapping", "type"]
    for field in fields_to_remove:
        if field in cleaned_config:
            print(f"Removing '{field}' field from config")
            del cleaned_config[field]

    # 如果存在 input_features 和 output_features，将它们转换为 PolicyFeature 对象
    if "input_features" in cleaned_config:
        cleaned_config["input_features"] = convert_features_to_policy_features(
            cleaned_config["input_features"]
        )
    if "output_features" in cleaned_config:
        cleaned_config["output_features"] = convert_features_to_policy_features(
            cleaned_config["output_features"]
        )

    # 将归一化映射添加到配置
    cleaned_config["normalization_mapping"] = norm_map

    # 使用工厂创建策略配置
    print(f"Creating {policy_type} policy configuration...")
    policy_config = make_policy_config(policy_type, **cleaned_config)

    # 使用工厂创建策略实例
    print(f"Instantiating {policy_type} policy...")
    policy_class = get_policy_class(policy_type)
    policy = policy_class(policy_config)

    # 加载清理后的状态字典
    policy.load_state_dict(new_state_dict, strict=True)
    print("Successfully loaded cleaned state dict into policy model")

    # 使用工厂创建预处理器和后处理器
    print("Creating preprocessor and postprocessor using make_pre_post_processors...")
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats)

    # 如果推送到 hub，确定 hub 仓库 ID
    hub_repo_id = None
    if args.push_to_hub:
        if args.hub_repo_id:
            hub_repo_id = args.hub_repo_id
        else:
            if not os.path.isdir(args.pretrained_path):
                # 使用带有 "_migrated" 后缀的相同仓库
                hub_repo_id = f"{args.pretrained_path}_migrated"
            else:
                raise ValueError("--hub-repo-id must be specified when pushing local model to hub")

    # 首先将所有组件保存到本地目录
    print(f"Saving preprocessor to {output_dir}...")
    preprocessor.save_pretrained(output_dir)

    print(f"Saving postprocessor to {output_dir}...")
    postprocessor.save_pretrained(output_dir)

    print(f"Saving model to {output_dir}...")
    policy.save_pretrained(output_dir)

    # 生成并保存模型卡
    print("Generating model card...")
    # 从原始配置获取元数据
    dataset_repo_id = train_config.get("repo_id", "unknown")
    license = config.get("license", "apache-2.0")

    tags = config.get("tags", ["robotics", "lerobot", policy_type]) or ["robotics", "lerobot", policy_type]
    tags = set(tags).union({"robotics", "lerobot", policy_type})
    tags = list(tags)

    # 生成模型卡
    card = policy.generate_model_card(
        dataset_repo_id=dataset_repo_id, model_type=policy_type, license=license, tags=tags
    )

    # 在本地保存模型卡
    card.save(str(output_dir / "README.md"))
    print(f"Model card saved to {output_dir / 'README.md'}")
    # 如果需要，在单个操作中将所有文件推送到 hub
    if args.push_to_hub and hub_repo_id:
        api = HfApi()

        # 确定是否应该创建 PR（如果指定了分支，则自动创建）
        create_pr = args.branch is not None
        target_location = f"branch '{args.branch}'" if args.branch else "main branch"

        print(f"Pushing all migrated files to {hub_repo_id} on {target_location}...")

        # 在单个提交中上传所有文件，如果指定了分支，则自动创建 PR
        commit_message = "Migrate policy to PolicyProcessorPipeline system"
        commit_description = None

        if create_pr:
            # 为 PR 主体单独设置提交描述
            commit_description = """🤖 **自动化策略迁移到 PolicyProcessorPipeline**

此 PR 使用现代 PolicyProcessorPipeline 架构将您的模型迁移到新的 LeRobot 策略格式。

## 变更内容

### ✨ **新架构 - PolicyProcessorPipeline**
您的模型现在使用外部 PolicyProcessorPipeline 组件进行数据处理，而不是内置的归一化层。这提供了：
- **模块化**：独立的预处理和后处理管道
- **灵活性**：易于交换、配置和调试处理步骤
- **兼容性**：与最新的 LeRobot 生态系统兼容

### 🔧 **归一化提取**
我们从模型的 state_dict 中提取了归一化统计信息，并移除了内置的归一化层：
- **提取的模式**：`normalize_inputs.*`、`unnormalize_outputs.*`、`normalize.*`、`unnormalize.*`、`input_normalizer.*`、`output_normalizer.*`
- **保留的统计信息**：所有特征的均值、标准差、最小值、最大值
- **干净的模型**：state dict 现在仅包含核心模型权重

### 📦 **添加的文件**
- **preprocessor_config.json**：输入预处理管道的配置
- **postprocessor_config.json**：输出后处理管道的配置
- **model.safetensors**：不含归一化层的干净模型权重
- **config.json**：更新的模型配置
- **train_config.json**：训练配置
- **README.md**：包含迁移信息的更新模型卡

### 🚀 **优势**
- **向后兼容**：您的模型行为保持不变
- **面向未来**：与最新的 LeRobot 功能和更新兼容
- **可调试**：易于检查和修改处理步骤
- **可移植**：处理器可以跨模型共享和重用

### 💻 **使用方法**
```python
# 加载迁移后的模型
from lerobot.policies import get_policy_class
from lerobot.processor import PolicyProcessorPipeline

# 预处理器和后处理器现在是外部的
preprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="preprocessor_config.json")
postprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="postprocessor_config.json")
policy = get_policy_class("your-policy-type").from_pretrained("your-model-repo")

# 通过管道处理数据
processed_batch = preprocessor(raw_batch)
action = policy(processed_batch)
final_action = postprocessor(action)
```

*由 LeRobot 策略迁移脚本自动生成*"""

        upload_kwargs = {
            "repo_id": hub_repo_id,
            "folder_path": output_dir,
            "repo_type": "model",
            "commit_message": commit_message,
            "revision": args.branch,
            "create_pr": create_pr,
            "allow_patterns": ["*.json", "*.safetensors", "*.md"],
            "ignore_patterns": ["*.tmp", "*.log"],
        }

        # Add commit_description for PR body if creating PR
        if create_pr and commit_description:
            upload_kwargs["commit_description"] = commit_description

        api.upload_folder(**upload_kwargs)

        if create_pr:
            print("All files pushed and pull request created successfully!")
        else:
            print("All files pushed to main branch successfully!")

    print("\nMigration complete!")
    print(f"Migrated model saved to: {output_dir}")
    if args.push_to_hub and hub_repo_id:
        if args.branch:
            print(
                f"Successfully pushed all files to branch '{args.branch}' and created PR on https://huggingface.co/{hub_repo_id}"
            )
        else:
            print(f"Successfully pushed to https://huggingface.co/{hub_repo_id}")
        if args.branch:
            print(f"\nView the branch at: https://huggingface.co/{hub_repo_id}/tree/{args.branch}")
            print(
                f"View the PR at: https://huggingface.co/{hub_repo_id}/discussions (look for the most recent PR)"
            )
        else:
            print(f"\nView the changes at: https://huggingface.co/{hub_repo_id}")


if __name__ == "__main__":
    main()
