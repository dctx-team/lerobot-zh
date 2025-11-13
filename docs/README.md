<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 生成文档

要生成文档，你首先需要构建它。构建文档需要几个依赖包，你可以在代码仓库的根目录使用以下命令安装它们：

```bash
pip install -e . -r docs-requirements.txt
```

你还需要安装 `nodejs`。请参考其 [安装页面](https://nodejs.org/en/download)

---

**注意**

只有在你需要本地查看文档时才需要生成文档（例如，如果你正在计划进行更改，并希望在提交前查看它们的效果）。你不需要 `git commit` 生成的文档。

---

## 构建文档

设置好 `doc-builder` 和其他依赖包后，你可以使用以下命令生成文档：

```bash
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
```

你可以调整 `--build_dir` 来设置你喜欢的任何临时文件夹。这个命令将创建该文件夹，并生成将在主网站上呈现为文档的 MDX 文件。你可以在你喜欢的 Markdown 编辑器中查看它们。

## 预览文档

要预览文档，首先使用以下命令安装 `watchdog` 模块：

```bash
pip install watchdog
```

然后运行以下命令：

```bash
doc-builder preview lerobot docs/source/
```

文档将在 [http://localhost:3000](http://localhost:3000) 上可见。在你打开 PR 后，你也可以预览文档。你会看到一个机器人添加一个评论，其中包含一个链接，指向包含你更改的文档预览页面。

---

**注意**

`preview` 命令仅适用于现有的文档文件。当你添加一个全新的文件时，你需要更新 `_toctree.yml` 并重新启动 `preview` 命令（使用 `ctrl-c` 停止它，然后再次调用 `doc-builder preview ...`）。

---

## 向导航栏添加新元素

接受的文件格式是 Markdown (.md) 或 MDX (.mdx)。

创建一个带有扩展名的文件，并将其放在 source 目录中。然后，你可以通过在 [`_toctree.yml`](https://github.com/huggingface/lerobot/blob/main/docs/source/_toctree.yml) 文件中添加不带扩展名的文件名，将其链接到目录树。

## 重命名节标题和移动节

在重命名节标题和/或将节从一个文档移动到另一个文档时，保持旧链接有效很有帮助。这是因为旧链接可能会在 Issues、论坛和社交媒体中使用，如果几个月后阅读这些内容的用户仍然可以轻松导航到最初预期的信息，这将带来更好的用户体验。

因此，我们只需在原始节所在的文档末尾保留一个小的已移动节映射。关键是保留原始锚点。

所以，如果你将一个节从 "Section A" 重命名为 "Section B"，那么你可以在文件末尾添加：

```
已移动的节：

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```

当然，如果你将其移动到另一个文件，那么：

```
已移动的节：

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

使用相对样式链接到新文件，以便版本化的文档继续工作。

有关丰富的已移动节集的示例，请参见 [transformers Trainer 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md) 的最末尾。

### 添加新教程

添加新教程或章节分两步完成：

- 在 `./source` 下添加新文件。该文件可以是 ReStructuredText (.rst)、Markdown (.md) 或 MDX (.mdx)。
- 在 `./source/_toctree.yml` 中的正确目录树上链接该文件。

确保将新文件放在适当的部分下。如果你有疑问，请随时在 Github Issue 或 PR 中询问。

### 编写源文档

应该放在 `code` 中的值应该用反引号括起来：\`like so\`。注意，参数名称和对象（如 True、None 或任何字符串）通常应该放在 `code` 中。

#### 编写多行代码块

多行代码块对于显示示例很有用。它们在 Markdown 中像往常一样在两行三个反引号之间完成：

````
```
# 第一行代码
# 第二行
# 等等
```
````

#### 添加图片

由于仓库快速增长，确保不添加会显著增加仓库大小的文件非常重要。这包括图片、视频和其他非文本文件。我们更喜欢利用托管在 hf.co 上的 `dataset`，比如托管在 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) 上的那些，将这些文件放在其中并通过 URL 引用它们。我们建议将它们放在以下数据集中：[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。

如果是外部贡献，请随意将图片添加到你的 PR 中，并要求 Hugging Face 成员将你的图片迁移到此数据集。

---

## 📚 文档结构

本项目的文档完全按照官方 LeRobot 文档结构组织，包含以下主要分类：

### 文档分类

| 分类 | 文档数量 | 主要内容 |
|------|---------|----------|
| 入门指南 | 2 | 介绍、安装 |
| 教程 | 7 | 实践导向教程 |
| 数据集 | 3 | 数据管理 |
| 策略模型 | 5 | 学习模型 |
| 仿真环境 | 4 | 仿真工具 |
| 机器人处理器 | 4 | 数据处理 |
| 机器人平台 | 6 | 硬件平台 |
| 遥操作器 | 1 | 遥操作设备 |
| 资源 | 2 | 额外资源 |
| 关于 | 1 | 项目信息 |
| **总计** | **35** | **完整覆盖** |

### 文档文件列表

所有文档位于 `source/` 目录：

**入门指南**
- `index.mdx` - LeRobot
- `installation.mdx` - 安装

**教程**
- `il_robots.mdx` - 机器人模仿学习
- `cameras.mdx` - 相机
- `integrate_hardware.mdx` - 集成你的硬件
- `hilserl.mdx` - 使用强化学习训练机器人
- `hilserl_sim.mdx` - 仿真中的强化学习训练
- `async.mdx` - 使用异步推理
- `multi_gpu_training.mdx` - 多GPU训练

**数据集**
- `lerobot-dataset-v3.mdx` - 使用 LeRobotDataset
- `porting_datasets_v3.mdx` - 迁移大型数据集
- `using_dataset_tools.mdx` - 使用数据集工具

**策略模型**
- `act.mdx` - ACT
- `smolvla.mdx` - SmolVLA
- `pi0.mdx` - π₀
- `pi05.mdx` - π₀.₅
- `groot.mdx` - NVIDIA GR00T

**仿真环境**
- `envhub.mdx` - Hub环境
- `il_sim.mdx` - 仿真中的模仿学习
- `libero.mdx` - 使用 LIBERO
- `metaworld.mdx` - 使用 MetaWorld

**机器人处理器**
- `introduction_processors.mdx` - 机器人处理器简介
- `debug_processor_pipeline.mdx` - 调试处理器管道
- `implement_your_own_processor.mdx` - 实现自定义处理器
- `processors_robots_teleop.mdx` - 机器人和遥操作器的处理器

**机器人平台**
- `so101.mdx` - SO-101
- `so100.mdx` - SO-100
- `koch.mdx` - Koch v1.1
- `lekiwi.mdx` - LeKiwi
- `hope_jr.mdx` - Hope Jr
- `reachy2.mdx` - Reachy 2

**遥操作器**
- `phone_teleop.mdx` - 手机

**资源**
- `notebooks.mdx` - Notebooks
- `feetech.mdx` - 更新 Feetech 固件

**关于**
- `backwardcomp.mdx` - 向后兼容性

---

## 🔄 与官方文档对应

本项目的文档文件名与官方文档完全一致，便于对照学习：

| 本项目文件 | 官方文档 URL |
|-----------|-------------|
| `index.mdx` | https://huggingface.co/docs/lerobot/index |
| `installation.mdx` | https://huggingface.co/docs/lerobot/installation |
| `act.mdx` | https://huggingface.co/docs/lerobot/act |
| ... | ... |

只需将文件名（不含扩展名）添加到官方文档 URL 即可访问对应的英文版本。

---

**更新日期**: 2025-11-12
