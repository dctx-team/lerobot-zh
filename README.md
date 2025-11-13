<p align="center">
  <img alt="LeRobot 中文版 - Hugging Face 机器人学习库" src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lerobot-logo-thumbnail.png" width="100%">
  <br/>
  <br/>
</p>

<div align="center">

[![原项目](https://img.shields.io/badge/Original-LeRobot-orange)](https://github.com/huggingface/lerobot)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![中文文档](https://img.shields.io/badge/Docs-中文-green)](docs/source/index.mdx)
[![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)

</div>

<h2 align="center">
    <p>🇨🇳 LeRobot 中文版</p>
</h2>

---

## 📖 关于本项目

这是 **[🤗 LeRobot](https://github.com/huggingface/lerobot)** 的**完整中文版项目**，包含：

- ✅ **完整源代码** - 与官方项目完全相同的代码（233个Python文件）
- ✅ **所有功能** - 训练、评估、数据处理等所有功能
- ✅ **中文文档** - 35篇完整翻译的技术文档（docs/source/）
- ✅ **保持同步** - 定期与官方项目同步更新

**唯一区别**：文档是中文的，代码和功能与官方完全一致！

---

<h2 align="center">
    <p><a href="docs/source/hope_jr.mdx">
        构建你自己的 HopeJR 机器人！</a></p>
</h2>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/hope_jr/hopejr.png"
    alt="HopeJR 机器人"
    title="HopeJR 机器人"
    width="60%"
  />

  <p><strong>认识 HopeJR – 用于灵巧操作的人形机器人手臂和手！</strong></p>
  <p>使用外骨骼和手套进行精确的手部运动控制。</p>
  <p>非常适合高级操作任务！🤖</p>

  <p><a href="docs/source/hope_jr.mdx">
      查看完整的 HopeJR 教程。</a></p>
</div>

<br/>

<h2 align="center">
    <p><a href="docs/source/so101.mdx">
        构建你自己的 SO-101 机器人！</a></p>
</h2>

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/so101/so101.webp" alt="SO-101 follower arm" title="SO-101 follower arm" width="90%"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/so101/so101-leader.webp" alt="SO-101 leader arm" title="SO-101 leader arm" width="90%"/></td>
    </tr>
  </table>

  <p><strong>认识升级版的 SO100，即 SO-101 – 每个手臂仅需 €114！</strong></p>
  <p>在你的笔记本电脑上只需几分钟就能训练它。</p>
  <p>然后坐下来看着你的创作自主行动！🤯</p>

  <p><a href="docs/source/so101.mdx">
      查看完整的 SO-101 教程。</a></p>

  <p>想要更进一步？通过构建 LeKiwi 让你的 SO-101 移动起来！</p>
  <p>查看 <a href="docs/source/lekiwi.mdx">LeKiwi 教程</a>，让你的机器人在轮子上动起来。</p>

  <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lekiwi/kiwi.webp" alt="LeKiwi 移动机器人" title="LeKiwi 移动机器人" width="50%">
</div>

<br/>

<h3 align="center">
    <p>LeRobot: 用于真实世界机器人的最先进 AI</p>
</h3>

---

🤗 LeRobot 旨在为 PyTorch 中的真实世界机器人学习提供模型、数据集和工具。目标是降低机器人技术的准入门槛，让每个人都能贡献并受益于共享数据集和预训练模型。

🤗 LeRobot 包含已被证明可迁移到真实世界的最先进方法，重点关注模仿学习和强化学习。

🤗 LeRobot 已经提供了一系列预训练模型、人类收集的演示数据集以及仿真环境，让你无需组装机器人即可开始使用。

🤗 LeRobot 在 Hugging Face 社区页面托管预训练模型和数据集：[huggingface.co/lerobot](https://huggingface.co/lerobot)

### 仿真环境中预训练模型的示例

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/aloha_act.gif" width="100%" alt="ALOHA 环境上的 ACT 策略"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/simxarm_tdmpc.gif" width="100%" alt="SimXArm 环境上的 TDMPC 策略"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/pusht_diffusion.gif" width="100%" alt="PushT 环境上的 Diffusion 策略"/></td>
  </tr>
  <tr>
    <td align="center">ALOHA 环境上的 ACT 策略</td>
    <td align="center">SimXArm 环境上的 TDMPC 策略</td>
    <td align="center">PushT 环境上的 Diffusion 策略</td>
  </tr>
</table>

---

## 📚 中文文档

**完整的中文技术文档位于 `docs/source/` 目录！**

- 📖 **[开始阅读](docs/source/index.mdx)** - LeRobot 介绍
- 🔧 **[安装指南](docs/source/installation.mdx)** - 环境配置
- 📊 **[数据集使用](docs/source/lerobot-dataset-v3.mdx)** - 数据处理
- 🤖 **[模型训练](docs/source/il_robots.mdx)** - 训练机器人
- 📝 **[查看所有文档](docs/source/_toctree.yml)** - 完整目录

**文档构建**：
```bash
# 预览中文文档
pip install -r docs-requirements.txt
doc-builder preview lerobot docs/source/
# 访问 http://localhost:3000
```

详见 [docs/README.md](docs/README.md) 获取完整的文档构建指南。

---

## 安装

LeRobot 支持 Python 3.10+ 和 PyTorch 2.2+。

### 环境设置

使用 [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html) 创建 Python 3.10 虚拟环境并激活：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

使用 `miniconda` 时，在环境中安装 `ffmpeg`：

```bash
conda install ffmpeg -c conda-forge
```

> **注意**：这通常会为你的平台安装使用 `libsvtav1` 编码器编译的 `ffmpeg 7.X`。如果不支持 `libsvtav1`（使用 `ffmpeg -encoders` 检查支持的编码器），你可以：
>
> - _[任何平台]_ 使用以下命令显式安装 `ffmpeg 7.X`：
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[仅 Linux]_ 安装 [ffmpeg 构建依赖](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies)并[使用 libsvtav1 从源代码编译 ffmpeg](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1)，并确保使用与你的安装对应的 ffmpeg 二进制文件（使用 `which ffmpeg`）。

### 安装 LeRobot 🤗

#### 从源代码安装（推荐）

首先，克隆本仓库并进入目录：

```bash
git clone https://github.com/dctx-team/lerobot-zh.git
cd lerobot-zh
```

然后，以可编辑模式安装库。如果你计划为代码做贡献，这会很有用。

```bash
pip install -e .
```

> **注意**：如果遇到构建错误，你可能需要安装额外的依赖项（`cmake`、`build-essential` 和 `ffmpeg libs`）。在 Linux 上，运行：
> `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev`。对于其他系统，参见：[编译 PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

对于仿真，🤗 LeRobot 附带可以作为额外组件安装的 gymnasium 环境：

- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

例如，要安装带有 aloha 和 pusht 的 🤗 LeRobot，使用：

```bash
pip install -e ".[aloha, pusht]"
```

#### 从 PyPI 安装

**核心库：**
使用以下命令安装基础包：

```bash
pip install lerobot
```

_这只安装默认依赖项。_

**额外功能：**
要安装额外功能，使用以下之一：

```bash
pip install 'lerobot[all]'          # 所有可用功能
pip install 'lerobot[aloha,pusht]'  # 特定功能（Aloha 和 Pusht）
pip install 'lerobot[feetech]'      # Feetech 电机支持
```

_用你想要的功能替换 `[...]`。_

**可用标签：**
有关可选依赖项的完整列表，请参见：
https://pypi.org/project/lerobot/

### Weights & Biases

要使用 [Weights and Biases](https://docs.wandb.ai/quickstart) 进行实验跟踪，使用以下命令登录：

```bash
wandb login
```

（注意：你还需要在配置中启用 WandB。见下文。）

### 可视化数据集

查看[示例 1](examples/1_load_lerobot_dataset.py)，它说明了如何使用我们的数据集类，该类会自动从 Hugging Face hub 下载数据。

你还可以通过从命令行执行我们的脚本来本地可视化 hub 上数据集的情节：

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

或使用 `root` 选项和 `--local-files-only` 从本地文件夹中的数据集（在以下情况下，数据集将在 `./my_local_data_dir/lerobot/pusht` 中搜索）：

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```

它将打开 `rerun.io` 并显示相机流、机器人状态和动作，如下所示：

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144

我们的脚本还可以可视化存储在远程服务器上的数据集。有关更多说明，请参见 `lerobot-dataset-viz --help`。

### LeRobotDataset 格式

`LeRobotDataset` 格式的数据集使用非常简单。可以简单地通过例如 `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` 从 Hugging Face hub 上的仓库或本地文件夹加载它，并且可以像任何 Hugging Face 和 PyTorch 数据集一样进行索引。例如，`dataset[0]` 将从数据集中检索包含观测和动作的单个时间帧，作为准备好馈送到模型的 PyTorch 张量。

`LeRobotDataset` 的一个特殊性是，与通过索引检索单个帧不同，我们可以通过将 `delta_timestamps` 设置为相对于索引帧的相对时间列表来基于它们与索引帧的时间关系检索多个帧。例如，使用 `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`，对于给定的索引，可以检索 4 个帧：3 个"先前"帧分别在索引帧之前 1 秒、0.5 秒和 0.2 秒，以及索引帧本身（对应于 0 条目）。有关 `delta_timestamps` 的更多详细信息，请参见示例 [1_load_lerobot_dataset.py](examples/dataset/load_lerobot_dataset.py)。

在底层，`LeRobotDataset` 格式使用多种方式来序列化数据，如果你计划更密切地使用此格式，了解这一点可能会很有用。我们试图创建一个灵活但简单的数据集格式，以涵盖强化学习和机器人技术中存在的大多数类型的特征和特殊性，在仿真和真实世界中，重点关注相机和机器人状态，但很容易扩展到其他类型的感官输入，只要它们可以用张量表示。

以下是使用 `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` 实例化的典型 `LeRobotDataset` 的重要细节和内部结构组织。确切的特征会因数据集而异，但主要方面不会：

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of an episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ meta: a LeRobotDatasetMetadata object containing:
  │  ├ info: a dictionary of metadata on the dataset
  │  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  │  ├ fps (int): frame per second the dataset is recorded/synchronized to
  │  │  ├ features (dict): all features contained in the dataset with their shapes and types
  │  │  ├ total_episodes (int): total number of episodes in the dataset
  │  │  ├ total_frames (int): total number of frames in the dataset
  │  │  ├ robot_type (str): robot type used for recording
  │  │  ├ data_path (str): formattable string for the parquet files
  │  │  └ video_path (str): formattable string for the video files (if using videos)
  │  ├ episodes: a DataFrame containing episode metadata with columns:
  │  │  ├ episode_index (int): index of the episode
  │  │  ├ tasks (list): list of tasks for this episode
  │  │  ├ length (int): number of frames in this episode
  │  │  ├ dataset_from_index (int): start index of this episode in the dataset
  │  │  └ dataset_to_index (int): end index of this episode in the dataset
  │  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  │  ├ observation.images.front_cam: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  │  └ ...
  │  └ tasks: a DataFrame containing task information with task names as index and task_index as values
  ├ root (Path): local directory where the dataset is stored
  ├ image_transforms (Callable): optional image transformations to apply to visual modalities
  └ delta_timestamps (dict): optional delta timestamps for temporal queries
```

`LeRobotDataset` 使用几种广泛使用的文件格式序列化其各个部分，即：

- hf_dataset 使用 Hugging Face datasets 库序列化为 parquet
- 视频以 mp4 格式存储以节省空间
- 元数据以纯 json/jsonl 文件存储

数据集可以无缝地从 HuggingFace hub 上传/下载。要处理本地数据集，如果它不在默认的 `~/.cache/huggingface/lerobot` 位置，你可以使用 `root` 参数指定其位置。

### 复现最先进（SOTA）结果

我们在[hub 页面](https://huggingface.co/lerobot)上提供了一些可以实现最先进性能的预训练策略。
你可以通过从运行中加载配置来复现它们的训练。只需运行：

```bash
lerobot-train --config_path=lerobot/diffusion_pusht
```

即可复现 PushT 任务上 Diffusion Policy 的 SOTA 结果。

---

## 贡献

如果你想为 🤗 LeRobot 做贡献，请查看我们的[贡献指南](CONTRIBUTING.md)。

### 添加预训练策略

一旦你训练了一个策略，你可以使用看起来像 `${hf_user}/${repo_name}` 的 hub id（例如 [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)）将其上传到 Hugging Face hub。

你首先需要找到位于实验目录内的 checkpoint 文件夹（例如 `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`）。在其中有一个 `pretrained_model` 目录，应该包含：

- `config.json`：策略配置的序列化版本（遵循策略的 dataclass 配置）。
- `model.safetensors`：一组 `torch.nn.Module` 参数，以 [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) 格式保存。
- `train_config.json`：包含所有用于训练的参数的合并配置。策略配置应完全匹配 `config.json`。这对于希望评估你的策略或为了可重现性的任何人都很有用。

要将这些上传到 hub，运行以下命令：

```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

有关其他人如何使用你的策略的示例，请参见 [eval.py](src/lerobot/scripts/eval.py)。

---

## 致谢

### 原项目团队
- Hugging Face LeRobot 团队 🤗 构建了 SmolVLA [论文](https://arxiv.org/abs/2506.01844)，[博客](https://huggingface.co/blog/smolvla)。
- 感谢 Tony Zhao、Zipeng Fu 及其同事开源 ACT 策略、ALOHA 环境和数据集。我们的版本改编自 [ALOHA](https://tonyzhaozh.github.io/aloha) 和 [Mobile ALOHA](https://mobile-aloha.github.io)。
- 感谢 Cheng Chi、Zhenjia Xu 及其同事开源 Diffusion 策略、Pusht 环境和数据集，以及 UMI 数据集。我们的版本改编自 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu) 和 [UMI Gripper](https://umi-gripper.github.io)。
- 感谢 Nicklas Hansen、Yunhai Feng 及其同事开源 TDMPC 策略、Simxarm 环境和数据集。我们的版本改编自 [TDMPC](https://github.com/nicklashansen/tdmpc) 和 [FOWM](https://www.yunhaifeng.com/FOWM)。
- 感谢 Antonio Loquercio 和 Ashish Kumar 的早期支持。
- 感谢 [Seungjae (Jay) Lee](https://sjlee.cc/)、[Mahi Shafiullah](https://mahis.life/) 及其同事开源 [VQ-BeT](https://sjlee.cc/vq-bet/) 策略并帮助我们将代码库适配到我们的仓库。该策略改编自 [VQ-BeT 仓库](https://github.com/jayLEE0301/vq_bet_official)。

### 中文翻译
- 感谢所有参与中文文档翻译的贡献者
- 感谢中文机器人学习社区的支持

---

## 引用

如果你愿意，可以引用此工作：

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

---

## 相关链接

- **官方项目**: https://github.com/huggingface/lerobot
- **官方文档**: https://huggingface.co/docs/lerobot
- **Hugging Face Hub**: https://huggingface.co/lerobot
- **Discord 社区**: https://discord.gg/s3KuuzsPFb

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/lerobot&type=Timeline)](https://star-history.com/#huggingface/lerobot&Timeline)

---

<div align="center">

**为中文机器人学习社区用心打造 ❤️**

</div>
