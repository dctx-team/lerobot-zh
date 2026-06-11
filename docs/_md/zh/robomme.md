# RoboMME

[RoboMME](https://robomme.github.io) 是一个基于 ManiSkill（SAPIEN）构建的记忆增强操作基准测试。它评估机器人在回合中保留和使用信息的能力 — 计数、物体永久性、引用和模仿。

- **16 个任务**跨 4 个记忆技能套件
- **1,600 个训练演示**（每个任务 100 个，50 个验证，50 个测试）
- **数据集**：[`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) — LeRobot v3.0，10 fps 下 768K 帧
- **模拟器**：ManiSkill / SAPIEN，Panda 机械臂，仅限 Linux

![RoboMME 基准测试任务概览](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2603.04639/gradient.png)

## 任务

| 套件                       | 任务                                                          |
| -------------------------- | ------------------------------------------------------------- |
| **Counting**（时间记忆）   | BinFill、PickXtimes、SwingXtimes、StopCube                    |
| **Permanence**（空间记忆） | VideoUnmask、VideoUnmaskSwap、ButtonUnmask、ButtonUnmaskSwap  |
| **Reference**（物体记忆）  | PickHighlight、VideoRepick、VideoPlaceButton、VideoPlaceOrder |
| **Imitation**（程序记忆）  | MoveCube、InsertPeg、PatternLock、RouteStick                  |

## 安装

> RoboMME 需要 **Linux**（ManiSkill/SAPIEN 使用 Vulkan 渲染）。推荐使用 Docker 来隔离依赖冲突。

### 原生（Linux）

```bash
pip install --override <(printf 'gymnasium==0.29.1\nnumpy==1.26.4\n') \
  -e '.[smolvla,av-dep]' \
  'robomme @ git+https://github.com/RoboMME/robomme_benchmark.git@main'
```

> **依赖说明**：`mani-skill`（由 `robomme` 拉取）固定 `gymnasium==0.29.1` 和 `numpy<2.0.0`，这与 lerobot 的基础 `numpy>=2.0.0` 冲突。这就是为什么 `robomme` 不是 pyproject extra — 使用上面的覆盖安装，或使用下面的 Docker 方法完全避免冲突。

### Docker（推荐）

```bash
# 首先构建基础镜像（从仓库根目录）
docker build -f docker/Dockerfile.eval-base -t lerobot-eval-base .

# 构建 RoboMME 评估镜像（应用 gymnasium + numpy 固定覆盖）
docker build -f docker/Dockerfile.benchmark.robomme -t lerobot-robomme .
```

`docker/Dockerfile.benchmark.robomme` 镜像在 lerobot 安装后覆盖 `gymnasium==0.29.1` 和 `numpy==1.26.4`。两个版本对于 lerobot 的实际 API 使用都是运行时安全的。

## 运行评估

### 默认（单任务，单回合）

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes \
    --env.dataset_split=test \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=1
```

### 多任务评估

通过逗号分隔任务名称在一次运行中评估多个任务。使用 `task_ids` 控制每个任务评估哪些回合。推荐：测试分割每个任务 50 个回合。

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes,BinFill,StopCube,MoveCube,InsertPeg \
    --env.dataset_split=test \
    --env.task_ids=[0,1,2,3,4,5,6,7,8,9] \
    --eval.batch_size=1 \
    --eval.n_episodes=50
```

### `env.type=robomme` 的关键 CLI 选项

| 选项                 | 默认值        | 描述                                       |
| -------------------- | ------------- | ------------------------------------------ |
| `env.task`           | `PickXtimes`  | 上述 16 个任务名称中的任何一个（逗号分隔） |
| `env.dataset_split`  | `test`        | `train`、`val` 或 `test`                   |
| `env.action_space`   | `joint_angle` | `joint_angle`（8-D）或 `ee_pose`（7-D）    |
| `env.episode_length` | `300`         | 每个回合的最大步数                         |
| `env.task_ids`       | `null`        | 要评估的回合索引列表（null = `[0]`）       |

## 数据集

数据集 [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) 采用 **LeRobot v3.0 格式**，可以直接加载：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/robomme")
```

### 数据集特征

| 特征               | 形状          | 描述                |
| ------------------ | ------------- | ------------------- |
| `image`            | (256, 256, 3) | 前置相机 RGB        |
| `wrist_image`      | (256, 256, 3) | 腕部相机 RGB        |
| `actions`          | (8,)          | 关节角度 + 夹爪     |
| `state`            | (8,)          | 关节位置 + 夹爪状态 |
| `simple_subgoal`   | str           | 高级语言注释        |
| `grounded_subgoal` | str           | 接地语言注释        |
| `episode_index`    | int           | 回合 ID             |
| `frame_index`      | int           | 回合内的帧          |

### 特征键对齐（训练）

环境包装器将 `pixels/image` 和 `pixels/wrist_image` 作为观测键公开。`RoboMMEEnv` 中的 `features_map` 将这些映射到 `observation.images.image` 和 `observation.images.wrist_image` 供策略使用。状态作为 `agent_pos` 公开并映射到 `observation.state`。

数据集的 `image` 和 `wrist_image` 列已经与策略输入键对齐，因此在微调时不需要重命名。

## 动作空间

| 类型          | 维度 | 描述                                                |
| ------------- | ---- | --------------------------------------------------- |
| `joint_angle` | 8    | 7 个关节角度 + 1 个夹爪（−1 闭合，+1 打开，绝对值） |
| `ee_pose`     | 7    | xyz + roll/pitch/yaw + 夹爪                         |

通过 `--env.action_space=joint_angle`（默认）或 `--env.action_space=ee_pose` 设置。

## 平台说明

- **仅限 Linux**：ManiSkill 需要 SAPIEN/Vulkan。不支持 macOS 和 Windows。
- **推荐 GPU**：渲染可以使用 CPU 但速度慢；CUDA + Vulkan 提供全速。
- **gymnasium / numpy 冲突**：参见上面的安装说明。Docker 镜像自动处理此问题。
- **ManiSkill 分支**：`robomme` 依赖于特定的 ManiSkill 分支（`YinpeiDai/ManiSkill`），通过 `robomme` 包自动拉取。
