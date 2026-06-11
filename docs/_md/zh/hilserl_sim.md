# 在模拟中训练 RL

本指南解释了如何使用 `gym_hil` 模拟环境作为使用 LeRobot 框架进行人在环路（HIL）强化学习时真实机器人的替代方案。

`gym_hil` 是一个提供 Gymnasium 兼容模拟环境的包，专门为人在环路强化学习设计。这些环境允许你：

- 在模拟中训练策略，在真实机器人上训练之前测试 RL 堆栈

- 使用游戏手柄或键盘等外部设备在模拟中收集演示
- 在策略学习期间执行人工干预

目前，主要环境是基于 MuJoCo 的 Franka Panda 机器人模拟，任务如拾取立方体。

## 安装

首先，在 LeRobot 环境中安装 `gym_hil` 包：

```bash
pip install -e ".[hilserl]"
```

## 我需要什么？

- 游戏手柄或键盘来控制机器人
- Nvidia GPU

## 配置

要将 `gym_hil` 与 LeRobot 一起使用，你需要创建一个配置文件。[此处](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json)提供了一个示例。关键配置部分包括：

### 环境类型和任务

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0",
    "fps": 10
  },
  "device": "cuda"
}
```

可用任务：

- `PandaPickCubeBase-v0`：基础环境
- `PandaPickCubeGamepad-v0`：带游戏手柄控制
- `PandaPickCubeKeyboard-v0`：带键盘控制

### 处理器配置

```json
{
  "env": {
    "processor": {
      "control_mode": "gamepad",
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": -0.02
      },
      "reset": {
        "control_time_s": 15.0,
        "fixed_reset_joint_positions": [
          0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785
        ]
      },
      "inverse_kinematics": {
        "end_effector_step_sizes": {
          "x": 0.025,
          "y": 0.025,
          "z": 0.025
        }
      }
    }
  }
}
```

重要参数：

- `gripper.gripper_penalty`：过度夹爪移动的惩罚
- `gripper.use_gripper`：是否启用夹爪控制
- `inverse_kinematics.end_effector_step_sizes`：末端执行器 x、y、z 轴的步长大小
- `control_mode`：设置为 `"gamepad"` 以使用游戏手柄控制器

## 使用 LeRobot 的 HIL RL 运行

### 基本使用

要运行环境，将模式设置为 null：

```bash
python -m lerobot.rl.gym_manipulator --config_path path/to/gym_hil_env.json
```

### 记录数据集

要收集数据集，将模式设置为 `record`，同时定义 repo_id 和要记录的片段数量：

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0"
  },
  "dataset": {
    "repo_id": "username/sim_dataset",
    "root": null,
    "task": "pick_cube",
    "num_episodes_to_record": 10,
    "replay_episode": null,
    "push_to_hub": true
  },
  "mode": "record"
}
```

```bash
python -m lerobot.rl.gym_manipulator --config_path path/to/gym_hil_env.json
```

### 训练策略

要训练策略，请查看[此处](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/train_config.json)提供的配置示例，并运行 actor 和 learner 服务器：

```bash
python -m lerobot.rl.actor --config_path path/to/train_gym_hil_env.json
```

在不同的终端中，运行 learner 服务器：

```bash
python -m lerobot.rl.learner --config_path path/to/train_gym_hil_env.json
```

模拟环境提供了一种安全且可重复的方式来开发和测试你的人在环路强化学习组件，然后再部署到真实机器人。

恭喜 🎉，你已经完成了本教程！

> [!TIP]
> 如果你有任何问题或需要帮助，请在 [Discord](https://discord.com/invite/s3KuuzsPFb) 上联系我们。

论文引用：

```
@article{luo2024precise,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},
  journal={arXiv preprint arXiv:2410.21845},
  year={2024}
}
```
