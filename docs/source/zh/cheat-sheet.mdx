# 速查表

所有 LeRobot 命令集中在一个地方。如果您忘记了如何使用特定命令或想了解新命令，可以在这里查看。

> [!WARNING]
> 对于下面列出的所有命令，请记住将端口/名称/ID 更改为您自己的值！

> [!TIP]
> 查看所有命令并为您的特定设置配置它们的另一个好方法是使用这个 [Jupyter Notebook](https://github.com/huggingface/lerobot/blob/main/examples/notebooks/quickstart.ipynb)。

### 设置和安装

有关安装，请查看 [LeRobot 安装](https://huggingface.co/docs/lerobot/main/en/installation)。

### 实用工具

###### 查找端口

使用此命令识别您的机器人连接到哪些串口。按照终端中的说明操作：系统会要求您拔下 USB 线并按 Enter。然后脚本将检测并打印该机器人的正确串口。

```bash
lerobot-find-port
```

###### 查找摄像头

快速查找摄像头索引并验证其输出。此命令将摄像头信息打印到终端，并将每个检测到的摄像头的测试帧保存到 `lerobot/outputs/captured_images`

```bash
lerobot-find-cameras
```

### 校准

在大多数情况下，您只需要为每个机器人和遥操作设备执行一次校准。在执行校准之前，请确保所有关节大致处于中间位置。

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
```

确保在校准期间使用的 ID 与其他脚本中使用的 ID 相同。这就是 LeRobot 查找校准文件的方式。

### 遥操作

使用两个摄像头进行遥操作并使用 Rerun 显示数据。

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --display_data=true
```

### 录制数据集

数据集会自动上传到服务器并保存在 repo_id 下，请确保您已使用 CLI 登录到您的 HF 账户：
`hf auth login`

您可以从这里获取令牌：[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --dataset.num_episodes=30 \
    --dataset.single_task="把红色积木放进碗里" \
    --dataset.streaming_encoding=true \
    --display_data=true
```

在收集数据集时，您可以使用键盘控制过程：
使用键盘快捷键控制数据录制流程：

- 按**右箭头 (`→`)**：保存回合并移至下一个。
- 按**左箭头 (`←`)**：删除当前回合并重试。
- 按**Escape (`ESC`)**：停止、编码视频并上传。

### 训练

根据您的硬件，训练策略可能需要几个小时。以下是训练简单 `ACT` 策略的方法：

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test \
    --job_name=act_so101_test \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/policy_test \
    --steps=20000
```

- 策略类型：`act`、`diffusion`、`smolvla`、`pi05`
- 设备：`cuda`（NVIDIA）、`mps`（Apple Silicon）、`cpu`

如果您想微调特定模型，可以提供模型的路径。在这种情况下，路径就足够了，可以跳过类型。

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --policy.path=username/the_policy_to_finetune \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/policy_test \
    --output_dir=outputs/train/act_so101_test \
    --steps=20000
```

### 推理

推理意味着在机器人上运行训练好的策略/模型。为此我们使用 `lerobot-rollout`。您需要提供策略的路径。它可以是本地路径或 Hugging Face 的路径，例如 "lerobot/folding_latest"。您的摄像头配置需要与收集数据集时使用的配置匹配。持续时间以秒为单位，如果未指定，它将永远运行。

> [!TIP]
> 如果您使用的是之前的版本 V0.5.1，则需要使用 `lerobot-record` 而不是 `lerobot-rollout`。更多信息[在这里](https://huggingface.co/docs/lerobot/v0.5.1/en/il_robots#run-inference-and-evaluate-your-policy)。

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video1, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video5, width: 640, height: 480, fps: 30}}" \
    --task="把乐高积木放进透明盒子里" \
    --duration=60
```
