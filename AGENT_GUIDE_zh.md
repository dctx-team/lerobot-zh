# LeRobot 代理指南

本指南将引导您完成使用 LeRobot 训练机器人策略的完整流程。我们将使用 SO-101 机械臂作为示例。

## 目录

1. [设置您的机器人](#设置您的机器人)
2. [录制演示数据](#录制演示数据)
3. [可视化您的数据集](#可视化您的数据集)
4. [选择策略](#选择策略)
5. [训练您的策略](#训练您的策略)
6. [评估您的策略](#评估您的策略)
7. [故障排除](#故障排除)

## 设置您的机器人

### SO-101 机械臂

SO-101 是一款经济实惠的 6 自由度机械臂，非常适合学习机器人操作。

#### 硬件要求

- SO-101 机械臂
- USB 转串口适配器
- 电源（12V，推荐 5A 或更高）
- 摄像头（可选，用于视觉策略）

#### 软件设置

1. 安装 LeRobot：

```bash
pip install lerobot
```

2. 连接您的机器人：

```bash
# 查找您的串口设备
ls /dev/ttyUSB*

# 测试连接
python -m lerobot.scripts.control_robot \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides env.port=/dev/ttyUSB0
```

3. 校准您的机器人：

```bash
python -m lerobot.scripts.calibrate_robot \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides env.port=/dev/ttyUSB0
```

## 录制演示数据

录制高质量的演示数据是训练成功策略的关键。

### 基本录制

```bash
python -m lerobot.scripts.control_robot \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides env.port=/dev/ttyUSB0 \
  --fps 30 \
  --repo-id your-username/your-dataset-name \
  --tags tutorial so100 \
  --warmup-time-s 3 \
  --episode-time-s 30 \
  --reset-time-s 5 \
  --num-episodes 50
```

### 参数说明

- `--fps`: 录制频率（帧/秒）
- `--repo-id`: Hugging Face Hub 上的数据集 ID
- `--tags`: 数据集标签，便于组织
- `--warmup-time-s`: 开始录制前的预热时间
- `--episode-time-s`: 每个回合的持续时间
- `--reset-time-s`: 回合之间的重置时间
- `--num-episodes`: 要录制的回合数

### 录制技巧

1. **保持一致性**：尽量使每个演示的起始和结束状态相似
2. **平滑运动**：避免突然的抖动或快速移动
3. **多样性**：在合理范围内变化您的演示
4. **质量胜于数量**：50 个好的演示比 200 个差的演示更有价值

## 可视化您的数据集

在训练之前，检查您的数据集质量：

```bash
python -m lerobot.scripts.visualize_dataset \
  --repo-id your-username/your-dataset-name \
  --episode-index 0
```

这将显示：

- 机器人关节位置随时间的变化
- 摄像头图像（如果有）
- 动作命令

## 选择策略

LeRobot 支持多种策略。以下是主要选项：

### ACT (Action Chunking Transformer)

**最适合**：需要预测动作序列的任务

**优点**：

- 可以预测未来的动作
- 对时间依赖性任务效果好
- 相对稳定的训练

**缺点**：

- 需要更多内存
- 训练时间较长

**使用场景**：拾取和放置、装配任务

```bash
python -m lerobot.scripts.train \
  --policy act \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/act_policy \
  --num-epochs 1000
```

### Diffusion Policy

**最适合**：需要多模态行为的复杂任务

**优点**：

- 可以学习多模态分布
- 对复杂任务效果好
- 生成平滑的动作

**缺点**：

- 推理速度较慢
- 需要更多计算资源

**使用场景**：灵巧操作、接触丰富的任务

```bash
python -m lerobot.scripts.train \
  --policy diffusion \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/diffusion_policy \
  --num-epochs 1000
```

### TDMPC (Temporal Difference Model Predictive Control)

**最适合**：需要在线适应的任务

**优点**：

- 可以在线学习
- 样本效率高
- 可以处理稀疏奖励

**缺点**：

- 需要奖励函数
- 训练可能不稳定

**使用场景**：强化学习任务、需要在线适应的场景

```bash
python -m lerobot.scripts.train \
  --policy tdmpc \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/tdmpc_policy \
  --num-epochs 1000
```

## 训练您的策略

### 基本训练命令

```bash
python -m lerobot.scripts.train \
  --policy act \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/my_policy \
  --num-epochs 1000 \
  --batch-size 8 \
  --lr 1e-4 \
  --save-freq 100
```

### 重要的训练参数

- `--num-epochs`: 训练轮数
- `--batch-size`: 批次大小（根据 GPU 内存调整）
- `--lr`: 学习率
- `--save-freq`: 保存检查点的频率
- `--eval-freq`: 评估频率
- `--wandb-project`: Weights & Biases 项目名称（用于日志记录）

### 监控训练

使用 Weights & Biases 监控训练进度：

```bash
python -m lerobot.scripts.train \
  --policy act \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/my_policy \
  --wandb-project my-robot-project \
  --wandb-entity your-username
```

### 从检查点恢复

```bash
python -m lerobot.scripts.train \
  --policy act \
  --dataset-repo-id your-username/your-dataset-name \
  --output-dir outputs/my_policy \
  --resume outputs/my_policy/checkpoints/last.ckpt
```

## 评估您的策略

### 在真实机器人上评估

```bash
python -m lerobot.scripts.eval \
  --policy-path outputs/my_policy/checkpoints/best.ckpt \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides env.port=/dev/ttyUSB0 \
  --num-episodes 10
```

### 评估指标

LeRobot 会自动计算：

- **成功率**：成功完成任务的回合百分比
- **平均奖励**：每个回合的平均奖励
- **回合长度**：完成任务所需的平均步数

### 录制评估视频

```bash
python -m lerobot.scripts.eval \
  --policy-path outputs/my_policy/checkpoints/best.ckpt \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides env.port=/dev/ttyUSB0 \
  --num-episodes 10 \
  --record-video \
  --video-dir outputs/eval_videos
```

## 故障排除

### 常见问题

#### 1. 机器人连接失败

**症状**：无法连接到机器人

**解决方案**：

- 检查 USB 连接
- 验证串口设备路径（`ls /dev/ttyUSB*`）
- 确保您有正确的权限（`sudo usermod -a -G dialout $USER`）
- 重启机器人和计算机

#### 2. 训练损失不下降

**症状**：训练损失保持高位或不稳定

**解决方案**：

- 降低学习率
- 增加批次大小
- 检查数据集质量
- 尝试不同的策略
- 增加训练轮数

#### 3. 策略在评估时表现不佳

**症状**：训练损失低但真实机器人表现差

**解决方案**：

- 录制更多多样化的演示
- 检查训练和评估环境的一致性
- 调整动作空间归一化
- 使用数据增强
- 尝试不同的策略架构

#### 4. 内存不足错误

**症状**：训练时 GPU 内存不足

**解决方案**：

- 减小批次大小
- 使用梯度累积
- 减小模型大小
- 使用混合精度训练（`--precision 16`）

#### 5. 摄像头问题

**症状**：无法捕获摄像头图像

**解决方案**：

- 检查摄像头连接
- 验证摄像头索引（`ls /dev/video*`）
- 测试摄像头（`ffplay /dev/video0`）
- 调整摄像头分辨率和 FPS

### 获取帮助

如果您遇到问题：

1. 查看 [LeRobot 文档](https://github.com/huggingface/lerobot)
2. 搜索 [GitHub Issues](https://github.com/huggingface/lerobot/issues)
3. 在 [Discord 社区](https://discord.gg/s3KuuzsPFb)提问
4. 提交新的 GitHub Issue

## 下一步

现在您已经了解了基础知识，可以：

1. **尝试不同的任务**：从简单的拾取和放置开始，逐步增加复杂性
2. **实验不同的策略**：每种策略都有其优势
3. **优化超参数**：使用 Weights & Biases 进行超参数搜索
4. **贡献回社区**：分享您的数据集和策略到 Hugging Face Hub
5. **探索高级功能**：多模态学习、sim-to-real 迁移等

祝您训练愉快！🤖
