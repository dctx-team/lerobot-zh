# 多 GPU 训练

本指南向你展示如何使用 [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) 在多个 GPU 上训练策略。

## 安装

`accelerate` 包含在 `training` 额外依赖中。使用以下命令安装：

```bash
pip install 'lerobot[training]'
```

## 使用多个 GPU 进行训练

你可以通过两种方式启动训练：

### 选项 1：不使用配置（直接指定参数）

你可以直接在命令中指定所有参数，而无需运行 `accelerate config`：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/my_trained_policy \
  --output_dir=outputs/train/act_multi_gpu \
  --job_name=act_multi_gpu \
  --wandb.enable=true
```

**关键 accelerate 参数：**

- `--multi_gpu`：启用多 GPU 训练
- `--num_processes=2`：要使用的 GPU 数量
- `--mixed_precision=fp16`：使用 fp16 混合精度（如果支持，可使用 `bf16`）

### 选项 2：使用 accelerate config

如果你希望保存配置，可以选择通过运行以下命令为你的硬件设置配置 accelerate：

```bash
accelerate config
```

这个交互式设置将询问你有关训练环境的问题（GPU 数量、混合精度设置等），并保存配置以供将来使用。对于单机上的简单多 GPU 设置，你可以使用以下推荐设置：

- 计算环境：This machine
- 机器数量：1
- 进程数量：（你想使用的 GPU 数量）
- 要使用的 GPU id：（留空以使用所有）
- 混合精度：fp16 或 bf16（推荐用于更快的训练）

然后使用以下命令启动训练：

```bash
accelerate launch $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/my_trained_policy \
  --output_dir=outputs/train/act_multi_gpu \
  --job_name=act_multi_gpu \
  --wandb.enable=true
```

## 工作原理

当你使用 accelerate 启动训练时：

1. **自动检测**：LeRobot 自动检测它是否在 accelerate 下运行
2. **数据分发**：你的批次自动在 GPU 之间分割
3. **梯度同步**：梯度在反向传播期间在 GPU 之间同步
4. **单进程日志记录**：只有主进程记录到 wandb 并保存检查点

## 学习率和训练步数缩放

**重要：** LeRobot **不会**根据 GPU 数量自动缩放学习率或训练步数。这使你可以完全控制训练超参数。

### 为什么不自动缩放？

许多分布式训练框架会根据 GPU 数量自动缩放学习率（例如，`lr = base_lr × num_gpus`）。
然而，LeRobot 保持学习率完全按照你指定的值。

### 何时以及如何缩放

如果你想在使用多个 GPU 时缩放超参数，应该手动执行：

**学习率缩放：**

```bash
# 示例：2 个 GPU，线性 LR 缩放
# 基础 LR：1e-4，使用 2 个 GPU -> 2e-4
accelerate launch --num_processes=2 $(which lerobot-train) \
  --optimizer.lr=2e-4 \
  --dataset.repo_id=lerobot/pusht \
  --policy=act
```

**训练步数缩放：**

由于有效批次大小 `bs` 随着多个 GPU 增加（batch_size × num_gpus），你可能希望按比例减少训练步数：

```bash
# 示例：2 个 GPU，有效批次大小增加 2 倍
# 原始：batch_size=8, steps=100000
# 使用 2 个 GPU：batch_size=8（总共 16），steps=50000
accelerate launch --num_processes=2 $(which lerobot-train) \
  --batch_size=8 \
  --steps=50000 \
  --dataset.repo_id=lerobot/pusht \
  --policy=act
```

## 注意事项

- `lerobot-train` 中的 `--policy.use_amp` 标志仅在**不**使用 accelerate 运行时使用。使用 accelerate 时，混合精度由 accelerate 的配置控制。
- 训练日志、检查点和 hub 上传仅由主进程完成，以避免冲突。非主进程禁用控制台日志记录以防止重复输出。
- 有效批次大小为 `batch_size × num_gpus`。如果你使用 4 个 GPU，`--batch_size=8`，你的有效批次大小为 32。
- 学习率调度在多个进程中正确处理——LeRobot 设置 `step_scheduler_with_optimizer=False` 以防止 accelerate 根据进程数量调整调度器步数。
- 保存或推送模型时，LeRobot 自动从 accelerate 的分布式包装器中解包模型以确保兼容性。
- WandB 集成自动仅在主进程上初始化，防止创建多个运行。

有关更高级的配置和故障排除，请参阅 [Accelerate 文档](https://huggingface.co/docs/accelerate)。如果你想了解更多关于如何在大量 GPU 上训练的信息，请查看这个很棒的指南：[Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。
