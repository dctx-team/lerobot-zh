# 多任务 DiT 策略

多任务扩散 Transformer（DiT）策略是原始扩散策略架构的演进，它利用大型 DiT 与文本和视觉条件进行多任务机器人学习。此实现支持扩散和流匹配目标用于动作生成，使机器人能够根据语言指令执行各种操作任务。

## 模型概述

该模型使用：

- **CLIP 视觉编码器**：处理来自多个摄像头视图的 RGB 图像
- **CLIP 文本编码器**：编码语言任务指令（冻结权重与可学习投影）
- **扩散 Transformer**：根据观测和语言预测动作序列
- **两个目标**：支持扩散（DDPM/DDIM）和流匹配用于动作生成

这个模型令人兴奋，因为您可以实现极高的灵巧性，与数十亿参数的 VLA 相媲美，但只需约 450M 参数和显著更少的训练。

## 安装要求

多任务 DiT 策略有额外的依赖项。使用以下命令安装：

```bash
pip install lerobot[multi_task_dit]
```

这将安装所有必要的依赖项，包括用于 CLIP 模型的 HuggingFace Transformers 库。

## 使用

要在 LeRobot 配置中使用多任务 DiT，请将策略类型指定为：

```python
policy.type=multi_task_dit
```

## 训练

### 基本训练命令

以下是在您的数据集上训练多任务 DiT 的完整训练命令：

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=./outputs/multitask_dit_training \
  --batch_size=32 \
  --steps=5000 \
  --save_freq=500 \
  --log_freq=100 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.repo_id="HF_USER/multitask-dit-your-robot" \
  --wandb.enable=true
```

### 推荐的超参数和数据集详细信息（30Hz 控制频率）

为了获得可靠的性能，从这些建议的默认超参数开始：

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=./outputs/mutitask_dit_training \
  --batch_size=320 \
  --steps=30000 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.repo_id="HF_USER/multitask-dit-your-robot" \
  --wandb.enable=true
```

**关键参数：**

- **批量大小**：192-320 - 如果您有可以支持此大小的 GPU，您将获得最佳的训练动态
- **Horizon**：32 - 要预测的动作步数，30Hz 时约 1.0 秒
- **n_action_steps**：24 - 30Hz 时约 0.8 秒
- **Objective**：`diffusion` - 从扩散开始，如果生成质量差，则尝试流匹配
- **训练步数**：单个任务推荐 >30k 步

### 训练配置参数

#### 目标选择

在扩散和流匹配之间选择：

```bash
# 扩散目标（默认）
--policy.objective=diffusion \
--policy.noise_scheduler_type=DDPM \  # 或 "DDIM"
--policy.num_train_timesteps=100 \
--policy.num_inference_steps=10 \  # 用于更快的推理
--policy.beta_schedule=squaredcos_cap_v2 \  # 噪声调度类型
--policy.prediction_type=epsilon \  # "epsilon"（预测噪声）或 "sample"（预测干净）
--policy.clip_sample=true \  # 在去噪期间裁剪样本
--policy.clip_sample_range=1.0  # 裁剪范围 [-x, x]

# 流匹配目标
--policy.objective=flow_matching \
--policy.timestep_sampling_strategy=beta \  # 或 "uniform" | beta 采样策略在实践中表现更好
--policy.num_integration_steps=100 \
--policy.integration_method=euler \  # 或 "rk4"
--policy.sigma_min=0.0  # 流插值路径中的最小噪声
```

#### Transformer 架构

根据数据集大小调整模型容量：

```bash
# 小数据集（< 100 个示例）
--policy.num_layers=4 \
--policy.hidden_dim=512 \
--policy.num_heads=8  # 理想情况下应为 hidden_dim // 64

# 中等数据集（100-5k 个示例）- 默认
--policy.num_layers=6 \
--policy.hidden_dim=512 \
--policy.num_heads=8  # 理想情况下应为 hidden_dim // 64

# 大数据集（> 5k 个示例）
--policy.num_layers=8 \
--policy.hidden_dim=512 \
--policy.num_heads=8   # 理想情况下应为 hidden_dim // 64
```

**位置编码选项：**

该模型支持两种动作序列的位置编码方法：

```bash
# 旋转位置嵌入（RoPE）- 默认，推荐
--policy.use_rope=true \
--policy.rope_base=10000.0  # RoPE 的基频

# 绝对位置编码
--policy.use_positional_encoding=true  # 为 true 时禁用 RoPE
```

**其他 Transformer 参数：**

```bash
--policy.dropout=0.1  # DiT 块的 dropout 率（0.0-1.0）
--policy.timestep_embed_dim=256  # 时间步嵌入维度
```

#### 视觉编码器配置

```bash
# 使用不同的 CLIP 模型以获得更多表现力，但代价是推理时间
# 根据任务的复杂性和数据集的大小尝试更大或更小的模型
--policy.vision_encoder_name=openai/clip-vit-large-patch14

# 每个摄像头使用单独的视觉编码器
# 当摄像头具有显著不同的特性时，这可能很有用，但
# 要注意增加的 VRAM 占用。
--policy.use_separate_rgb_encoder_per_camera=true

# 图像预处理
--policy.image_resize_shape=[XXX,YYY] \ # 您可能需要调整图像大小以加快推理速度
--policy.image_crop_shape=[224,224] \
--policy.image_crop_is_random=true  # 训练期间随机，推理时居中
```

#### 文本编码器配置

```bash
# 使用不同的 CLIP 文本编码器模型
# 与视觉相同：根据任务的复杂性和数据集的大小尝试更大或更小的模型
--policy.text_encoder_name=openai/clip-vit-large-patch14
```

#### 学习率配置

视觉编码器使用单独的学习率乘数，建议从 1/10 开始作为理想起点：

```bash
--policy.optimizer_lr=2e-5 \
--policy.vision_encoder_lr_multiplier=0.1  # 视觉编码器 LR = 0.1 * optimizer_lr
```

### 训练调优指南

#### 1. 使用 Beta 采样的流匹配

这里的原始扩散实现基于 [TRI 的 LBM 论文](https://arxiv.org/abs/2507.05331)中描述的工作

此外，我们实现了一个流匹配目标，在 [Boston Dynamics 博客文章](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)中进行了高层次描述。

考虑测试流匹配目标并评估您的任务的性能差异：

```bash
--policy.objective=flow_matching \
--policy.timestep_sampling_strategy=beta \
--policy.timestep_sampling_alpha=1.5 \
--policy.timestep_sampling_beta=1.0 \
--policy.timestep_sampling_s=0.999
```

这并不是在每个用例中都是灵丹妙药，但它偶尔会产生更平滑和更一致的动作。

#### 2. Transformer 层数

将模型容量与数据集大小匹配：

- **小数据集**（< 100 个示例）：减少到 4 层
- **大数据集**（> 5k 个示例）：增加到 8 层

#### 3. `horizon` 调优

模型对您选择的 horizon 可能很敏感。根据您的控制频率从大约 1 秒的 horizon 开始：

- **30 Hz 频率**：`horizon=30`
- **10 Hz 频率**：`horizon=10`

然后尝试从那里增加。horizon 决定了模型预测动作的未来距离。

#### 4. `n_action_steps` 敏感性

模型对 `n_action_steps` 也可能非常敏感。根据您的控制频率从大约 0.8 秒开始并从那里调整：

- **较低值**：更具反应性，但对于长时域任务可能不太稳定
- **较高值**：更适合长时域执行，但开环故障在恢复方面受到限制

### 推理调优

为了更快的推理，使用 DDIM 和更少的采样步骤：

```bash
--policy.noise_scheduler_type=DDIM \
--policy.num_inference_steps=10
```

### 恢复训练

要从检查点恢复训练：

```bash
lerobot-train \
  --config_path=./outputs/mutitask_dit_training/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

检查点目录应包含 `model.safetensors` 和 `config.json` 文件（在训练期间自动保存）。恢复时，配置从检查点加载，因此您不需要指定其他参数。

## 常见故障模式和调试

训练这些模型可能很棘手。以下是常见的故障模式和调试方法：

### 空闲/无运动

模型在推理期间可能"崩溃"，导致静态或无运动。这可能发生在以下情况：

1. **训练数据不足**：如果您只有 20-50 个示例，请尝试大致将数据集大小翻倍。一旦您有超过 300 个示例，如果您仍然看到这种情况，任务可能太复杂了。

2. **多个相似任务**：当您的数据集包含多个相似任务（例如，拾取 2 个不同的物体）时，模型可能过度依赖语言条件，而语言条件可能不够丰富。

**调试提示：**

- 增加数据集大小（翻倍直到超过 300 个示例）
- 训练更长时间，最多 100k 步，即使损失趋于平稳
- 检查模型是否接收到正确的语言指令或增加指令的多样性

### 执行错误的任务

有时机器人会完全忽略您的指令并执行其他任务。这通常只在您训练了多个任务时才会发生。

**潜在原因：**

- 语言指令模糊
- 任务特定训练数据不足
- 模型在多任务数据集中的相似任务之间混淆

**调试提示：**

- 验证语言指令的特异性，特别是如果多个任务之间的描述相似
- 检查训练数据集中的任务分布，并为失败/被忽略的任务添加权重
- 考虑任务特定的微调

### 训练不稳定

如果训练损失不稳定或发散：

- 尝试在 `1e-5` 和 `3e-4` 之间调整学习率
- 如果可能，增加批量大小
- 检查数据集归一化是否正确
- 验证图像预处理是否正常工作

## 性能考虑

### GPU 要求

- **推理**：建议至少使用 RTX 5070 Ti（或同等 GPU）以获得合理的速度性能
- **训练**：具有足够 VRAM 以加载 >64 批量大小的 GPU 是理想的，这将根据图像观测的数量等而变化

### 批量大小建议

- **最小值**：64（小于此值可能导致训练不稳定）
- **推荐值**：256-320（最佳性能，需要更大的 GPU）

## 示例：在自定义数据集上训练

以下是在自定义数据集上训练的完整示例：

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=./outputs/mutitask_dit_training \
  --batch_size=320 \
  --steps=30000 \
  --save_freq=1000 \
  --log_freq=100 \
  --eval_freq=1000 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_layers=6 \
  --policy.hidden_dim=512 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[320,240] \
  --policy.image_crop_shape=[224,224] \
  --policy.repo_id="HF_USER/multitask-dit-your-robot" \
  --wandb.enable=true \
  --wandb.project=multitask_dit
```

## Libero 结果

```
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=multi_task_dit \
  --policy.push_to_hub=false \
  --output_dir="./outputs/multitask_dit_libero" \
  --job_name="multitask-dit-libero" \
  --wandb.enable=true \
  --wandb.project=multitask_dit_libero \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-5,5]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.1,0.1]}}}' \
  --dataset.video_backend=torchcodec \
  --policy.use_amp=true \
  --policy.horizon=48 \
  --policy.n_obs_steps=2 \
  --policy.use_rope=true \
  --policy.use_positional_encoding=false \
  --policy.hidden_dim=768 \
  --policy.num_layers=8 \
  --policy.num_heads=12 \
  --policy.dropout=0.1 \
  --policy.timestep_embed_dim=256 \
  --policy.objective=diffusion \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0 \
  --policy.scheduler_warmup_steps=0 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[256,256] \
  --policy.image_crop_is_random=true \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --policy.device=cuda \
  --num_workers=8 \
  --save_freq=4000 \
  --log_freq=100 \
  --steps=100000 \
  --batch_size=320
```

结果：

| LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO 10 | 平均 |
| -------------- | ------------- | ----------- | --------- | ---- |
| 87.0           | 98.2          | 93.8        | 83.2      | 90.6 |

## 参考文献

有关技术实现和架构的更多详细信息，请参见：

- [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/abs/2507.05331)
- [Large Behavior Models and Atlas Find New Footing](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)
- [Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy](https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy)
