# 使用 🤗 PEFT 进行参数高效微调

[🤗 PEFT](https://github.com/huggingface/peft)（参数高效微调）是一个库，用于高效地将大型预训练模型（如预训练策略，例如 SmolVLA、π₀ 等）适应到新任务，而无需训练模型的所有参数，同时产生可比的性能。

安装 `lerobot[peft]` 可选包以启用 PEFT 支持。

要了解所有可能的适应方法，请参阅 [🤗 PEFT 文档](https://huggingface.co/docs/peft/index)。

## 训练 SmolVLA

在本节中，我们将向你展示如何使用 PEFT 在 libero 数据集上训练预训练的 SmolVLA 策略。
为简洁起见，我们仅在 `libero_spatial` 子集上训练。我们将使用 `lerobot/smolvla_base` 作为要进行参数高效微调的模型：

```
lerobot-train \
 --policy.path=lerobot/smolvla_base \
 --policy.repo_id=your_hub_name/my_libero_smolvla \
 --dataset.repo_id=HuggingFaceVLA/libero \
 --policy.output_features=null \
 --policy.input_features=null \
 --policy.optimizer_lr=1e-3 \
 --policy.scheduler_decay_lr=1e-4 \
 --env.type=libero \
 --env.task=libero_spatial \
 --steps=100000 \
 --batch_size=32 \
 --peft.method_type=LORA \
 --peft.r=64 \
 --peft.lora_alpha=64
```

注意 `--peft.method_type` 参数，它允许你选择要使用的 PEFT 方法。这里我们使用 [LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)（低秩适配器），这可能是迄今为止最流行的微调方法。低秩适应意味着我们只微调一个秩相对较低的矩阵，而不是完整的权重矩阵。这个秩可以使用 `--peft.r` 参数指定，LoRA 缩放因子使用 `--peft.lora_alpha` 指定（其中 `scaling = lora_alpha / r`）。秩越高，越接近完全微调。

还有更复杂的方法，它们有更多参数。这些目前尚不支持，如果你想看到特定的 PEFT 方法得到支持，请随时提出问题。

默认情况下，PEFT 将针对 SmolVLA 中 LM 专家的 `q_proj` 和 `v_proj` 层。它还将针对状态和动作投影矩阵，因为它们很可能是任务相关的。如果你需要针对不同的层，可以使用 `--peft.target_modules` 来指定要针对的层。你可以参考相应 PEFT 方法的文档以查看支持的输入，（例如，[LoRA 的 target_modules 文档](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.target_modules)）。
通常支持后缀列表或正则表达式。例如，要针对 `lm_expert` 的 MLP 而不是 `q` 和 `v` 投影，请使用：

```
--peft.target_modules='(model\.vlm_with_expert\.lm_expert\..*\.(down|gate|up)_proj|.*\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
```

如果你需要完全微调一个层而不仅仅是适应它，可以向 `--peft.full_training_modules` 参数提供层后缀列表：

```
--peft.full_training_modules=["state_proj"]
```

与用于完全微调的学习率相比，学习率和计划的目标学习率通常可以缩放 10 倍（例如，正常情况下为 1e-4，因此使用 LoRA 时为 1e-3）。
