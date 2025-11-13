#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""动作分块Transformer策略

根据论文 Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705)。
这里的主要变化包括删除未使用的代码、统一命名和添加有用的注释。
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTPolicy(PreTrainedPolicy):
    """
    动作分块Transformer策略，根据论文 Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (论文: https://huggingface.co/papers/2304.13705, 代码: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        参数:
            config: 策略配置类实例或None，如果为None则使用配置类的默认实例化。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): 目前 lr_backbone == lr
        # 是否应该删除这个，只返回 `self.parameters()`？
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """每当环境重置时应调用此方法。"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观察选择单个动作。

        此方法包装了 `select_actions`，以便每次向环境返回一个动作用于执行。它通过管理
        队列中的动作来工作，仅在队列为空时调用 `select_actions`。
        """
        self.eval()  # 将策略保持在eval模式，因为在消费队列时它可能被设置为train模式

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # n_action_steps > 1 时的动作队列逻辑。当 action_queue 耗尽时，通过查询策略来填充它。
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` 返回 (batch_size, n_action_steps, action_dim) 张量，但队列
            # 实际上具有形状 (n_action_steps, batch_size, *)，因此需要转置。
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观察预测一个动作块。"""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝以确保添加键不会修改原始数据
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """通过模型运行批次并计算训练或验证的损失。"""
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝以确保添加键不会修改原始数据
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # 计算 Dₖₗ(latent_pdf || standard_normal)。注意：在独立计算每个维度的KL散度后，
            # 我们对潜在维度求和以获得每个批次元素的总KL散度，然后对批次取平均值。
            # (详见 https://huggingface.co/papers/1312.6114 的附录B)。
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """时序集成，如 https://huggingface.co/papers/2304.13705 算法2中所述。

        权重计算为 wᵢ = exp(-temporal_ensemble_coeff * i)，其中 w₀ 是最旧的动作。
        然后通过除以 Σwᵢ 归一化为总和为1。以下是关于系数如何工作的一些直觉：
            - 设置为0会均匀加权所有动作。
            - 设置为正数会给较旧的动作更多权重。
            - 设置为负数会给较新的动作更多权重。
        注意：原始ACT工作使用的 `temporal_ensemble_coeff` 默认值为0.01。这导致
        较旧的动作比较新的动作权重更高（https://github.com/huggingface/lerobot/pull/319 中
        记录的实验暗示了为什么高权重新动作可能是有害的：这样做可能会削弱动作分块的好处）。

        这里我们使用在线方法来计算平均值，而不是缓存动作历史来离线计算平均值。
        对于简单的1D序列，它看起来像这样：

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # 离线计算
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # 在线计算
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """重置在线计算变量。"""
        self.ensembled_actions = None
        # (chunk_size,) 每个时间步序列中集成的动作数量计数。
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        接收一个 (batch, chunk_size, action_dim) 的动作序列，更新所有时间步的时序集成，
        并弹出/返回序列中的下一批动作。
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # 将 `self._ensembled_action` 初始化为回合第一个时间步预测的动作序列。
            self.ensembled_actions = actions.clone()
            # 注意：最后一个维度被unsqueeze以确保稍后可以正确广播进行张量操作。
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions 将具有形状 (batch_size, chunk_size - 1, action_dim)。
            # 计算这些条目的在线更新。
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # 最后一个动作没有先前的在线平均值，需要连接到末尾。
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "消费"第一个动作。
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """动作分块Transformer: ACTPolicy的底层神经网络。

    注意：在此代码中，我们使用术语 `vae_encoder`、'encoder'、`decoder`。含义如下：
        - `vae_encoder` 是变分自编码器(VAE)文献中的术语，指模型中编码目标数据（动作序列）
          和条件（机器人关节空间）的部分。
        - 使用带有 `encoder`（非VAE编码器）和 `decoder`（非VAE解码器）以及交叉注意力的
          transformer作为VAE解码器。对于这些术语，我们删除 `vae_` 前缀，因为我们有一个选项
          可以在不使用变分目标的情况下训练此模型（在这种情况下，我们完全删除 `vae_encoder`，
          并且此模型的任何内容都与VAE无关）。

                                 Transformer
                                 在推理时单独使用
                                 (在训练时充当VAE解码器)
                                ┌───────────────────────┐
                                │             输出      │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                输入      └─────┼──┘  │ 图像嵌入        │
                                │    状态嵌入           │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT风格的VAE编码器，输入tokens为 [cls, robot_state, *action_sequence]。
        # cls token形成潜在分布的参数（如 [*means, *log_variances]）。
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # 关节空间配置到隐藏维度的投影层。
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # 动作（关节空间目标）到隐藏维度的投影层。
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # 从VAE编码器输出到潜在分布参数空间的投影层。
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # VAE编码器输入的固定正弦位置嵌入。为批次维度unsqueeze。
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # 用于图像特征提取的骨干网络。
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 注意：这里假设我们使用的是ResNet模型（因此layer4是最终的特征图）。
            # 注意：此方法的forward返回一个字典：{"feature_map": output}。
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer（使用变分目标训练时充当VAE解码器）。
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer编码器输入投影。tokens结构为
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)]。
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer编码器位置嵌入。
        n_1d_tokens = 1  # 用于潜在变量
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer解码器。
        # transformer解码器的可学习位置嵌入（按照DETR对象查询的风格）。
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # transformer解码器输出上的最终动作回归头。
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """如原始代码中所述，对transformer参数进行Xavier-uniform初始化。"""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """通过动作分块Transformer（带可选VAE编码器）的前向传播。

        `batch` 应具有以下结构:
        {
            [robot_state_feature] (可选): (B, state_dim) 批次的机器人状态。

            [image_features]: (B, n_cameras, C, H, W) 批次的图像。
                和/或
            [env_state_feature]: (B, env_dim) 批次的环境状态。

            [action_feature] (可选，仅在使用VAE训练时): (B, chunk_size, action dim) 批次的动作。
        }

        返回:
            (B, chunk_size, action_dim) 批次的动作序列
            包含潜在PDF参数（均值，log(σ²)）的元组，均为 (B, L) 张量，其中L是潜在维度。
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "在训练模式下使用变分目标时必须提供动作。"
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # 准备潜在变量以输入到transformer编码器。
        if self.config.use_vae and ACTION in batch and self.training:
            # 准备VAE编码器的输入：[cls, *joint_space_configuration, *action_sequence]。
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # 准备固定位置嵌入。
            # 注意：detach()不应该是必需的，但保持与原始代码相同以防万一。
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # 为transformer编码器准备key padding mask。根据是否使用输入状态，
            # 我们在序列开始有1或2个额外的tokens（cls和robot state）
            # False表示不是padding token。
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # 通过VAE编码器的前向传播以获得潜在PDF参数。
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # 选择class token，形状为 (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 这是2log(sigma)。这样做是为了匹配原始实现。
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 使用重参数化技巧采样潜在变量。
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 当不使用VAE编码器时，我们将潜在变量设置为全零。
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): 移除对 `.to` 的调用以加速forward；预计算并使用buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # 准备transformer编码器输入。
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # 机器人状态token。
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # 环境状态token。
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # 对于图像列表，H和W可能变化，但H*W是恒定的。
            # 注意：如果修改此部分，请在MPS设备上验证梯度保持稳定（无爆炸或NaN）。
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # 将特征重排为 (sequence, batch, dim)。
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # 立即扩展而不是累积和连接
                # 转换为列表以正确扩展
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 沿序列维度堆叠所有tokens。
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # 通过transformer模块的前向传播。
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): 移除对 `device` 的调用；预计算并使用buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # 转回 (B, S, C)。
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """便捷模块，用于运行多个编码器层，可能后跟归一化。"""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈层。
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # 注意：[0] 仅选择输出，不选择注意力权重
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """便捷模块，用于运行多个解码器层，后跟归一化。"""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈层。
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        参数:
            x: (Decoder Sequence, Batch, Channel) 输入tokens张量。
            encoder_out: (Encoder Sequence, B, C) 我们正在交叉注意的编码器最后一层的输出特征。
            decoder_pos_embed: (ES, 1, C) keys的位置嵌入（来自编码器）。
            encoder_pos_embed: (DS, 1, C) queries的位置嵌入（来自解码器）。
        返回:
            (DS, B, C) 解码器输出特征的张量。
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # 仅选择输出，不选择注意力权重
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # 仅选择输出，不选择注意力权重
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """如 Attention is All You Need 中的1D正弦位置嵌入。

    参数:
        num_positions: 所需的token位置数量。
    返回: (num_positions, dimension) 位置嵌入（第一个维度是批次维度）。

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D正弦位置嵌入，类似于 Attention Is All You Need 中提出的内容。

    变化在于位置索引在 [0, 2π] 中归一化（不完全是：垂直方向的下界是1/H，
    水平方向的下界是1/W）。
    """

    def __init__(self, dimension: int):
        """
        参数:
            dimension: 嵌入的所需维度。
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # 正弦频率几何级数中的逆"公比"。
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
            x: (B, C, H, W) 批次的2D特征图，用于生成嵌入。
        返回:
            (1, C, H, W) 批次的对应正弦位置嵌入。
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # 注意：这些分别类似于 range(1, H+1) 和 range(1, W+1)，但在大多数实现中
        # 它们应该是 range(0, H) 和 range(0, W)。保持原样以匹配原始代码。
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "归一化"位置索引，使其范围在 [0, 2π]。
        # 注意：在分母上添加epsilon不应该是必需的，因为通过构造，y_embed和x_range的所有值
        # 都是非零的。这是原始代码的遗留物。
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # 注意：此堆叠然后展平操作会产生交错的正弦和余弦项。
        # pos_embed_x 和 pos_embed_y 是 (1, H, W, C // 2)。
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """根据字符串返回激活函数。"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation应该是relu/gelu/glu，而不是{activation}。")
