#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
# and The HuggingFace Inc. team. All rights reserved.
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

import warnings
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_output_shape, populate_queues
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.vqbet_utils import GPT, ResidualVQ
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# ruff: noqa: N806


class VQBeTPolicy(PreTrainedPolicy):
    """
    VQ-BeT策略，基于论文"Behavior Generation with Latent Actions"
    """

    config_class = VQBeTConfig
    name = "vqbet"

    def __init__(
        self,
        config: VQBeTConfig | None = None,
    ):
        """
        参数:
            config: 策略配置类实例，如果为None，则使用配置类的默认实例化。
            dataset_stats: 用于归一化的数据集统计信息。如果在此处未传递，则期望在使用策略之前
                通过调用`load_state_dict`传递。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.vqbet = VQBeTModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        vqvae_params = (
            list(self.vqbet.action_head.vqvae_model.encoder.parameters())
            + list(self.vqbet.action_head.vqvae_model.decoder.parameters())
            + list(self.vqbet.action_head.vqvae_model.vq_layer.parameters())
        )
        decay_params, no_decay_params = self.vqbet.policy.configure_parameters()
        decay_params = (
            decay_params
            + list(self.vqbet.rgb_encoder.parameters())
            + list(self.vqbet.state_projector.parameters())
            + list(self.vqbet.rgb_feature_projector.parameters())
            + [self.vqbet.action_token]
            + list(self.vqbet.action_head.map_to_cbet_preds_offset.parameters())
        )

        if self.config.sequentially_select:
            decay_params = (
                decay_params
                + list(self.vqbet.action_head.map_to_cbet_preds_primary_bin.parameters())
                + list(self.vqbet.action_head.map_to_cbet_preds_secondary_bin.parameters())
            )
        else:
            decay_params = decay_params + list(self.vqbet.action_head.map_to_cbet_preds_bin.parameters())

        return [
            {
                "params": decay_params,
            },
            {
                "params": vqvae_params,
                "weight_decay": self.config.optimizer_vqvae_weight_decay,
                "lr": self.config.optimizer_vqvae_lr,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

    def reset(self):
        """
        清空观测和动作队列。应该在`env.reset()`时调用。
        队列在策略推演过程中被填充，它们包含最新的n个观测和动作。
        """
        self._queues = {
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.action_chunk_size),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.vqbet(batch, rollout=True)[:, : self.config.action_chunk_size]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """在给定环境观测的情况下选择单个动作。

        此方法封装了`select_actions`，以便一次返回一个动作在环境中执行。它通过管理队列中的
        动作来工作，仅在队列为空时调用`select_actions`。
        """
        # 注意：对于离线评估，批次中有动作，因此我们需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)
        batch = dict(batch)  # 浅拷贝，以便添加键不会修改原始数据
        # 注意：重要的是，这发生在将图像堆叠成单个键之后。
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # 注意：对于离线评估，批次中有动作，因此我们需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if not self.vqbet.action_head.vqvae_model.discretized.item():
            warnings.warn(
                "要在环境中评估，您的VQ-BeT模型应包含预训练的残差VQ。",
                stacklevel=1,
            )

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # 由于动作队列中的数据维度是(action_chunk_size, batch_size, action_dim)，我们转置动作并填充队列
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """通过模型运行批次数据并计算训练或验证的损失。"""
        batch = dict(batch)  # 浅拷贝，以便添加键不会修改原始数据
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # VQ-BeT在训练BeT之前使用VQ-VAE离散化动作（请参阅VQ-BeT论文的3.2节 https://huggingface.co/papers/2403.03181）
        if not self.vqbet.action_head.vqvae_model.discretized.item():
            # loss: 训练RVQ的总损失
            # n_different_codes: 单个批次中使用的总可能VQ代码数量（其中有多少至少有一个编码器嵌入作为最近邻）。最多可以是`vqvae_n_embed * RVQ层数(=2)`。
            # n_different_combinations: 单个批次中使用的所有可能组合中有多少种不同的代码组合。最多可以是`vqvae_n_embed ^ RVQ层数(=2)`（提示：将RVQ视为决策树）。
            loss, n_different_codes, n_different_combinations, recon_l1_error = (
                self.vqbet.action_head.discretize(self.config.n_vqvae_training_steps, batch[ACTION])
            )
            return loss, {
                "n_different_codes": n_different_codes,
                "n_different_combinations": n_different_combinations,
                "recon_l1_error": recon_l1_error,
            }
        # 如果残差VQ已经训练完成，VQ-BeT训练其GPT和码预测头/偏移预测头部分。
        _, loss_dict = self.vqbet(batch, rollout=False)
        loss = loss_dict.pop("loss")

        return loss, loss_dict


class SpatialSoftmax(nn.Module):
    """
    Finn等人在"Deep Spatial Autoencoders for Visuomotor Learning"中描述的空间软性Argmax操作
    (https://huggingface.co/papers/1509.06113)。robomimic实现的最小化移植版本。

    从高层次来看，这采用2D特征图（来自卷积网络/ViT）并返回每个通道激活的"质心"，
    即策略要关注的图像空间中的关键点。

    示例：采用大小为(512x10x12)的特征图。我们生成归一化坐标网格(10x12x2)：
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    这通过在激活(512x120)上应用按通道的softmax并与坐标(120x2)计算点积来实现，
    以获得最大激活的期望点(512x2)。

    上述示例生成512个关键点（对应于512个输入通道）。我们可以选择性地提供num_kp != None
    来控制关键点的数量。这是通过首先应用可学习的线性映射(in_channels, H, W) -> (num_kp, H, W)来实现的。
    """

    def __init__(self, input_shape, num_kp=None):
        """
        参数:
            input_shape (list): (C, H, W) 输入特征图形状。
            num_kp (int): 输出中关键点的数量。如果为None，输出将具有与输入相同数量的通道。
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # 我们可以直接使用torch.linspace，但这似乎与numpy的行为略有不同，
        # 会导致预训练模型的pc_success略有下降。
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # 注册为缓冲区，以便将其移动到正确的设备。
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        参数:
            features: (B, C, H, W) 输入特征图。
        返回:
            (B, K, 2) 关键点的图像空间坐标。
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W]，其中K是关键点数量
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax归一化
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2]，用于x和y维度中的空间坐标均值
        expected_xy = attention @ self.pos_grid
        # 重塑为[B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class VQBeTModel(nn.Module):
    """VQ-BeT：VQ-BeT的底层神经网络

    注意：在此代码中，我们使用术语`rgb_encoder`、'policy'、`action_head`。其含义如下：
        - `rgb_encoder`将rgb风格的图像观测处理为一维嵌入向量
        - `policy`是一个minGPT架构，接受观测序列和动作查询标记以生成`features`。
        - 这些`features`通过动作头，该头经过代码预测、偏移预测头，
        最终生成动作块的预测。

        -------------------------------** 图例 **-------------------------------
        │   n = n_obs_steps, p = n_action_pred_token, c = action_chunk_size)   │
        │   o_{t} : 时间步{t}的视觉观测                           │
        │   s_{t} : 时间步{t}的状态观测                            │
        │   a_{t} : 时间步{t}的动作                                       │
        │   A_Q : action_query_token（动作查询标记）                                             │
        --------------------------------------------------------------------------


        训练阶段1. 使用残差VQ离散化动作（进行config.n_vqvae_training_steps步）


        ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
        │                 │            │                 │            │                 │
        │   RVQ编码器     │    ─►      │     残差        │    ─►      │   RVQ解码器     │
        │ (a_{t}~a_{t+p}) │            │  代码量化器     │            │                 │
        │                 │            │                 │            │                 │
        └─────────────────┘            └─────────────────┘            └─────────────────┘

        训练阶段2.

          时间步{t-n+1}   时间步{t-n+2}                时间步{t}
            ┌─────┴─────┐     ┌─────┴─────┐                 ┌─────┴─────┐

        o_{t-n+1}         o_{t-n+2}           ...         o_{t}
            │                 │                             │
            │ s_{t-n+1}       │ s_{t-n+2}         ...       │   s_{t}           p
            │     │           │     │                       │     │     ┌───────┴───────┐
            │     │    A_Q    │     │    A_Q          ...   │     │    A_Q     ...     A_Q
            │     │     │     │     │     │                 │     │     │               │
        ┌───▼─────▼─────▼─────▼─────▼─────▼─────────────────▼─────▼─────▼───────────────▼───┐
        │                                                                                   │
        │                                       GPT                                         │       =>    policy
        │                                                                                   │
        └───────────────▼─────────────────▼─────────────────────────────▼───────────────▼───┘
                        │                 │                             │               │
                    ┌───┴───┐         ┌───┴───┐                     ┌───┴───┐       ┌───┴───┐
                  code    offset    code    offset                code    offset  code    offset
                    ▼       │         ▼       │                     ▼       │       ▼       │       =>    action_head
               RVQ解码器    │    RVQ解码器    │                RVQ解码器    │  RVQ解码器    │
                    └── + ──┘         └── + ──┘                     └── + ──┘       └── + ──┘
                        ▼                 ▼                             ▼               ▼
                   动作块            动作块                        动作块           动作块
                    a_{t-n+1} ~       a_{t-n+2} ~                   a_{t} ~     ...  a_{t+p-1} ~
                     a_{t-n+c}         a_{t-n+c+1}                   a_{t+c-1}        a_{t+p+c-1}

                                                                        ▼
                                                      仅此块在推演中使用！
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = VQBeTRgbEncoder(config)
        self.num_images = len(self.config.image_features)
        # 此动作查询标记用作查询动作块的提示。请参阅上图中的"A_Q"。
        # 注意：在前向传播过程中，此标记会根据需要重复多次。作者还尝试独立初始化所需数量的标记，但观察到结果较差。
        self.action_token = nn.Parameter(torch.randn(1, 1, self.config.gpt_input_dim))

        # 要将状态和观测特征输入GPT层，我们首先投影这些特征以适应GPT输入大小的形状。
        self.state_projector = MLP(
            config.robot_state_feature.shape[0], hidden_channels=[self.config.gpt_input_dim]
        )
        self.rgb_feature_projector = MLP(
            self.rgb_encoder.feature_dim, hidden_channels=[self.config.gpt_input_dim]
        )

        # VQ-BeT的GPT部分
        self.policy = GPT(config)
        # VQ-BeT的码预测头/偏移预测头部分
        self.action_head = VQBeTHead(config)

        # 动作标记用于：每个观测步骤、当前动作标记以及所有未来动作标记。
        num_tokens = self.config.n_action_pred_token + self.config.n_obs_steps - 1
        self.register_buffer(
            "select_target_actions_indices",
            torch.row_stack([torch.arange(i, i + self.config.action_chunk_size) for i in range(num_tokens)]),
        )

    def forward(self, batch: dict[str, Tensor], rollout: bool) -> tuple[dict, dict]:
        # 输入验证。
        assert set(batch).issuperset({OBS_STATE, OBS_IMAGES})
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # 提取图像特征（首先合并批次和序列维度）。
        img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
        # 分离批次和序列维度。
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s n ...", b=batch_size, s=n_obs_steps, n=self.num_images
        )

        # 按类文档字符串中所示安排先前和当前观测步骤标记。
        # 首先将特征投影到标记维度。
        rgb_tokens = self.rgb_feature_projector(
            img_features
        )  # (batch, obs_step, 不同相机数量, 投影维度)
        input_tokens = [rgb_tokens[:, :, i] for i in range(rgb_tokens.size(2))]
        input_tokens.append(self.state_projector(batch[OBS_STATE]))  # (batch, obs_step, 投影维度)
        input_tokens.append(einops.repeat(self.action_token, "1 1 d -> b n d", b=batch_size, n=n_obs_steps))
        # 通过堆叠和重新排列来交错标记。
        input_tokens = torch.stack(input_tokens, dim=2)
        input_tokens = einops.rearrange(input_tokens, "b n t d -> b (n t) d")

        len_additional_action_token = self.config.n_action_pred_token - 1
        future_action_tokens = self.action_token.repeat(batch_size, len_additional_action_token, 1)

        # 添加用于预测未来动作块的额外动作查询标记
        input_tokens = torch.cat([input_tokens, future_action_tokens], dim=1)

        # 获取动作特征（通过GPT传递）
        features = self.policy(input_tokens)
        # len(self.config.input_features)是不同观测模式的数量。
        # 这一行获取动作提示标记的索引。
        historical_act_pred_index = np.arange(0, n_obs_steps) * (len(self.config.input_features) + 1) + len(
            self.config.input_features
        )

        # 仅提取动作查询位置的输出标记：
        # 行为Transformer（BeT）和VQ-BeT都是序列到序列预测模型，
        # 将顺序观测映射到顺序动作（请参阅BeT论文的2.2节 https://huggingface.co/papers/2206.11251）。
        # 因此，除了当前和未来动作（预测未来动作：可选）外，它还预测历史动作序列。
        if len_additional_action_token > 0:
            features = torch.cat(
                [features[:, historical_act_pred_index], features[:, -len_additional_action_token:]], dim=1
            )
        else:
            features = features[:, historical_act_pred_index]
        # 通过动作头传递
        action_head_output = self.action_head(features)
        # 如果是推演，VQ-BeT不计算损失
        if rollout:
            return action_head_output["predicted_action"][:, n_obs_steps - 1, :].reshape(
                batch_size, self.config.action_chunk_size, -1
            )
        # 否则，它计算总体损失（码预测损失和偏移损失）
        else:
            output = batch[ACTION][:, self.select_target_actions_indices]
            loss = self.action_head.loss_fn(action_head_output, output, reduction="mean")
            return action_head_output, loss


class VQBeTHead(nn.Module):
    def __init__(self, config: VQBeTConfig):
        """
        VQBeTHead接收GPT层的输出，并通过码预测头（`self.map_to_cbet_preds_bin`）和偏移预测头（`self.map_to_cbet_preds_offset`）传递特征

        self.map_to_cbet_preds_bin: 输出每个代码（对于每一层）的概率。
            `self.map_to_cbet_preds_bin`的输入维度与GPT的输出相同，
            `self.map_to_cbet_preds_bin`的输出维度是`self.vqvae_model.vqvae_num_layers (=固定为2) * self.config.vqvae_n_embed`。
            如果智能体顺序选择代码，我们使用self.map_to_cbet_preds_primary_bin和self.map_to_cbet_preds_secondary_bin而不是self._map_to_cbet_preds_bin。

        self.map_to_cbet_preds_offset: 输出所有层中所有代码的预测偏移量。
            `self.map_to_cbet_preds_offset`的输入维度与GPT的输出相同，
            `self.map_to_cbet_preds_offset`的输出维度是`self.vqvae_model.vqvae_num_layers (=固定为2) * self.config.vqvae_n_embed * config.action_chunk_size * config.action_feature.shape[0]`。
        """

        super().__init__()
        self.config = config
        # 初始化vqvae
        self.vqvae_model = VqVae(config)
        if config.sequentially_select:
            self.map_to_cbet_preds_primary_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[self.config.vqvae_n_embed],
            )
            self.map_to_cbet_preds_secondary_bin = MLP(
                in_channels=config.gpt_output_dim + self.config.vqvae_n_embed,
                hidden_channels=[self.config.vqvae_n_embed],
            )
        else:
            self.map_to_cbet_preds_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[self.vqvae_model.vqvae_num_layers * self.config.vqvae_n_embed],
            )
        self.map_to_cbet_preds_offset = MLP(
            in_channels=config.gpt_output_dim,
            hidden_channels=[
                self.vqvae_model.vqvae_num_layers
                * self.config.vqvae_n_embed
                * config.action_chunk_size
                * config.action_feature.shape[0],
            ],
        )
        # 损失
        self._focal_loss_fn = FocalLoss(gamma=2.0)

    def discretize(self, n_vqvae_training_steps, actions):
        # 使用滑动窗口方法调整动作序列数据的大小以适应动作块大小。
        actions = torch.cat(
            [
                actions[:, j : j + self.config.action_chunk_size, :]
                for j in range(actions.shape[1] + 1 - self.config.action_chunk_size)
            ],
            dim=0,
        )
        # `actions`是形状为(new_batch, action_chunk_size, action_dim)的张量，其中new_batch是使用滑动窗口从原始序列创建的可能块数。

        loss, metric = self.vqvae_model.vqvae_forward(actions)
        n_different_codes = sum(
            [len(torch.unique(metric[2][:, i])) for i in range(self.vqvae_model.vqvae_num_layers)]
        )
        n_different_combinations = len(torch.unique(metric[2], dim=0))
        recon_l1_error = metric[0].detach().cpu().item()
        self.vqvae_model.optimized_steps += 1
        # 如果我们更新RVQ超过`n_vqvae_training_steps`步，我们冻结RVQ部分。
        if self.vqvae_model.optimized_steps >= n_vqvae_training_steps:
            self.vqvae_model.discretized = torch.tensor(True)
            self.vqvae_model.vq_layer.freeze_codebook = torch.tensor(True)
            print("完成离散化动作数据！")
            self.vqvae_model.eval()
            for param in self.vqvae_model.vq_layer.parameters():
                param.requires_grad = False
        return loss, n_different_codes, n_different_combinations, recon_l1_error

    def forward(self, x, **kwargs) -> dict:
        # N是批次大小，T是通过同一个GPT处理的动作查询标记数量
        N, T, _ = x.shape
        # 我们并行计算N和T。因此，维度将是
        # (批次大小 * 动作查询标记数量, 动作块大小, 动作维度)
        x = einops.rearrange(x, "N T WA -> (N T) WA")

        # 采样偏移量
        cbet_offsets = self.map_to_cbet_preds_offset(x)
        cbet_offsets = einops.rearrange(
            cbet_offsets,
            "(NT) (G C WA) -> (NT) G C WA",
            G=self.vqvae_model.vqvae_num_layers,
            C=self.config.vqvae_n_embed,
        )
        # 如果self.config.sequentially_select为True，码预测头首先采样主代码，然后采样次代码
        if self.config.sequentially_select:
            cbet_primary_logits = self.map_to_cbet_preds_primary_bin(x)

            # 首先选择主码
            cbet_primary_probs = torch.softmax(
                cbet_primary_logits / self.config.bet_softmax_temperature, dim=-1
            )
            NT, choices = cbet_primary_probs.shape
            sampled_primary_centers = einops.rearrange(
                torch.multinomial(cbet_primary_probs.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )

            cbet_secondary_logits = self.map_to_cbet_preds_secondary_bin(
                torch.cat(
                    (x, F.one_hot(sampled_primary_centers, num_classes=self.config.vqvae_n_embed)),
                    axis=1,
                )
            )
            cbet_secondary_probs = torch.softmax(
                cbet_secondary_logits / self.config.bet_softmax_temperature, dim=-1
            )
            sampled_secondary_centers = einops.rearrange(
                torch.multinomial(cbet_secondary_probs.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack((sampled_primary_centers, sampled_secondary_centers), axis=1)
            cbet_logits = torch.stack([cbet_primary_logits, cbet_secondary_logits], dim=1)
        # 如果self.config.sequentially_select为False，码预测头同时采样主代码和次代码。
        else:
            cbet_logits = self.map_to_cbet_preds_bin(x)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self.vqvae_model.vqvae_num_layers
            )
            cbet_probs = torch.softmax(cbet_logits / self.config.bet_softmax_temperature, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        device = get_device_from_parameters(self)
        indices = (
            torch.arange(NT, device=device).unsqueeze(1),
            torch.arange(self.vqvae_model.vqvae_num_layers, device=device).unsqueeze(0),
            sampled_centers,
        )
        # 使用高级索引来采样值（仅提取对应于采样代码的偏移量。）
        sampled_offsets = cbet_offsets[indices]
        # 然后，对RVQ层上的偏移量求和，以获得码预测的净偏移量
        sampled_offsets = sampled_offsets.sum(dim=1)
        with torch.no_grad():
            # 获取每层的质心（= 对应于代码的向量）以通过RVQ解码器传递
            return_decoder_input = self.vqvae_model.get_embeddings_from_code(sampled_centers).clone().detach()
            # 通过解码器传递质心以获取动作。
            decoded_action = self.vqvae_model.get_action_from_latent(return_decoder_input).clone().detach()
        # 重塑提取的偏移量以匹配解码的质心
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self.config.action_chunk_size
        )
        # 添加偏移量和解码的质心
        predicted_action = decoded_action + sampled_offsets
        predicted_action = einops.rearrange(
            predicted_action,
            "(N T) W A -> N T (W A)",
            N=N,
            T=T,
            W=self.config.action_chunk_size,
        )

        return {
            "cbet_logits": cbet_logits,
            "predicted_action": predicted_action,
            "sampled_centers": sampled_centers,
            "decoded_action": decoded_action,
        }

    def loss_fn(self, pred, target, **kwargs):
        """
        对于给定的真实动作值（target）和预测（pred），此函数计算总体损失。

        predicted_action: 预测的动作块（偏移量 + 解码的质心）
        sampled_centers: 采样的质心（RVQ的代码）
        decoded_action: 解码的动作，通过将sampled_centers传递给RVQ解码器产生
        NT: 批次大小 * T
        T: 通过同一个GPT处理的动作查询标记数量
        cbet_logits: 每层中所有代码的概率
        """
        action_seq = target
        predicted_action = pred["predicted_action"]
        sampled_centers = pred["sampled_centers"]
        decoded_action = pred["decoded_action"]
        NT = predicted_action.shape[0] * predicted_action.shape[1]

        cbet_logits = pred["cbet_logits"]

        predicted_action = einops.rearrange(
            predicted_action, "N T (W A) -> (N T) W A", W=self.config.action_chunk_size
        )

        action_seq = einops.rearrange(action_seq, "N T W A -> (N T) W A")
        # 计算动作的损失。
        # 首先，我们需要为每个真实动作找到最近的聚类中心。
        with torch.no_grad():
            state_vq, action_bins = self.vqvae_model.get_code(action_seq)  # action_bins: NT, G

        # 现在我们可以计算损失。

        # 偏移损失是预测动作和真实动作之间的L1距离
        offset_loss = F.l1_loss(action_seq, predicted_action)

        # 计算主代码预测损失
        cbet_loss1 = self._focal_loss_fn(
            cbet_logits[:, 0, :],
            action_bins[:, 0],
        )
        # 计算次代码预测损失
        cbet_loss2 = self._focal_loss_fn(
            cbet_logits[:, 1, :],
            action_bins[:, 1],
        )
        # 添加所有预测损失
        cbet_loss = (
            cbet_loss1 * self.config.primary_code_loss_weight
            + cbet_loss2 * self.config.secondary_code_loss_weight
        )

        equal_primary_code_rate = torch.sum((action_bins[:, 0] == sampled_centers[:, 0]).int()) / (NT)
        equal_secondary_code_rate = torch.sum((action_bins[:, 1] == sampled_centers[:, 1]).int()) / (NT)

        action_mse_error = torch.mean((action_seq - predicted_action) ** 2)
        vq_action_error = torch.mean(torch.abs(action_seq - decoded_action))
        offset_action_error = torch.mean(torch.abs(action_seq - predicted_action))
        action_error_max = torch.max(torch.abs(action_seq - predicted_action))

        loss = cbet_loss + self.config.offset_loss_weight * offset_loss

        loss_dict = {
            "loss": loss,
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "equal_primary_code_rate": equal_primary_code_rate.detach().cpu().item(),
            "equal_secondary_code_rate": equal_secondary_code_rate.detach().cpu().item(),
            "vq_action_error": vq_action_error.detach().cpu().item(),
            "offset_action_error": offset_action_error.detach().cpu().item(),
            "action_error_max": action_error_max.detach().cpu().item(),
            "action_mse_error": action_mse_error.detach().cpu().item(),
        }
        return loss_dict


class VQBeTRgbEncoder(nn.Module):
    """将RGB图像编码为1D特征向量。

    包括首先归一化和裁剪图像的能力。

    与modeling_diffusion.py中的DiffusionRgbEncoder相同
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        # 设置可选的预处理。
        if config.crop_shape is not None:
            self.do_crop = True
            # 对于评估始终使用中心裁剪
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # 设置骨干网络。
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # 注意：这假设layer4特征图是children()[-3]
        # TODO(alexander-soare)：使用更安全的替代方法。
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "在不破坏权重的情况下，不能替换预训练模型中的BatchNorm！"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # 设置池化和最终层。
        # 使用干运行来获取特征图形状。
        # 虚拟输入应从`config.image_features`获取图像通道数，如果提供了`config.crop_shape`，
        # 则应使用其高度和宽度，否则应使用`config.image_features`的高度和宽度。

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
            x: (B, C, H, W) 像素值在[0, 1]范围内的图像张量。
        返回:
            (B, D) 图像特征。
        """
        # 预处理：可能裁剪（如果在__init__中设置）。
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # 对于评估始终使用中心裁剪。
                x = self.center_crop(x)
        # 提取骨干特征。
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # 带非线性的最终线性层。
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    参数:
        root_module: 需要替换子模块的模块
        predicate: 接受模块作为参数，如果该模块要被替换则必须返回True。
        func: 接受模块作为参数并返回一个新模块来替换它。
    返回:
        替换了子模块的根模块。
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # 验证所有BN都已被替换
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class VqVae(nn.Module):
    def __init__(
        self,
        config: VQBeTConfig,
    ):
        """
        VQ-VAE由三部分组成：编码器、vq_layer和解码器。
        编码器和解码器是由输入层、输出层和隐藏层组成的MLP。
        vq_layer使用残差VQ。

        此类包含用于训练编码器和解码器以及残差VQ层（用于训练阶段1）的函数，
        以及在训练阶段2中帮助BeT训练部分的函数。
        """

        super().__init__()
        self.config = config
        # 'discretized'表示残差VQ部分是否已训练。（训练完成后，我们设置discretized=True）
        self.register_buffer("discretized", torch.tensor(False))
        self.optimized_steps = 0
        # 我们在所有环境中使用固定数量的残差VQ层。
        self.vqvae_num_layers = 2

        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=self.vqvae_num_layers,
            codebook_size=config.vqvae_n_embed,
        )

        self.encoder = MLP(
            in_channels=self.config.action_feature.shape[0] * self.config.action_chunk_size,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                config.vqvae_embedding_dim,
            ],
        )
        self.decoder = MLP(
            in_channels=config.vqvae_embedding_dim,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                self.config.action_feature.shape[0] * self.config.action_chunk_size,
            ],
        )

    def get_embeddings_from_code(self, encoding_indices):
        # 此函数获取代码索引作为输入，并输出对应于代码索引的嵌入向量。
        with torch.no_grad():
            z_embed = self.vq_layer.get_codebook_vector_from_indices(encoding_indices)
            # 由于RVQ有多层，它在层的轴上添加向量以为该代码组合提供一个向量。
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        # 给定潜在向量，此函数输出解码的动作。
        output = self.decoder(latent)
        if self.config.action_chunk_size == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.action_feature.shape[0])
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.action_feature.shape[0])

    def get_code(self, state):
        # 在VQ-BeT训练的阶段2中，我们需要`动作数据的真实标签`来计算代码预测头的Focal损失。（请参阅论文的3.3节 https://huggingface.co/papers/2403.03181）
        # 此函数使用冻结的编码器和量化层输出给定动作的`GT代码`。（请参阅论文中的图2 https://huggingface.co/papers/2403.03181）
        state = einops.rearrange(state, "N T A -> N (T A)")
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            return state_vq, vq_code

    def vqvae_forward(self, state):
        # 此函数将给定的数据通过带有编码器和解码器的残差VQ传递。请参阅论文的3.2节（https://huggingface.co/papers/2403.03181）。
        state = einops.rearrange(state, "N T A -> N (T A)")
        # 我们首先通过编码器ϕ传递动作（或动作块）at:t+n。
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        # 然后通过最近邻查找将结果潜在嵌入向量x = ϕ(at:t+n)映射到RVQ层码本中的嵌入向量。
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        # 由于RVQ有多层，它在层的轴上添加向量以为该代码组合提供一个向量。
        vq_loss_state = torch.sum(vq_loss_state)
        # 然后，离散化向量zq(x)通过解码器ψ重建为ψ(zq(x))。
        dec_out = self.decoder(state_vq)
        # 计算L1重建损失
        encoder_loss = (state - dec_out).abs().mean()
        # 添加编码器重建损失和承诺损失
        rep_loss = encoder_loss + vq_loss_state * 5

        metric = (
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            vq_code,
            rep_loss.item(),
        )
        return rep_loss, metric


class FocalLoss(nn.Module):
    """
    来自 https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(N, T, 1)).view(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1]))

        super().__init__(*layers)
