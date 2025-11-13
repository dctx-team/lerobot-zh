#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: 用于通用机器人控制的视觉-语言-动作流模型

[论文](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax代码](https://github.com/Physical-Intelligence/openpi)

由 Physical Intelligence 设计。由 Hugging Face 从 Jax 移植。
免责声明：预计性能不如原始实现。

安装 pi0 额外依赖项：
```bash
pip install -e ".[pi0]"
```

微调 pi0 预训练模型的示例（`openpi` 中的 `pi0_base`）：
```bash
lerobot-train \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

使用 PaliGemma 和 expert Gemma 微调 pi0 神经网络的示例，
在 pi0 微调之前使用 VLM 默认参数进行预训练：
```bash
lerobot-train \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

在 LeRobot 训练框架外使用 pi0 预训练模型的示例：
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """计算标量位置的正弦-余弦位置编码向量。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # 计算外积
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """从 big_vision 复制。

    token 可以关注累积 mask_ar 小于或等于其自身值的有效输入 token。
    这样 `mask_ar` int[B, N] 可以用于设置多种类型的注意力，例如：

      [[1 1 1 1 1 1]]：纯因果注意力。

      [[0 0 0 1 1 1]]：前缀-LM 注意力。前 3 个 token 可以相互关注，
          后 3 个 token 具有因果注意力。第一个条目也可以是 1 而不改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]：4 个块之间的因果注意力。一个块的 token
          可以关注所有先前的块和同一块上的所有 token。

    参数：
      input_mask: bool[B, N]，如果是输入的一部分则为 true，如果是填充则为 false。
      mask_ar: int32[B, N]，在先前 token 不能依赖它的地方为 1，
        在它与先前 token 共享相同注意力掩码的地方为 0。
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # 假设当宽度和高度已经适配时不执行操作
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # 在图像的左侧和顶部填充
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """可以是 (batch_size x sequence_length x features_dimension)
    或 (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # 这确保输入保持在 [−1,1] 范围内
    # 以避免 arcsin 的无效值
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha 将夹爪位置转换为线性空间。以下代码
    # 反转此转换以与在角度空间中预训练的 pi0 保持一致。
    #
    # 这些值来自 Aloha 代码：
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 这是 Interbotix 代码中角度到线性转换的逆变换。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # 常量取自 Interbotix 代码。
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # 归一化到 [0, 1]。
    # 值 0.4 和 1.5 是在实际的 Trossen 机器人上测量的。
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # 从 pi0 使用的夹爪位置转换为 Aloha 使用的夹爪位置。
    # 注意单位仍然是角度，但范围不同。

    # 值 0.4 和 1.5 是在实际的 Trossen 机器人上测量的。
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # 这些值来自 Aloha 代码：
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # 直接反转 gripper_from_angular 函数。
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class PI0Policy(PreTrainedPolicy):
    """PI0FlowMatching 模型的包装类，用于在 LeRobot 内训练和运行推理。"""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
    ):
        """
        参数：
            config: 策略配置类实例，或 None（在这种情况下使用配置类的默认实例化）。
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = PI0FlowMatching(config)

        self.reset()

    def reset(self):
        """每当环境重置时都应调用此方法。"""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """重写 from_pretrained 方法以显示重要免责声明。"""
        print(
            "⚠️  DISCLAIMER: The PI0 model is ported from JAX by the Hugging Face team. \n"
            "   It is not expected to perform as well as the original implementation. \n"
            "   Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        return super().from_pretrained(*args, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测预测一组动作。"""
        raise NotImplementedError("Currently not implemented for PI0")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """根据环境观测选择单个动作。

        此方法包装 `select_actions`，以便一次返回一个动作用于在环境中执行。
        它通过在队列中管理动作并仅在队列为空时调用 `select_actions` 来工作。
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # n_action_steps > 1 的动作队列逻辑。当 action_queue 耗尽时，
        # 通过查询策略来填充它。
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )

            # 去除动作的填充
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` 返回一个 (batch_size, n_action_steps, action_dim) 张量，
            # 但队列的有效形状是 (n_action_steps, batch_size, *)，因此需要转置。
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """执行完整的训练前向传递以计算损失"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # 移除填充
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # 用于反向传递
        loss = losses.mean()
        # 用于日志记录
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """对图像应用 Pi0 预处理，例如调整大小到 224x224 并填充以保持纵横比，
        并按照 SigLIP 的要求将像素范围从 [0.0, 1.0] 转换为 [-1.0, 1.0]。
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # 预处理批次中存在的图像特征
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # 按照 siglip 的预期从范围 [0,1] 归一化到 [-1,1]
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # 创建批次中不存在的图像特征
        # 作为完全 0 填充的图像。
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # 翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # 反转 Aloha 运行时应用的夹爪转换。
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # 翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # 反转 Aloha 运行时应用的夹爪转换。
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # 再次翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # 反转 Aloha 运行时应用的夹爪转换。
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """填充状态"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """填充动作"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class PI0FlowMatching(nn.Module):
    """
    π0: 用于通用机器人控制的视觉-语言-动作流模型

    [论文](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax代码](https://github.com/Physical-Intelligence/openpi)

    由 Physical Intelligence 设计。由 Hugging Face 从 Jax 移植。
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # 投影是 float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用 SigLIP 嵌入图像，使用嵌入层嵌入语言 token，
        为 PaliGemma 变换器处理做准备。
        """
        # TODO: 避免使用 Python 列表和 torch.cat；优先使用 torch.empty 进行预分配
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: 移除 for 循环
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # 归一化图像嵌入
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # 创建注意力掩码，使图像 token 可以相互关注
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # 归一化语言嵌入
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像和语言输入之间的完全注意力
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """嵌入状态、噪声动作、时间步，为 Expert Gemma 处理做准备。"""
        embs = []
        pad_masks = []
        att_masks = []

        # 嵌入状态
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # 设置注意力掩码，使图像和语言输入不关注状态或动作
        att_masks += [1]

        # 使用正弦-余弦位置编码嵌入时间步，灵敏度范围为 [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # 使用 MLP 融合时间步 + 动作信息
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu（swish 等同于 silu）
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # 添加到输入 token
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # 设置注意力掩码，使图像、语言和状态输入不关注动作 token
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """执行完整的训练前向传递并计算损失 (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # 原始 openpi 代码，向上转换注意力输出
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """执行完整的推理前向传递并计算动作 (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 计算图像和语言的键值缓存
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # 欧拉步
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """在给定的时间步对噪声 `x_t` 应用一次去噪步骤。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
