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

import math
from collections.abc import Callable
from functools import partial
from math import ceil
from random import randrange

import torch
import torch.distributed as distributed
import torch.nn.functional as F  # noqa: N812
from einops import pack, rearrange, reduce, repeat, unpack
from torch import einsum, nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

# ruff: noqa: N806

"""
本文件是 VQ-BeT 的一部分，使用了以下代码库的代码：

    - Vector Quantize PyTorch 代码采用 MIT 许可证：
        原始来源：https://github.com/lucidrains/vector-quantize-pytorch

    - nanoGPT 部分是 Andrej Karpathy 的 nanoGPT 在 PyTorch 中实现的改编版本。
        原始来源：https://github.com/karpathy/nanoGPT

我们还对原始代码进行了一些修改以适应我们的需求。这些更改在下面的代码中有描述。
"""

"""
这是 nanoGPT 的一部分，使用了以下代码库的代码：

    - Andrej Karpathy 的 nanoGPT 在 PyTorch 中的实现。
        原始来源：https://github.com/karpathy/nanoGPT

    - nanoGPT 代码采用 MIT 许可证：

    MIT License

    Copyright (c) 2022 Andrej Karpathy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    - 我们对原始代码进行了一些修改以适应我们的需求。

        变量名称的更改：
            - n_head -> gpt_n_head
            - n_embd -> gpt_hidden_dim
            - block_size -> gpt_block_size
            - n_layer -> gpt_n_layer


        class GPT(nn.Module):
            - 删除了未使用的函数 `def generate`、`def estimate_mfu` 和 `def from_pretrained`
            - 将 `configure_optimizers` 更改为 `def configure_parameters`，使其仅返回模型的参数：我们在训练循环中使用外部优化器。
            - 在函数 `forward` 中，我们删除了目标损失计算部分，因为它将在训练循环中计算（在通过 bin 预测和 offset 预测头之后）。

"""


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.gpt_hidden_dim % config.gpt_n_head == 0
        # 对所有头的 key、query、value 投影，但以批处理方式
        self.c_attn = nn.Linear(config.gpt_hidden_dim, 3 * config.gpt_hidden_dim)
        # 输出投影
        self.c_proj = nn.Linear(config.gpt_hidden_dim, config.gpt_hidden_dim)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 因果掩码以确保注意力仅应用于输入序列的左侧
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.gpt_block_size, config.gpt_block_size)).view(
                1, 1, config.gpt_block_size, config.gpt_block_size
            ),
        )
        self.gpt_n_head = config.gpt_n_head
        self.gpt_hidden_dim = config.gpt_hidden_dim

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # 批大小，序列长度，嵌入维度（gpt_hidden_dim）

        # 计算批中所有头的 query、key、values，并将头向前移动成为批维度
        q, k, v = self.c_attn(x).split(self.gpt_hidden_dim, dim=2)
        k = k.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 将所有头的输出并排重新组装

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    # GPT 的因果自注意力块
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.gpt_hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.gpt_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.gpt_hidden_dim, 4 * config.gpt_hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.gpt_hidden_dim, config.gpt_hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    原始注释：
    GPT 语言模型的完整定义，全部在这个单一文件中。
    参考：
    1) OpenAI 发布的官方 GPT-2 TensorFlow 实现：
    https://github.com/openai/gpt-2/blob/master/src/model.py
    2) huggingface/transformers PyTorch 实现：
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """

    def __init__(self, config: VQBeTConfig):
        """
        GPT 模型从配置对象获取超参数。更多详情请参阅 configuration_vqbet.py。
        """
        super().__init__()
        assert config.gpt_output_dim is not None
        assert config.gpt_block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Linear(config.gpt_input_dim, config.gpt_hidden_dim),
                "wpe": nn.Embedding(config.gpt_block_size, config.gpt_hidden_dim),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.gpt_n_layer)]),
                "ln_f": nn.LayerNorm(config.gpt_hidden_dim),
            }
        )
        self.lm_head = nn.Linear(config.gpt_hidden_dim, config.gpt_output_dim, bias=False)
        # 初始化所有权重，并根据 GPT-2 论文对残差投影应用特殊的缩放初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.gpt_n_layer))

        # 报告参数数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(self, input, targets=None):
        device = input.device
        b, t, d = input.size()
        assert t <= self.config.gpt_block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.gpt_block_size}"
        )

        # 添加到输入嵌入的位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # 形状 (1, t)

        # 前向传播 GPT 模型本身
        tok_emb = self.transformer.wte(input)  # 形状为 (b, t, gpt_hidden_dim) 的token嵌入
        pos_emb = self.transformer.wpe(pos)  # 形状为 (1, t, gpt_hidden_dim) 的位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, gpt_block_size):
        # 如果需要，进行模型手术以减小块大小
        # 例如，我们可能会加载 GPT2 预训练模型检查点（块大小为 1024）
        # 但希望为一些更小、更简单的模型使用更小的块大小
        assert gpt_block_size <= self.config.gpt_block_size
        self.config.gpt_block_size = gpt_block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:gpt_block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :gpt_block_size, :gpt_block_size]

    def configure_parameters(self):
        """
        这个冗长的函数不幸地只是在做一件非常简单的事情，并且非常保守：
        我们将模型的所有参数分为两组：那些将经历权重衰减正则化的参数和那些不会的参数
        （偏置以及 layernorm/embedding 权重）。
        """

        # 将所有参数分为将会和不会经历正则化权重衰减的参数
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # 完整的参数名称
                if pn.endswith("bias"):
                    # 所有偏置不会被衰减
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将被权重衰减
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重不会被权重衰减
                    no_decay.add(fpn)

        # 验证我们考虑了每个参数
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(
            str(inter_params)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters {} were not separated into either decay/no_decay set!".format(
                str(param_dict.keys() - union_params),
            )
        )

        decay = [param_dict[pn] for pn in sorted(decay)]
        no_decay = [param_dict[pn] for pn in sorted(no_decay)]
        # 分别返回需要权重衰减的参数和不需要的参数。
        return decay, no_decay


"""
本文件是残差向量量化的一部分，使用了以下代码库的代码：

    - Phil Wang 的 vector-quantize-pytorch 在 PyTorch 中的实现。
        原始来源：https://github.com/lucidrains/vector-quantize-pytorch

    - vector-quantize-pytorch 代码采用 MIT 许可证：

        MIT License

        Copyright (c) 2020 Phil Wang

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    - 我们对原始代码进行了一些修改以适应我们的需求。

        class ResidualVQ(nn.Module):
            - 在 __init__ 方法中添加了 `self.register_buffer('freeze_codebook', torch.tensor(False))`：
                这使用户能够保存码本是否被冻结的指示器。
            - 更改了函数名称 `get_codes_from_indices` → `get_codebook_vector_from_indices`：
                这是为了使函数名称更具描述性。

        class VectorQuantize(nn.Module):
            - 从 __init__ 方法中删除了 `use_cosine_sim` 和 `layernorm_after_project_in` 参数：
                这些参数在代码中未使用。
            - 更改了函数名称 `get_codes_from_indices` → `get_codebook_vector_from_indices`：
                这是为了使函数名称更具描述性。

"""


class ResidualVQ(nn.Module):
    """
    残差 VQ 由多个 VectorQuantize 层组成。

    遵循 https://huggingface.co/papers/2107.03312 中的算法 1。
        "残差向量量化器（也称为多阶段向量量化器 [36]）按如下方式级联 Nq 层 VQ。未量化的输入向量
        通过第一个 VQ，并计算量化残差。然后通过额外的 Nq-1 个向量量化器序列迭代量化残差，
        如算法 1 中所述。"


    self.project_in: 用于将输入投影到码本维度的函数
    self.project_out: 用于将码本维度投影到输出维度的函数
    self.layers: VectorQuantize 层的 nn.ModuleList，包含论文中描述的 Nq 层 VQ。
    self.freeze_codebook: 用于保存码本是否被冻结的指示器的缓冲区。VQ-BeT 将检查此项以确定是否更新码本。
    """

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim=None,
        shared_codebook=False,
        heads=1,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        accept_image_fmap=False,
        **kwargs,
    ):
        super().__init__()
        assert heads == 1, "residual vq is not compatible with multi-headed codes"
        codebook_dim = codebook_dim if (codebook_dim is not None) else dim
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=codebook_dim, codebook_dim=codebook_dim, accept_image_fmap=accept_image_fmap, **kwargs
                )
                for _ in range(num_quantizers)
            ]
        )

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.register_buffer("freeze_codebook", torch.tensor(False))
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec 论文提出结构化 dropout，认为这设置为 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        codebooks = rearrange(codebooks, "q 1 c d -> q c d")
        return codebooks

    def get_codebook_vector_from_indices(self, indices):
        # 此函数将返回对应于索引的所有层码本中的代码
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # 也可能接收形状为 'b h w q' 的索引（accept_image_fmap）

        indices, ps = pack([indices], "b * q")

        # 由于量化 dropout，可以传入粗糙的索引
        # 并且网络应该能够重建

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, (
                "如果希望从具有较少精细量化的信号重建，量化 dropout 必须大于 0"
            )
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # 准备收集

        codebooks = repeat(self.codebooks, "q c d -> q b c d", b=batch)
        gather_indices = repeat(indices, "b n q -> q b n d", d=codebooks.shape[-1])

        # 处理量化器 dropout

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(
            mask, 0
        )  # 让它获取一个虚拟代码，稍后将被掩码

        all_codes = codebooks.gather(2, gather_indices)  # 收集所有代码

        # 掩码掉任何被 dropout 的代码

        all_codes = all_codes.masked_fill(mask, 0.0)

        # 如果（accept_image_fmap = True）则返回形状 (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")

        return all_codes

    def forward(self, x, indices=None, return_all_codes=False, sample_codebook_temp=None):
        """
        对于给定的输入张量 x，此函数将返回量化输出、量化输出的索引和损失。
        首先，输入张量 x 被投影到码本维度。然后，输入张量 x 通过 Nq 层 VectorQuantize。
        每层的残差值被馈送到下一层。
        """
        num_quant, quant_dropout_multiple_of, return_loss, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            (indices is not None),
            x.device,
        )

        x = self.project_in(x)

        assert not (self.accept_image_fmap and (indices is not None))

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), (
                "某些残差 vq 索引被 dropout。请使用模块处于 eval 模式时派生的索引来派生交叉熵损失"
            )
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # 在该层索引处 dropout 进一步的残差量化
        # 同时准备空索引和损失

        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    ceil((rand_quantize_dropout_index + 1) / quant_dropout_multiple_of)
                    * quant_dropout_multiple_of
                    - 1
                )

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0.0, device=device, dtype=x.dtype)

        # 遍历各层

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=self.freeze_codebook,
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # 如有需要，进行输出投影

        quantized_out = self.project_out(quantized_out)

        # 是否提前返回交叉熵损失

        if return_loss:
            return quantized_out, sum(ce_losses)

        # 堆叠所有损失和索引

        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # 是否返回跨层所有码本的所有代码
            all_codes = self.get_codebook_vector_from_indices(all_indices)

            # 将以形状 (quantizer, batch, sequence length, codebook dimension) 返回所有代码
            ret = (*ret, all_codes)

        return ret


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
        straight_through=False,
        reinmax=False,  # 使用 reinmax 改进直通，假设直通在所有情况下都有帮助
        sync_codebook=None,
        sync_affine_param=False,
        ema_update=True,
        learnable_codebook=False,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # 如果使用 learnable_codebook，用于更新码本嵌入的优化器
        affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        sync_update_v=0.0,  # 控制同步更新规则 (21) 的乐观与悲观更新的 v 参数 https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = codebook_dim if (codebook_dim is not None) else dim
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # 是否使用交叉熵损失到码本作为承诺损失

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), "可学习码本与 EMA 更新不兼容"

        assert 0 <= sync_update_v <= 1.0
        assert not (sync_update_v > 0.0 and not learnable_codebook), "必须打开可学习码本"

        self.sync_update_v = sync_update_v

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through,
        )

        if sync_codebook is None:
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = {
            "dim": codebook_dim,
            "num_codebooks": heads if separate_codebook_per_head else 1,
            "codebook_size": codebook_size,
            "kmeans_init": kmeans_init,
            "kmeans_iters": kmeans_iters,
            "sync_kmeans": sync_kmeans,
            "decay": decay,
            "eps": eps,
            "threshold_ema_dead_code": threshold_ema_dead_code,
            "use_ddp": sync_codebook,
            "learnable_codebook": has_codebook_orthogonal_loss or learnable_codebook,
            "sample_codebook_temp": sample_codebook_temp,
            "gumbel_sample": gumbel_sample_fn,
            "ema_update": ema_update,
        }

        if affine_param:
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )

        self._codebook = EuclideanCodebook(**codebook_kwargs)

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.parameters())
            if (in_place_codebook_optimizer is not None)
            else None
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        self._codebook.embed.copy_(codes)

    def get_codebook_vector_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack_one(indices, "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        codes = unpack_one(codes, ps, "b * d")
        return codes

    def forward(
        self,
        x,
        indices=None,
        mask=None,
        sample_codebook_temp=None,
        freeze_codebook=False,
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert mask is None
            x = rearrange(x, "b d -> b 1 d")

        shape, device, heads, is_multiheaded, _codebook_size, return_loss = (
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            self.codebook_size,
            (indices is not None),
        )

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = self.in_place_codebook_optimizer is not None

        # 重排输入

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")

        if need_transpose:
            x = rearrange(x, "b d n -> b n d")

        # 投影输入

        x = self.project_in(x)

        # 处理多头分离码本

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        # 用于余弦相似度的 l2 范数，否则为恒等

        x = self._codebook.transform_input(x)

        # 码本前向 kwargs

        codebook_forward_kwargs = {
            "sample_codebook_temp": sample_codebook_temp,
            "mask": mask,
            "freeze_codebook": freeze_codebook,
        }

        # 量化

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # 一步就地更新

        if should_inplace_optimize and self.training and not freeze_codebook:
            if mask is not None:
                loss = F.mse_loss(quantize, x.detach(), reduction="none")

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(
                        mask,
                        "b n -> c (b h) n",
                        c=loss.shape[0],
                        h=loss.shape[1] // mask.shape[0],
                    )

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # 再次量化

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # 确定用于承诺损失的代码
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)

            # 直通

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.0:
                # https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf 中的 (21)
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # 用于计算距离矩阵的交叉熵损失的函数
        # 用于 (1) naturalspeech2 训练残差 vq 潜变量接近正确代码 和 (2) 基于交叉熵的承诺损失

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = "1 b n l -> b l n"
            elif self.separate_codebook_per_head:
                dist_einops_eq = "c b n l -> b l n c"
            else:
                dist_einops_eq = "1 (b h) n l -> b l n h"

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b=shape[0]), codes, ignore_index=-1
            )

            return ce_loss

        # 如果返回传入代码的交叉熵损失

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # 转换嵌入索引

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, "b (h w) ... -> b h w ...", h=height, w=width)

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        # 聚合损失

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if mask is not None:
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, "b n -> b n h", h=heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if mask is not None:
                        # 使用可变长度序列
                        commit_loss = F.mse_loss(commit_quantize, x, reduction="none")

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(
                                loss_mask,
                                "b n -> c (b h) n",
                                c=commit_loss.shape[0],
                                h=commit_loss.shape[1] // mask.shape[0],
                            )

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # 仅计算此批次激活代码的正交损失

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), (
                        "仅针对激活代码的正交正则化尚不兼容多头分离码本"
                    )
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if (self.orthogonal_reg_max_codes is not None) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[: self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # 处理多头量化嵌入

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)

        # 输出投影

        quantize = self.project_out(quantize)

        # 重排量化嵌入

        if need_transpose:
            quantize = rearrange(quantize, "b n d -> b d n")

        if self.accept_image_fmap:
            quantize = rearrange(quantize, "b (h w) c -> b c h w", h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # 如果掩码，仅返回掩码为 True 的量化结果

        if mask is not None:
            quantize = torch.where(rearrange(mask, "... -> ... 1"), quantize, orig_input)

        return quantize, embed_ind, loss


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), (
        "reinmax 只能在使用直通 gumbel softmax 时打开"
    )

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # 使用 reinmax 获得更好的二阶精度 - https://huggingface.co/papers/2304.08612
    # 算法 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=noop,
):
    num_codebooks, dim, dtype, _device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


def orthogonal_loss_fn(t):
    # 来自 https://huggingface.co/papers/2112.00384 的方程 (2)
    h, n = t.shape[:2]
    normed_codes = F.normalize(t, p=2, dim=-1)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        affine_param=False,
        sync_affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = (
            reset_cluster_size if (reset_cluster_size is not None) else threshold_ema_dead_code
        )

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), (
            "kmeans 初始化目前在分布式环境中与多个码本不兼容"
        )

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

        # 仿射相关参数

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer("batch_mean", None)
        self.register_buffer("batch_variance", None)

        self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
        self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_variance", torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if self.initted:
            return

        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not (old_value is not None) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask=None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased=False)

        # 计算码本均值和方差

        embed = rearrange(embed, "h ... d -> h (...) d")

        if self.training:
            self.update_with_decay(
                "codebook_mean",
                reduce(embed, "h n d -> h 1 d", "mean"),
                self.affine_param_codebook_decay,
            )
            self.update_with_decay(
                "codebook_variance",
                reduce(embed, "h n d -> h 1 d", var_fn),
                self.affine_param_codebook_decay,
            )

        # 准备批量数据，取决于是否有掩码

        data = rearrange(data, "h ... d -> h (...) d")

        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # 计算批量均值和方差

        if not self.sync_affine_param:
            self.update_with_decay(
                "batch_mean",
                reduce(data, "h n d -> h 1 d", "mean"),
                self.affine_param_batch_decay,
            )
            self.update_with_decay(
                "batch_variance",
                reduce(data, "h n d -> h 1 d", var_fn),
                self.affine_param_batch_decay,
            )
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # 向量数量，用于分母

        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        distributed.all_reduce(num_vectors)

        # 计算分布式均值

        batch_sum = reduce(data, "h n d -> h 1 d", "sum")
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay("batch_mean", batch_mean, self.affine_param_batch_decay)

        # 计算分布式方差

        variance_number = reduce((data - batch_mean) ** 2, "h n d -> h 1 d", "sum")
        distributed.all_reduce(variance_number)
        batch_variance = variance_number / num_vectors

        self.update_with_decay("batch_variance", batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0), strict=False)
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, "... -> 1 ..."), mask.sum().item())
            sampled = rearrange(sampled, "1 ... -> ...")

            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = (
            sample_codebook_temp if (sample_codebook_temp is not None) else self.sample_codebook_temp
        )

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")

        if mask is not None:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        self.init_embed_(flatten, mask=mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if mask is not None:
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(dim=1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = tuple(rearrange(t, "1 ... -> ...") for t in (quantize, embed_ind))

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist
