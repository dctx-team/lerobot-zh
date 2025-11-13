# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.utils.constants import OBS_IMAGE, REWARD


class ClassifierOutput:
    """分类器输出的包装器，包含额外的元数据。"""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        学习型空间嵌入的 PyTorch 实现

        参数:
            height: 输入特征的空间高度
            width: 输入特征的空间宽度
            channel: 输入通道数
            num_features: 输出嵌入维度数
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        空间嵌入的前向传播

        参数:
            features: 输入张量，形状为 [B, H, W, C] 或 [H, W, C](如果没有批次)
        返回:
            输出张量，形状为 [B, C*F] 或 [C*F](如果没有批次)
        """

        features = features.last_hidden_state

        original_shape = features.shape
        if features.dim() == 3:
            features = features.unsqueeze(0)  # 添加批次维度

        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, H, W, C, F]

        # 逐元素乘法和空间降维
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # 对 H,W 求和

        # 重塑以组合通道和特征维度
        output = output.view(output.size(0), -1)  # [B, C*F]

        # 移除批次维度
        if len(original_shape) == 3:
            output = output.squeeze(0)

        return output


class Classifier(PreTrainedPolicy):
    """基于预训练编码器构建的图像分类器。"""

    name = "reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # 设置编码器
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # 如果给定多模态模型，则提取视觉模型
        if hasattr(encoder, "vision_model"):
            logging.info("检测到多模态模型 - 仅使用视觉编码器")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # 从配置获取模型类型
        self.is_cnn = self.config.model_type == "cnn"

        # 对于 CNN，初始化骨干网络
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()

        # 从 input_features 中提取图像键
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        if self.is_cnn:
            self.encoders = nn.ModuleDict()
            for image_key in self.image_keys:
                encoder = self._create_single_encoder()
                self.encoders[image_key] = encoder

        self._build_classifier_head()

    def _setup_cnn_backbone(self):
        """设置 CNN 编码器"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # 最后的通道维度
        else:
            raise ValueError("不支持的 CNN 架构")

    def _freeze_encoder(self) -> None:
        """冻结编码器参数。"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_single_encoder(self):
        encoder = nn.Sequential(
            self.encoder,
            SpatialLearnedEmbeddings(
                height=4,
                width=4,
                channel=self.feature_dim,
                num_features=self.config.image_embedding_pooling_dim,
            ),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.feature_dim * self.config.image_embedding_pooling_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.Tanh(),
        )

        return encoder

    def _build_classifier_head(self) -> None:
        """初始化分类器头部架构。"""
        # 根据模型类型获取输入维度
        if self.is_cnn:
            input_dim = self.config.latent_dim
        else:  # Transformer 模型
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("不支持的 Transformer 架构，因为未找到 hidden_size")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * self.config.num_cameras, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """从编码器中提取适当的输出。"""
        with torch.no_grad():
            if self.is_cnn:
                # HF ResNet 内部应用池化
                outputs = self.encoders[image_key](x)
                return outputs
            else:  # Transformer 模型
                outputs = self.encoder(x)
                return outputs.last_hidden_state[:, 0, :]

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor]:
        """从批次中提取图像张量和标签张量。"""
        # 检查 OBS_IMAGE 和 OBS_IMAGES 前缀
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        labels = batch[REWARD]

        return images, labels

    def predict(self, xs: list) -> ClassifierOutput:
        """分类器推理的前向传播。"""
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
        )
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """与 train.py 兼容的训练标准前向传播。"""
        # 提取图像和标签
        images, labels = self.extract_images_and_labels(batch)

        # 获取预测
        outputs = self.predict(images)

        # 计算损失
        if self.config.num_classes == 2:
            # 二分类
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # 多分类
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        # 计算准确率用于日志记录
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # 返回损失和指标用于日志记录
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch, threshold=0.5):
        """评估方法。返回预测的奖励，决策阈值作为参数。"""
        # 检查 OBS_IMAGE 和 OBS_IMAGES 前缀
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # 从批次字典中提取图像
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        if self.config.num_classes == 2:
            probs = self.predict(images).probabilities
            logging.debug(f"预测的奖励图像: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images).probabilities, dim=1)

    def get_optim_params(self):
        """返回策略的优化器参数。"""
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        此方法由 PreTrainedPolicy 要求，但不用于奖励分类器。
        奖励分类器不是一个演员，不会选择动作。
        """
        raise NotImplementedError("奖励分类器不选择动作")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        此方法由 PreTrainedPolicy 要求，但不用于奖励分类器。
        奖励分类器不是一个演员，不会生成动作块。
        """
        raise NotImplementedError("奖励分类器不预测动作块")

    def reset(self):
        """
        此方法由 PreTrainedPolicy 要求，但不用于奖励分类器。
        奖励分类器不是一个演员，不会选择动作。
        """
        pass
