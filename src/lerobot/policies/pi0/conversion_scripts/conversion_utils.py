# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和
# 限制的特定语言。

from transformers import GemmaConfig, PaliGemmaConfig


def get_paligemma_config(precision: str):
    config = {
        "image_token_index": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }

    # image_sizes = {"2b-test": 224, "3b-224px": 224, "3b-448px": 448, "3b-896px": 896}

    image_size = 224  # image_sizes[variant] 图像尺寸[变体]
    patch_size = 14
    num_image_tokens = (image_size**2) // (patch_size**2)

    config["image_token_index"] = 257152
    text_config = {
        "vocab_size": 257152,
        "num_hidden_layers": 18,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "torch_dtype": precision,
        "hidden_size": 2048,
        "hidden_activation": "gelu_pytorch_tanh",
        "num_attention_heads": 8,
        "intermediate_size": 16384,
        "is_encoder_decoder": False,
    }
    vision_config = {
        "torch_dtype": precision,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_image_tokens": num_image_tokens,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    }
    final_config = PaliGemmaConfig(text_config=text_config, vision_config=vision_config, **config)
    return final_config


def get_gemma_config(precision: str):
    config = {
        "image_token_index": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }

    config["image_token_index"] = 257152
    text_config = {
        "vocab_size": 257152,
        "num_hidden_layers": 18,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "torch_dtype": precision,
        "hidden_size": 1024,
        "hidden_activation": "gelu_pytorch_tanh",
        "num_attention_heads": 8,
        "intermediate_size": 4096,
        "is_encoder_decoder": False,
    }
    final_config = GemmaConfig()
    final_config.update(text_config)
    return final_config
