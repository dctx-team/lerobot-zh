# VLA-JEPA

本仓库包含 **VLA-JEPA** 的 LeRobot 移植版本，这是一个视觉-语言-动作模型，将 Qwen3-VL 语言骨干网络与自监督视频世界模型（V-JEPA2）以及基于流匹配的 DiT 动作头相结合。

转换自 [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA)。

---

## 架构概述

| 组件                    | 模块                              | 作用                                        |
| ----------------------- | --------------------------------- | ------------------------------------------- |
| **Qwen3-VL 骨干网络**   | `Qwen3VLInterface`                | 将图像与语言指令融合为上下文令牌             |
| **DiT-B 动作头**        | `VLAJEPAActionHead`               | 对动作块进行基于流匹配的扩散                 |
| **V-JEPA2 世界模型**    | `ActionConditionedVideoPredictor` | 自监督视频预测损失（仅训练阶段使用）         |

推理时仅使用 Qwen 骨干网络和动作头；世界模型在推理时不需要。

---

## 引用

```bibtex
@misc{sun2026vlajepaenhancingvisionlanguageactionmodel,
  title         = {VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author        = {Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
  year          = {2026},
  eprint        = {2602.10098},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2602.10098},
}
```

---

## 许可证

权重依据原始 [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) 仓库的许可证条款（**Apache 2.0 许可证**）分发。LeRobot 集成代码遵循 **Apache 2.0 许可证**。
