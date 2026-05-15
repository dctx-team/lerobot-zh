本文件为 AI 代理在此仓库中工作时提供指导。

> **面向用户的帮助 → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 设置、录制、选择策略、训练时长、评估 — 包含可复制粘贴的命令)。

## 项目概述

LeRobot 是一个基于 PyTorch 的真实世界机器人库，提供数据集、预训练策略以及用于训练、评估、数据收集和机器人控制的工具。它与 Hugging Face Hub 集成以实现模型/数据集共享。

## 技术栈

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (配置/CLI) · Gymnasium (环境) · uv (包管理)

## 开发环境设置

```bash
uv sync --locked                            # 基础依赖
uv sync --locked --extra test --extra dev   # 测试 + 开发工具
uv sync --locked --extra all                # 所有依赖
git lfs install && git lfs pull             # 测试工件
```

## 关键命令

```bash
uv run pytest tests -svv --maxfail=10                 # 所有测试
DEVICE=cuda make test-end-to-end                      # 所有端到端测试
pre-commit run --all-files                           # 代码检查 + 格式化 (ruff, typos, bandit 等)
```

## 架构 (`src/lerobot/`)

- **`scripts/`** — CLI 入口点 (`lerobot-train`, `lerobot-eval`, `lerobot-record` 等)，在 `pyproject.toml [project.scripts]` 中映射。
- **`configs/`** — 由 draccus 解析的数据类配置。`train.py` 有 `TrainPipelineConfig` (顶层)。`policies.py` 有 `PreTrainedConfig` 基类。通过 `draccus.ChoiceRegistry` 和 `@register_subclass("name")` 装饰器实现多态。
- **`policies/`** — 每个策略在其自己的子目录中。所有策略都继承自 `pretrained.py` 中的 `PreTrainedPolicy` (`nn.Module` + `HubMixin`)。`factory.py` 中的工厂使用延迟导入。
- **`processor/`** — 数据转换管道。`ProcessorStep` 基类带有注册表。`DataProcessorPipeline` / `PolicyProcessorPipeline` 链式步骤。
- **`datasets/`** — `LeRobotDataset` (片段感知采样 + 视频解码) 和 `LeRobotDatasetMetadata`。
- **`envs/`** — `configs.py` 中的 `EnvConfig` 基类，`factory.py` 中的工厂。每个环境子类定义 `gym_kwargs` 和 `create_envs()`。
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — 硬件抽象层。
- **`types.py`** 和 **`configs/types.py`** — 核心类型别名和特征类型定义。

## 仓库结构 (`src/` 之外)

- **`tests/`** — 按模块组织的 Pytest 测试套件。`tests/fixtures/` 中的 fixtures，`tests/mocks/` 中的 mocks。硬件测试使用 `tests/utils.py` 中的跳过装饰器。端到端测试通过 `Makefile` 写入 `tests/outputs/`。
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (基础依赖，每个 PR), `full_tests.yml` (所有额外依赖 + E2E + GPU，批准后), `latest_deps_tests.yml` (每日锁文件升级), `security.yml` (TruffleHog), `release.yml` (标签上的 PyPI 发布)。
- **`docs/source/`** — HF 文档 (`.mdx` 文件)。每个策略的 README、硬件指南、教程。通过 `docs-requirements.txt` 和 CI 工作流单独构建。
- **`examples/`** — 按用例组织的最终用户教程和脚本（数据集创建、训练、硬件设置）。
- **`docker/`** — 用户 (`Dockerfile.user`) 和 CI (`Dockerfile.internal`) 的 Dockerfile。
- **`benchmarks/`** — 性能基准测试脚本。
- **根文件**: `pyproject.toml` (依赖、构建、工具配置的单一真实来源), `Makefile` (E2E 测试目标), `uv.lock`, `CONTRIBUTING.md` & `README.md` (一般信息)。

## 注意事项

- **Mypy 是渐进式的**: 仅对 `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport` 严格。修改这些模块时添加类型注释。
- **可选依赖**: 许多策略、环境和机器人位于额外依赖之后（例如 `lerobot[aloha]`）。可选包的新导入必须受保护或延迟加载。参见 `pyproject.toml [project.optional-dependencies]`。
- **视频解码**: 数据集可以将观测存储为视频文件。`LeRobotDataset` 处理帧提取，但测试需要安装 ffmpeg。
- **优先使用 `uv run`** 执行 Python 命令（而不是原始的 `python` 或 `pip`）。
