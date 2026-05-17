# LeRobot 中文文档 / LeRobot Chinese Documentation

[English](#english) | [中文](#中文)

---

## 中文

LeRobot 是一个基于 PyTorch 的真实世界机器人库，提供数据集、预训练策略以及用于训练、评估、数据收集和机器人控制的工具。

### 📚 文档

本仓库提供完整的中英文双语文档：

- **在线文档**: [GitHub Pages](https://your-username.github.io/lerobot-zh/) (待部署)
- **英文文档**: `docs/source/en/`
- **中文文档**: `docs/source/zh/`

### 🌐 语言切换

文档支持自动语言检测和切换：

- 访问在线文档时，系统会根据浏览器语言自动跳转到对应语言版本
- 手动访问：
  - 英文: `https://your-username.github.io/lerobot-zh/en/`
  - 中文: `https://your-username.github.io/lerobot-zh/zh/`

### 📖 本地预览文档

```bash
# 安装依赖
pip install -e .
pip install -r docs-requirements.txt

# 预览英文文档
doc-builder preview lerobot docs/source/en

# 预览中文文档
doc-builder preview lerobot docs/source/zh
```

文档将在 [http://localhost:3000](http://localhost:3000) 可访问。

### 🔧 构建文档

```bash
# 构建英文文档
doc-builder build lerobot docs/source/en --build_dir ~/tmp/docs-en

# 构建中文文档
doc-builder build lerobot docs/source/zh --build_dir ~/tmp/docs-zh
```

### 🤝 贡献翻译

我们欢迎社区贡献更多的中文翻译！

**当前翻译进度**:

- ✅ 核心文档: 7/82 (8.5%)
  - index.mdx - 文档主页
  - installation.mdx - 安装指南
  - hardware_guide.mdx - 硬件指南
  - cheat-sheet.mdx - 速查表
  - cameras.mdx - 摄像头配置
  - act.mdx - ACT 策略
  - smolvla.mdx - SmolVLA 策略

**如何贡献**:

1. Fork 本仓库
2. 在 `docs/source/zh/` 目录下翻译文档
3. 确保文件名与 `docs/source/en/` 中的对应文件一致
4. 提交 Pull Request

**翻译指南**:

- 保持原文档的格式和结构
- 代码示例和命令保持英文
- 专业术语首次出现时可标注英文原文
- 参考已翻译的文档保持术语一致性

### 📋 文档结构

```
docs/
├── source/
│   ├── _config.py          # 文档构建配置
│   ├── en/                 # 英文文档
│   │   ├── _toctree.yml    # 英文目录结构
│   │   ├── index.mdx
│   │   └── ...
│   └── zh/                 # 中文文档
│       ├── _toctree.yml    # 中文目录结构
│       ├── index.mdx
│       └── ...
└── _build/                 # 构建输出（自动生成）
```

### 🚀 部署

文档通过 GitHub Actions 自动部署到 GitHub Pages：

- 推送到 `main` 分支时自动触发构建
- 构建完成后自动部署到 GitHub Pages
- 支持中英文双语版本

### 📝 许可证

本项目遵循 Apache License 2.0 许可证。

---

## English

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control.

### 📚 Documentation

This repository provides complete bilingual documentation in English and Chinese:

- **Online Documentation**: [GitHub Pages](https://your-username.github.io/lerobot-zh/) (pending deployment)
- **English Documentation**: `docs/source/en/`
- **Chinese Documentation**: `docs/source/zh/`

### 🌐 Language Switching

The documentation supports automatic language detection and switching:

- When visiting the online documentation, the system automatically redirects to the corresponding language version based on browser language
- Manual access:
  - English: `https://your-username.github.io/lerobot-zh/en/`
  - Chinese: `https://your-username.github.io/lerobot-zh/zh/`

### 📖 Preview Documentation Locally

```bash
# Install dependencies
pip install -e .
pip install -r docs-requirements.txt

# Preview English documentation
doc-builder preview lerobot docs/source/en

# Preview Chinese documentation
doc-builder preview lerobot docs/source/zh
```

Documentation will be accessible at [http://localhost:3000](http://localhost:3000).

### 🔧 Build Documentation

```bash
# Build English documentation
doc-builder build lerobot docs/source/en --build_dir ~/tmp/docs-en

# Build Chinese documentation
doc-builder build lerobot docs/source/zh --build_dir ~/tmp/docs-zh
```

### 🤝 Contributing Translations

We welcome community contributions for more Chinese translations!

**Current Translation Progress**:

- ✅ Core Documentation: 7/82 (8.5%)
  - index.mdx - Documentation homepage
  - installation.mdx - Installation guide
  - hardware_guide.mdx - Hardware guide
  - cheat-sheet.mdx - Cheat sheet
  - cameras.mdx - Camera configuration
  - act.mdx - ACT policy
  - smolvla.mdx - SmolVLA policy

**How to Contribute**:

1. Fork this repository
2. Translate documents in the `docs/source/zh/` directory
3. Ensure filenames match corresponding files in `docs/source/en/`
4. Submit a Pull Request

**Translation Guidelines**:

- Maintain the format and structure of the original document
- Keep code examples and commands in English
- Technical terms can be annotated with English originals on first appearance
- Refer to already translated documents to maintain terminology consistency

### 📋 Documentation Structure

```
docs/
├── source/
│   ├── _config.py          # Documentation build configuration
│   ├── en/                 # English documentation
│   │   ├── _toctree.yml    # English table of contents
│   │   ├── index.mdx
│   │   └── ...
│   └── zh/                 # Chinese documentation
│       ├── _toctree.yml    # Chinese table of contents
│       ├── index.mdx
│       └── ...
└── _build/                 # Build output (auto-generated)
```

### 🚀 Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions:

- Automatically triggered on push to `main` branch
- Automatically deployed to GitHub Pages after build completion
- Supports bilingual versions in English and Chinese

### 📝 License

This project is licensed under the Apache License 2.0.
