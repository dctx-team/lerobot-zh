# 贡献指南

感谢您对 LeRobot 中文文档项目的关注！我们欢迎所有形式的贡献。

## 🎯 贡献方式

### 1. 报告问题

如果您发现以下问题，请创建 Issue：
- 翻译错误或不准确
- 术语翻译不当
- 格式问题
- 链接失效
- 文档内容过时

**创建 Issue 时请包含**:
- 问题所在的文件名和行号
- 问题的详细描述
- 如果可能，提供改进建议

### 2. 改进翻译

如果您想改进现有翻译：

1. **Fork 本仓库**
2. **创建分支**
   ```bash
   git checkout -b improve/document-name
   ```
3. **编辑文档**
   - 文档位于 `docs/source/` 目录
   - 保持 MDX 格式和原有结构
   - 确保技术术语翻译准确
4. **提交更改**
   ```bash
   git add .
   git commit -m "改进 xxx.mdx 的翻译"
   ```
5. **推送并创建 Pull Request**
   ```bash
   git push origin improve/document-name
   ```

### 3. 翻译新文档

当官方 LeRobot 添加新文档时：

1. 在 Issue 中声明您要翻译哪个文档（避免重复工作）
2. 参考现有文档的翻译风格和术语
3. 翻译完成后提交 Pull Request
4. 更新 `docs/source/_toctree.yml` 添加新文档

### 4. 同步更新

当官方文档更新时：

1. 检查官方文档的变更
2. 更新对应的中文文档
3. 在 Pull Request 中说明更新内容

## 📋 翻译规范

### 术语翻译

请参考以下术语表保持翻译一致性：

| English | 中文 |
|---------|------|
| Imitation Learning | 模仿学习 |
| Reinforcement Learning | 强化学习 |
| Policy | 策略 |
| Robot | 机器人 |
| Simulation | 仿真 |
| Dataset | 数据集 |
| Processor | 处理器 |
| Teleoperation | 遥操作 |
| Episode | 情节/回合 |
| Frame | 帧 |
| Action | 动作 |
| Observation | 观测 |
| Trajectory | 轨迹 |

### 格式要求

1. **保持 MDX 格式**
   - 不要修改代码块
   - 保留所有 HTML/JSX 标签
   - 保持图片链接

2. **代码注释**
   - 代码示例中的注释可以翻译
   - 变量名和函数名保持英文

3. **链接处理**
   - 保持指向官方资源的链接不变
   - 内部链接确保指向正确的文件

4. **格式美化**
   - 中英文之间添加空格（可选）
   - 保持段落结构清晰

### 翻译原则

1. **准确性优先** - 确保技术内容准确无误
2. **保持原意** - 不要过度意译
3. **通俗易懂** - 使用易于理解的中文表达
4. **专业术语** - 使用业界通用的中文术语
5. **完整性** - 不要遗漏任何内容

## ✅ Pull Request 检查清单

提交 PR 前请确认：

- [ ] 翻译内容准确
- [ ] 保持了原有的 MDX 格式
- [ ] 代码块和链接完整
- [ ] 术语翻译与现有文档一致
- [ ] 没有语法或拼写错误
- [ ] 文件名与原项目一致
- [ ] 如有需要，更新了 `_toctree.yml`

## 🔄 工作流程

```
1. 创建 Issue 声明工作
        ↓
2. Fork 并创建分支
        ↓
3. 进行翻译/修改
        ↓
4. 自我审查
        ↓
5. 提交 Pull Request
        ↓
6. 代码审查
        ↓
7. 合并到主分支
```

## 📚 资源

- **原项目**: https://github.com/huggingface/lerobot
- **官方文档**: https://huggingface.co/docs/lerobot
- **MDX 语法**: https://mdxjs.com/

## ❓ 需要帮助？

如有任何疑问，请：
- 在 Issue 中提问
- 在 Discussions 中讨论
- 查看已有的 Pull Request 作为参考

## 🙏 感谢

感谢每一位贡献者！您的努力让更多中文用户能够学习和使用 LeRobot。

---

**记住**：无论贡献大小，都是对社区的宝贵帮助！
