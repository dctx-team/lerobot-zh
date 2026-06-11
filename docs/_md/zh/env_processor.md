# 环境处理器

环境处理器是 LeRobot 数据处理架构中的关键层，处理**环境特定**的转换，与策略特定的处理分离。这种关注点分离实现了更清晰的代码、更好的模块化以及更容易地使用不同环境和策略进行实验。

## 为什么需要环境处理器？

在使用不同的机器人环境（LIBERO、MetaWorld、Aloha 等）时，每个环境通常具有独特的数据格式、坐标系统和约定，需要在**策略处理之前**进行标准化。如果没有环境处理器，这些转换将是：

1. **硬编码在环境代码中** - 使得难以尝试不同的状态表示
2. **在策略之间重复** - 每个策略都需要处理环境特定的怪癖
3. **与策略逻辑混合** - 违反关注点分离并使调试更困难

环境处理器通过在原始环境观测和策略输入之间提供**专用处理层**来解决这个问题。

## 处理管道

以下是评估期间数据如何流经完整处理管道的方式：

```python
# 在 lerobot_eval.py rollout() 函数中：

# 1. 原始环境观测（numpy 数组，各种格式）
raw_observation = env.step(action)

# 2. 将 numpy 转换为 torch，归一化图像 [0,1]
observation = preprocess_observation(raw_observation)

# 3. 添加任务元数据（用于多任务环境）
observation = add_envs_task(env, observation)

# 4. 环境特定预处理（新功能！）
#    - 展平机器人状态
#    - 旋转图像以匹配数据集约定
#    - 处理环境特定的坐标系统
observation = env_preprocessor(observation)

# 5. 策略特定预处理
#    - 使用数据集统计信息归一化
#    - 添加批次维度
#    - 移动到 GPU
#    - 标记化语言指令
observation = preprocessor(observation)

# 6. 策略推理
action = policy.select_action(observation)

# 7. 策略特定后处理
#    - 反归一化动作
#    - 移除批次维度
action = postprocessor(action)

# 8. 环境特定后处理（新功能！）
#    - 如果需要转换动作格式
#    - 应用环境特定约束
action_transition = {"action": action}
action_transition = env_postprocessor(action_transition)
action = action_transition["action"]

# 9. 在环境中执行
env.step(action)
```

## 优势

### 1. **关注点分离**

环境处理器处理特定于**环境数据格式**的转换，而策略处理器处理特定于**模型要求**的转换。

```python
# ❌ 之前：混合关注点
class LiberoVLAPolicy:
    def preprocess(self, obs):
        # 环境特定：展平机器人状态（不应该在策略中！）
        state = self._flatten_robot_state(obs["robot_state"])
        # 策略特定：使用数据集统计信息归一化
        state = self.normalizer(state)
        return state

# ✅ 之后：清晰分离
# 环境处理器：处理 LIBERO 的嵌套机器人状态
env_preprocessor = LiberoProcessorStep()  # 展平 robot_state

# 策略处理器：处理模型要求
policy_preprocessor = NormalizerProcessorStep(stats=dataset_stats)
```

### 2. **灵活性和可重用性**

相同的策略可以与不同的环境处理器一起工作，相同的环境处理器可以与不同的策略一起工作。

### 3. **更容易的实验**

想要为 LIBERO 尝试不同的状态表示？只需创建一个新的处理器。

### 4. **更清晰的环境代码**

环境暴露**所有可用数据**，而无需知道下游模型将使用什么。

## 使用环境处理器

### 工厂函数

`make_env_pre_post_processors` 函数遵循与策略的 `make_pre_post_processors` 相同的模式：

```python
from lerobot.envs import make_env_pre_post_processors

# 对于 LIBERO：在预处理器中返回 LiberoProcessorStep
env_preprocessor, env_postprocessor = make_env_pre_post_processors(libero_cfg)

# 对于其他环境：返回恒等处理器（无操作）
env_preprocessor, env_postprocessor = make_env_pre_post_processors(pusht_cfg)
```

## 最佳实践

1. **保持环境处理器简单**：它们应该只处理环境特定的数据格式问题，而不是复杂的学习相关转换。

2. **使用策略处理器处理模型要求**：归一化、批处理、设备放置和标记化属于策略处理器。

3. **从环境暴露所有数据**：让处理器决定使用什么，而不是在环境中硬编码选择。

4. **记录约定**：清楚地记录处理器处理的任何坐标系统约定、相机方向或数据格式。

5. **独立测试**：环境处理器应该可以在不加载完整策略或环境的情况下进行测试。

## 总结

环境处理器在环境特定的数据转换和策略特定的模型要求之间提供了**清晰的分离**。这种架构：

- ✅ 能够轻松尝试不同的状态表示
- ✅ 允许策略在不同环境中无缝工作
- ✅ 保持环境代码专注于模拟/硬件接口
- ✅ 使处理器管道更易于维护和调试
- ✅ 遵循单一职责原则

关键见解：**环境定义数据格式，处理器标准化它们，策略消费标准化数据。**每一层都有清晰、专注的职责。
