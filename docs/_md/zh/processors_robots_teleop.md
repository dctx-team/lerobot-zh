# 机器人和遥操作器的处理器

本指南展示如何构建和修改连接遥操作器（例如手机）到机器人和数据集的处理管道。管道标准化不同动作/观测空间之间的转换，因此你可以交换遥操作器和机器人而无需重写粘合代码。

我们使用手机到 SO-100 跟随器示例以具体说明，但相同的模式适用于其他机器人。

**你将学到什么**

- 绝对与相对末端执行器控制：每种的含义、权衡以及如何为你的任务选择。
- 三管道模式：如何映射遥操作动作 → 数据集动作 → 机器人命令，以及机器人观测 → 数据集观测。
- 适配器（`to_transition` / `to_output`）：这些如何将原始字典转换为 `EnvTransition` 并返回以减少样板代码。
- 数据集特征契约：步骤如何通过 `transform_features(...)` 声明特征，以及如何聚合/合并它们以进行记录。
- 选择表示：何时存储关节、绝对末端执行器姿态或相对末端执行器增量——以及这如何影响训练。
- 管道自定义指导：如何安全地交换机器人/URDF 并调整边界、步长和 IK 初始化等选项。

### 绝对与相对末端执行器控制

本指南中的示例使用绝对末端执行器（EE）姿态，因为它们易于推理。在实践中，相对 EE 增量或关节位置通常作为学习特征更受欢迎。

使用处理器，你可以选择要用于策略的学习特征。这可以是关节位置/速度、绝对 EE 或相对 EE 位置。你还可以选择存储其他特征，如关节扭矩、电机电流等。

## 三个管道

我们经常组合三个管道。根据你的设置，如果动作和观测空间已经匹配，某些管道可以为空。
这些管道中的每一个都处理不同动作和观测空间之间的不同转换。以下是每个管道的快速说明。

1. 管道 1：遥操作动作空间 → 数据集动作空间（手机姿态 → EE 目标）
2. 管道 2：数据集动作空间 → 机器人命令空间（EE 目标 → 关节）
3. 管道 3：机器人观测空间 → 数据集观测空间（关节 → EE 姿态）

以下是我们在手机到 SO-100 跟随器示例中使用的三个管道的示例：

```python
phone_to_robot_ee_pose_processor = RobotProcessorPipeline[RobotAction, RobotAction]( # 遥操作 -> 数据集动作
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        EEReferenceAndDelta(
            kinematics=kinematics_solver, end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5}, motor_names=list(robot.bus.motors.keys()),
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}, max_ee_step_m=0.20,
        ),
        GripperVelocityToJoint(),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

robot_ee_to_joints_processor = RobotProcessorPipeline[RobotAction, RobotAction]( # 数据集动作 -> 机器人
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()), initial_guess_current_joints=True,
        ),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

robot_joints_to_ee_pose = RobotProcessorPipeline[RobotObservation, RobotObservation]( # 机器人观测 -> 数据集观测
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)
```

## 为什么使用 to_transition / to_output

为了从机器人/遥操作器转换到管道并返回，我们使用 `to_transition` 和 `to_output` 管道适配器。
它们标准化转换以减少样板代码，并在机器人和遥操作器的原始字典与管道的 `EnvTransition` 格式之间形成桥梁。
在手机到 SO-100 跟随器示例中，我们使用以下适配器：

- `robot_action_to_transition`：将遥操作动作字典转换为管道转换。
- `transition_to_robot_action`：将管道转换转换为机器人动作字典。
- `observation_to_transition`：将机器人观测字典转换为管道转换。
- `transition_to_observation`：将管道转换转换为观测字典。

查看 [src/lerobot/processor/converters.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/converters.py) 了解更多详情。

## 数据集特征契约

数据集特征由保存在数据集中的键确定。每个步骤可以在称为 `transform_features(...)` 的契约中声明它修改的特征。一旦你构建了处理器，处理器就可以使用 `aggregate_pipeline_dataset_features()` 聚合所有这些特征，并使用 `combine_feature_dicts(...)` 合并多个特征字典。

以下是我们在手机到 SO-100 跟随器示例中如何使用 `transform_features` 方法声明特征的示例：

```python
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 我们只在数据集中使用 ee 姿态，所以我们不需要关节位置
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # 我们指定此步骤的数据集特征，我们希望将其存储在数据集中
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features
```

在这里，我们声明在此步骤中修改的 PolicyFeatures，因此我们知道运行处理器时可以期望什么特征。然后可以聚合这些特征并用于创建数据集特征。

以下是我们在手机到 SO-100 记录示例中如何聚合和合并特征的示例：

```python
features=combine_feature_dicts(
        # 运行管道的特征契约
        # 这告诉你管道步骤后特征的样子
        aggregate_pipeline_dataset_features(
            pipeline=phone_to_robot_ee_pose_processor,
            initial_features=create_initial_features(action=phone.action_features), # <- 我们可以期望的动作特征，这些来自我们的遥操作设备（手机）和动作处理器
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_joints_to_ee_pose,
            initial_features=create_initial_features(observation=robot.observation_features), # <- 我们可以期望的观测特征，这些来自我们的机器人和观测处理器
            use_videos=True,
            patterns=["observation.state.ee"], # <- 在这里你可以选择性地过滤我们想要存储在数据集中的特征，使用特定模式

        ),
    ),
```

工作原理：

- `aggregate_pipeline_dataset_features(...)`：在管道中应用 `transform_features` 并按模式过滤（当 `use_videos=True` 时包含图像，当指定 `patterns` 时包含状态特征）。
- `combine_feature_dicts(...)`：组合多个特征字典。
- 使用 `record_loop(...)` 记录使用 `build_dataset_frame(...)` 构建与 `dataset.features` 一致的帧，然后我们调用 `add_frame(...)` 将帧添加到数据集。

## 自定义机器人管道时的指导

你可以将以下任何特征存储为你的动作/观测空间：

- 关节位置
- 绝对 EE 姿态
- 相对 EE 增量
- 其他特征：关节速度、扭矩等。

选择你想要用于策略动作和观测空间的内容，并相应地配置/修改管道和步骤。

### 不同的机器人

- 你可以轻松重用管道，例如要将另一个机器人与手机遥操作一起使用，修改示例并交换机器人 `RobotKinematics`（URDF）和 `motor_names` 以使用你自己的机器人与手机遥操作。此外，你应该确保 `target_frame_name` 指向你的夹爪/手腕。

### 安全第一

- 更改管道时，从严格的边界开始，在使用真实机器人时实施安全步骤。
- 建议首先从模拟开始，然后再转向真实机器人。

就是这样！我们希望本指南能帮助你开始自定义机器人管道，如果你在任何时候遇到任何问题，请加入我们的 [Discord 社区](https://discord.com/invite/s3KuuzsPFb) 寻求支持。
