# Reachy 2

Reachy 2 是由 Pollen Robotics 制造的开源人形机器人，专为具身 AI 和真实世界应用的开发而设计。
查看 [Pollen Robotics 网站](https://www.pollen-robotics.com/reachy/)，或访问 [Reachy 2 文档](https://docs.pollen-robotics.com/)以获取有关该平台的更多信息！

## 遥操作 Reachy 2

目前，有两种方式可以遥操作 Reachy 2：

- Pollen Robotics 的 VR 遥操作（不包含在 LeRobot 中）。
- 机器人对机器人遥操作（使用一个 Reachy 2 控制另一个）。

## Reachy 2 仿真

**（仅限 Linux）** 您可以使用提供的 [Docker 镜像](https://hub.docker.com/r/pollenrobotics/reachy2_core)在仿真中运行 Reachy 2（Gazebo 或 MuJoCo）。

1. 安装 [Docker Engine](https://docs.docker.com/engine/)。
2. 运行（对于 MuJoCo）：

```
docker run --rm -it \
  --name reachy \
  --privileged \
  --network host \
  --ipc host \
  --device-cgroup-rule='c 189:* rwm' \
  --group-add audio \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e DISPLAY="$DISPLAY" \
  -e RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}]: {message}" \
  -e REACHY2_CORE_SERVICE_FAKE="${REACHY2_CORE_SERVICE_FAKE:-true}" \
  -v /dev:/dev \
  -v "$HOME/.reachy_config":/home/reachy/.reachy_config_override \
  -v "$HOME/.reachy.log":/home/reachy/.ros/log \
  -v /usr/lib/x86_64-linux-gnu:/opt/host-libs \
  --entrypoint /package/launch.sh \
  pollenrobotics/reachy2_core:1.7.5.9_deploy \
  start_rviz:=true start_sdk_server:=true mujoco:=true
```

> [!NOTE]
> 如果 MuJoCo 运行缓慢（仿真频率低），请在前面的命令中附加 `-e LD_LIBRARY_PATH="/opt/host-libs:$LD_LIBRARY_PATH" \` 以提高性能：
>
> ```
> docker run --rm -it \
>   --name reachy \
>   --privileged \
>   --network host \
>   --ipc host \
>   --device-cgroup-rule='c 189:* rwm' \
>   --group-add audio \
>   -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
>   -e DISPLAY="$DISPLAY" \
>   -e RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}]: {message}" \
>   -e REACHY2_CORE_SERVICE_FAKE="${REACHY2_CORE_SERVICE_FAKE:-true}" \
>   -e LD_LIBRARY_PATH="/opt/host-libs:$LD_LIBRARY_PATH" \
>   -v /dev:/dev \
>   -v "$HOME/.reachy_config":/home/reachy/.reachy_config_override \
>   -v "$HOME/.reachy.log":/home/reachy/.ros/log \
>   -v /usr/lib/x86_64-linux-gnu:/opt/host-libs \
>   --entrypoint /package/launch.sh \
>   pollenrobotics/reachy2_core:1.7.5.9_deploy \
>   start_rviz:=true start_sdk_server:=true mujoco:=true
> ```

## 设置

### 先决条件

- 在您的机器人上，检查**服务镜像**是否满足最低版本：
  - **reachy2-core >= 1.7.5.2**
  - **webrtc >= 2.0.1.1**

然后，如果您想使用 VR 遥操作：

- 安装 [Reachy 2 遥操作应用程序](https://docs.pollen-robotics.com/teleoperation/teleoperation-introduction/discover-teleoperation/)。
  使用版本 **>=v1.2.0**

我们建议使用两台计算机：一台用于遥操作（需要 Windows），另一台用于使用 LeRobot 进行记录。

### 安装 LeRobot

按照[安装说明](https://github.com/huggingface/lerobot#installation)安装 LeRobot。

使用 Reachy 2 依赖项安装 LeRobot：

```bash
pip install -e ".[reachy2]"
```

### （可选但推荐）安装 pollen_data_acquisition_server

您如何管理 Reachy 2 记录会话取决于您，但**最简单**的方法是使用此服务器，以便您可以直接从 VR 遥操作应用程序控制会话。

> **注意：** 目前，只有 VR 遥操作应用程序可以作为此服务器的客户端，因此此步骤主要针对遥操作。您可以自由开发自定义客户端来根据您的需求管理会话。

在您的 LeRobot 环境中，从源代码安装服务器：

```bash
git clone https://github.com/pollen-robotics/pollen_data_acquisition_server.git
cd pollen_data_acquisition_server
pip install -e .
```

在[此处](https://github.com/pollen-robotics/pollen_data_acquisition_server)找到 pollen_data_acquisition_server 文档。

## 步骤 1：记录

### 获取 Reachy 2 IP 地址

在开始遥操作和数据记录之前，找到[机器人的 IP 地址](https://docs.pollen-robotics.com/getting-started/setup-reachy2/connect-reachy2/)。
我们强烈建议通过**以太网**连接所有设备（PC 和机器人）。

### 启动记录

使用 Reachy 2 VR 遥操作应用程序时，有两种方式管理记录会话：

- **使用数据采集服务器（推荐用于 VR 遥操作）**：VR 应用程序编排会话（通过服务器告诉 LeRobot 何时创建数据集、启动/停止片段），同时还控制机器人的运动。
- **使用 LeRobot 的记录脚本**：LeRobot 拥有会话控制并决定何时启动/停止片段。如果您还使用 VR 遥操作应用程序，它仅用于运动控制。

### 选项 1：使用 Pollen 数据采集服务器（推荐用于 VR 遥操作）

确保您已安装 pollen_data_acquisition_server，如设置部分所述。

启动数据采集服务器，以便能够直接从遥操作应用程序管理您的会话：

```bash
python -m pollen_data_acquisition_server.server
```

然后进入遥操作应用程序并选择"Data acquisition session"。
您最终可以按照显示的屏幕设置您的会话。

> 即使没有 VR 应用程序，您也可以使用自己的客户端实现来使用 `pollen_data_acquisition_server`。

### 选项 2：使用 lerobot.record

Reachy 2 完全支持 LeRobot 的记录功能。
如果您选择此选项但仍想使用 VR 遥操作应用程序，请在应用程序中选择"Standard session"。

**示例：在没有移动底座的情况下开始记录：**
首先将 reachy2 和 reachy2_teleoperator 添加到记录脚本的导入中。然后您可以使用以下命令：

```bash
lerobot-record \
    --robot.type=reachy2 \
    --robot.ip_address=192.168.0.200 \
    --robot.id=r2-0000 \
    --robot.use_external_commands=true \
    --robot.with_mobile_base=false \
    --teleop.type=reachy2_teleoperator \
    --teleop.ip_address=192.168.0.200 \
    --teleop.with_mobile_base=false \
    --robot.with_torso_camera=true \
    --dataset.repo_id=pollen_robotics/record_test \
    --dataset.single_task="Reachy 2 recording test" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=5 \
    --dataset.fps=15 \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    # --dataset.camera_encoder.vcodec=auto \
    --display_data=true
```

#### 特定选项

**扩展设置概述（包含所有选项）：**

```bash
lerobot-record \
    --robot.type=reachy2 \
    --robot.ip_address=192.168.0.200 \
    --robot.use_external_commands=true \
    --robot.with_mobile_base=true \
    --robot.with_l_arm=true \
    --robot.with_r_arm=true \
    --robot.with_neck=true \
    --robot.with_antennas=true \
    --robot.with_left_teleop_camera=true \
    --robot.with_right_teleop_camera=true \
    --robot.with_torso_camera=false \
    --robot.camera_width=640 \
    --robot.camera_height=480 \
    --robot.disable_torque_on_disconnect=false \
    --robot.max_relative_target=5.0 \
    --teleop.type=reachy2_teleoperator \
    --teleop.ip_address=192.168.0.200 \
    --teleop.use_present_position=false \
    --teleop.with_mobile_base=false \
    --teleop.with_l_arm=true \
    --teleop.with_r_arm=true \
    --teleop.with_neck=true \
    --teleop.with_antennas=true \
    --dataset.repo_id=pollen_robotics/record_test \
    --dataset.single_task="Reachy 2 recording test" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=5 \
    --dataset.fps=15 \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    # --dataset.camera_encoder.vcodec=auto \
    --display_data=true
```

##### `--robot.use_external_commands`

确定 LeRobot robot.send_action() 是否向机器人发送命令。
在使用 VR 遥操作应用程序时**必须**设置为 false，因为应用程序已经发送命令。

##### `--teleop.use_present_position`

确定遥操作器读取机器人的目标位置还是当前位置。
如果使用柔顺的 Reachy 2 来控制另一个，则必须设置为 true。

##### 使用相关部件

根据我们的初步测试，在某些策略下，当只有部分关节移动时记录**所有**关节可能会降低模型质量。
为避免这种情况，您可以使用以下方式从记录和重放中排除特定部件：

```bash
--robot.with_<part>=false
```

其中 `<part>` 是以下之一：`mobile_base`、`l_arm`、`r_arm`、`neck`、`antennas`。
它确定相应的部件是否记录在观测中。如果未设置则为 True。

默认情况下，**记录所有部件**。

相同的按部件机制在 `reachy2_teleoperator` 中也可用。

```bash
--teleop.with_<part>
```

其中 `<part>` 是以下之一：`mobile_base`、`l_arm`、`r_arm`、`neck`、`antennas`。
确定相应的部件是否记录在动作中。如果未设置则为 True。

> **重要：** 在给定会话中，机器人和遥操作器上的**启用部件必须匹配**。
> 例如，如果机器人使用 `--robot.with_mobile_base=false` 运行，遥操作器必须禁用相同的部件 `--teleoperator.with_mobile_base=false`。

##### 使用相关相机

您可以对**相机**执行相同的操作。使用默认参数启用或禁用每个相机：

```bash
--robot.with_left_teleop_camera=<true|false> \
--robot.with_right_teleop_camera=<true|false> \
--robot.with_torso_camera=<true|false>
```

默认情况下，不记录相机，所有相机参数都设置为 `false`。
如果您愿意，可以使用 `--robot.camera_width` 和 `--robot.camera_height` 参数为 Reachy 2 的相机使用自定义 `width` 和 `height` 参数：

```bash
--robot.camera_width=1920 \
--robot.camera_height=1080
```

这将更改所有 3 个默认机器人相机（由上述布尔参数启用）的分辨率。

如果您愿意，可以像往常一样添加机器人中的其他相机：

```bash
--robot.cameras="{ extra: {type: opencv, index_or_path: 42, width: 640, height: 480, fps: 30}}" \
```

## 步骤 2：重放

确保机器人配置了与数据集相同的部件：

```bash
lerobot-replay \
    --robot.type=reachy2 \
    --robot.ip_address=192.168.0.200 \
    --robot.use_external_commands=false \
    --robot.with_mobile_base=false \
    --dataset.repo_id=pollen_robotics/record_test \
    --dataset.episode=0
```

## 步骤 3：训练

```bash
lerobot-train \
  --dataset.repo_id=pollen_robotics/record_test \
  --policy.type=act \
  --output_dir=outputs/train/reachy2_test \
  --job_name=reachy2 \
  --policy.device=mps \
  --wandb.enable=true \
  --policy.repo_id=pollen_robotics/record_test_policy
```

## 步骤 4：评估

```bash
lerobot-eval \
  --robot.type=reachy2 \
  --robot.ip_address=192.168.0.200 \
  --dataset.repo_id=pollen_robotics/eval_record_test \
  --dataset.single_task="Evaluate reachy2 policy" \
  --dataset.num_episodes=10 \
  --policy.path=outputs/train/reachy2_test/checkpoints/last/pretrained_model
```
