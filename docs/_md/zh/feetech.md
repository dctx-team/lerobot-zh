# Feetech 电机固件更新

本教程指导您使用官方 Feetech 软件更新 Feetech 电机的固件。

## 前置条件

- Windows 计算机（Feetech 软件仅适用于 Windows）
- Feetech 电机控制板
- USB 线缆，用于将控制板连接到计算机
- 连接到控制板的 Feetech 电机

## 步骤 1：下载 Feetech 软件

1. 访问官方 Feetech 软件下载页面：[https://www.feetechrc.com/software.html](https://www.feetechrc.com/software.html)
2. 下载最新版本的 Feetech 调试软件（FD）
3. 在您的 Windows 计算机上安装该软件

## 步骤 2：硬件设置

1. 将您的 Feetech 电机连接到电机控制板
2. 通过 USB 线缆将电机控制板连接到您的 Windows 计算机
3. 确保为电机供电

## 步骤 3：配置连接

1. 启动 Feetech 调试软件
2. 从端口下拉菜单中选择正确的 COM 端口
   - 如果不确定使用哪个端口，请在 Windows 设备管理器的"端口 (COM & LPT)"下查看
3. 设置适当的波特率（大多数 Feetech 电机通常为 1000000）
4. 点击"打开"以建立与控制板的通信

## 步骤 4：扫描电机

1. 连接后，点击"搜索"按钮以检测所有连接的电机
2. 软件将自动发现并列出总线上的所有电机
3. 每个电机将显示其 ID 号

## 步骤 5：更新固件

对于您想要更新的每个电机：

1. **从列表中选择电机**，点击它
2. **点击升级选项卡**：
3. **点击在线按钮**：
   - 如果发现潜在的固件更新，它将显示在框中
4. **点击升级按钮**：
   - 将显示更新进度

## 步骤 6：验证更新

1. 更新完成后，软件应自动刷新电机信息
2. 验证固件版本已更新到预期版本

## 重要说明

⚠️ **警告**：在固件更新期间不要断开电源或 USB，这可能会导致电机变砖。

## 附加功能：在 Linux/macOS 上调试电机

仅用于调试目的，您可以使用开源 Feetech 调试工具：

- **仓库**：[FT_SCServo_Debug_Qt](https://github.com/CarolinePascal/FT_SCServo_Debug_Qt/tree/fix/port-search-timer)

### 安装说明

按照仓库中的说明安装该工具，对于 Ubuntu，您可以直接安装它，对于 macOS，您需要从源代码构建它。

**限制：**

- 此工具仅用于调试和参数调整
- 固件更新仍必须在 Windows 上使用官方 Feetech 软件完成
