#!/usr/bin/env python

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

from ..utils import TeleopEvents


class InputController:
    """生成运动增量的输入控制器基类。"""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        初始化控制器。

        参数：
            x_step_size: X轴基础移动步长（米）
            y_step_size: Y轴基础移动步长（米）
            z_step_size: Z轴基础移动步长（米）
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success" 或 "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """启动控制器并初始化资源。"""
        pass

    def stop(self):
        """停止控制器并释放资源。"""
        pass

    def get_deltas(self):
        """获取当前的移动增量（dx, dy, dz），单位为米。"""
        return 0.0, 0.0, 0.0

    def should_quit(self):
        """如果用户请求退出，则返回 True。"""
        return not self.running

    def update(self):
        """更新控制器状态 - 每帧调用一次。"""
        pass

    def __enter__(self):
        """支持 'with' 语句使用。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """确保在退出 'with' 块时释放资源。"""
        self.stop()

    def get_episode_end_status(self):
        """
        获取当前回合结束状态。

        返回：
            如果回合应继续则返回 None，否则返回 "success" 或 "failure"
        """
        status = self.episode_end_status
        self.episode_end_status = None  # 读取后重置
        return status

    def should_intervene(self):
        """如果设置了干预标志，则返回 True。"""
        return self.intervention_flag

    def gripper_command(self):
        """返回当前的夹爪命令。"""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """从键盘输入生成运动增量。"""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """启动键盘监听器。"""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """停止键盘监听器。"""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """从键盘状态获取当前的移动增量。"""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """如果按下 ESC，则返回 True。"""
        return self.key_states["quit"]

    def should_save(self):
        """如果按下 Enter（保存回合），则返回 True。"""
        return self.key_states["success"] or self.key_states["failure"]


class GamepadController(InputController):
    """从游戏手柄输入生成运动增量。"""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """初始化 pygame 和游戏手柄。"""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")

    def stop(self):
        """清理 pygame 资源。"""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """处理 pygame 事件以获取最新的游戏手柄读数。"""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A 按钮 (1) 表示失败
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X 按钮 (0) 表示重新记录
                elif event.button == 0:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # RB 按钮 (6) 用于关闭夹爪
                elif event.button == 6:
                    self.close_gripper_command = True

                # LT 按钮 (7) 用于打开夹爪
                elif event.button == 7:
                    self.open_gripper_command = True

            # 按钮释放时重置回合状态
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == 6:
                    self.close_gripper_command = False

                elif event.button == 7:
                    self.open_gripper_command = False

            # 检查 RB 按钮（通常是按钮 5）用于干预标志
            if self.joystick.get_button(5):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """从游戏手柄状态获取当前的移动增量。"""
        import pygame

        try:
            # 读取操纵杆轴
            # 左摇杆 X 和 Y（通常是轴 0 和 1）
            y_input = self.joystick.get_axis(0)  # 上/下（通常是反向的）
            x_input = self.joystick.get_axis(1)  # 左/右

            # 右摇杆 Y（通常是轴 3 或 4）
            z_input = self.joystick.get_axis(3)  # 上/下用于 Z 轴

            # 应用死区以避免漂移
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # 计算增量（注意：可能需要根据控制器反转轴）
            delta_x = -x_input * self.x_step_size  # 前/后
            delta_y = -y_input * self.y_step_size  # 左/右
            delta_z = -z_input * self.z_step_size  # 上/下

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """使用 HIDAPI 从游戏手柄输入生成运动增量。"""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        初始化 HID 游戏手柄控制器。

        参数：
            step_size: 基础移动步长（米）
            z_scale: Z 轴移动的缩放因子
            deadzone: 操纵杆死区以防止漂移
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # 移动值（归一化为 -1.0 到 1.0）
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # 按钮状态
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """通过供应商和产品 ID 查找游戏手柄设备。"""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """使用 HIDAPI 连接到游戏手柄。"""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick: Move in Z axis (vertical)")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """关闭 HID 设备连接。"""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        读取并处理最新的游戏手柄数据。
        由于 HIDAPI 存在问题，我们需要多次读取设备以获得稳定的读数。
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """读取并处理最新的游戏手柄数据。"""
        if not self.device or not self.running:
            return

        try:
            # 从游戏手柄读取数据
            data = self.device.read(64)
            # 解析游戏手柄数据 - 这将因控制器型号而异
            # 这些偏移量适用于 Logitech RumblePad 2
            if data and len(data) >= 8:
                # 将操纵杆值从 0-255 归一化到 -1.0-1.0
                self.left_y = (data[1] - 128) / 128.0
                self.left_x = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # 应用死区
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # 解析按钮状态（Logitech RumblePad 2 中的第 5 字节）
                buttons = data[5]

                # 如果按下 RB，则应设置干预标志
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # 检查是否按下 RT
                self.open_gripper_command = data[6] in [8, 10, 12]

                # 检查是否按下 LT
                self.close_gripper_command = data[6] in [4, 6, 12]

                # 检查是否按下 Y/Triangle 按钮（位 7）用于保存
                # 检查是否按下 X/Square 按钮（位 5）表示失败
                # 检查是否按下 A/Cross 按钮（位 4）用于重新记录
                if buttons & 1 << 7:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif buttons & 1 << 5:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif buttons & 1 << 4:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """从游戏手柄状态获取当前的移动增量。"""
        # 计算增量 - 根据需要基于控制器方向进行反转
        delta_x = -self.left_x * self.x_step_size  # 前/后
        delta_y = -self.left_y * self.y_step_size  # 左/右
        delta_z = -self.right_y * self.z_step_size  # 上/下

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """如果按下退出按钮，则返回 True。"""
        return self.quit_requested

    def should_save(self):
        """如果按下保存按钮，则返回 True。"""
        return self.save_requested
