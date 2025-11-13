# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# TODO(aliberts, Steven, Pepijn): 使用 gRPC 调用而不是 zmq？

import base64
import json
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_lekiwi import LeKiwiClientConfig


class LeKiwiClient(Robot):
    config_class = LeKiwiClientConfig
    name = "lekiwi_client"

    def __init__(self, config: LeKiwiClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}

        self.last_remote_state = {}

        # 定义三个速度等级和当前索引
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # 慢速
            {"xy": 0.2, "theta": 60},  # 中速
            {"xy": 0.3, "theta": 90},  # 快速
        ]
        self.speed_index = 0  # 从慢速开始

        self._is_connected = False
        self.logs = {}

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        """与远程移动机器人建立 ZMQ 套接字连接"""

        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "LeKiwi Daemon is already connected. Do not run `robot.connect()` twice."
            )

        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for LeKiwi Host to connect expired.")

        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def _poll_and_get_latest_message(self) -> str | None:
        """轮询 ZMQ 套接字一段有限时间，并返回最新的消息字符串。"""
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            logging.info("No new data available within timeout.")
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        if last_msg is None:
            logging.warning("Poller indicated data, but failed to retrieve message.")

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> dict[str, Any] | None:
        """解析 JSON 观测字符串。"""
        try:
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> np.ndarray | None:
        """将 base64 编码的图像字符串解码为 OpenCV 图像。"""
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image.")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """从解析的观测中提取帧和状态。"""

        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}

        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)

        obs_dict: dict[str, Any] = {**flat_state, OBS_STATE: state_vec}

        # 解码图像
        current_frames: dict[str, np.ndarray] = {}
        for cam_name, image_b64 in observation.items():
            if cam_name not in self._cameras_ft:
                continue
            frame = self._decode_image_from_b64(image_b64)
            if frame is not None:
                current_frames[cam_name] = frame

        return current_frames, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
        """
        轮询视频套接字以获取最新的观测数据。

        尝试在短暂超时内检索并解码最新消息。
        如果成功，更新并返回新的帧、速度和机械臂状态。
        如果没有新数据到达或解码失败，则返回最后已知的值。
        """

        # 1. 从套接字获取最新消息字符串
        latest_message_str = self._poll_and_get_latest_message()

        # 2. 如果没有消息，返回缓存数据
        if latest_message_str is None:
            return self.last_frames, self.last_remote_state

        # 3. 解析 JSON 消息
        observation = self._parse_observation_json(latest_message_str)

        # 4. 如果 JSON 解析失败，返回缓存数据
        if observation is None:
            return self.last_frames, self.last_remote_state

        # 5. 处理有效的观测数据
        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = new_frames
        self.last_remote_state = new_state

        return new_frames, new_state

    def get_observation(self) -> dict[str, Any]:
        """
        从远程机器人捕获观测：当前从动机械臂位置、
        当前车轮速度（转换为机体坐标系速度：x, y, theta）
        和摄像头帧。通过 ZMQ 接收，转换为机体坐标系速度
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("LeKiwiClient is not connected. You need to run `robot.connect()`.")

        frames, obs_dict = self._get_data()

        # 遍历每个配置的摄像头
        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("Frame is None")
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = frame

        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # 速度控制
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # 例如 0.1、0.25 或 0.4
        theta_speed = speed_setting["theta"]  # 例如 30、60 或 90

        x_cmd = 0.0  # m/s 前后
        y_cmd = 0.0  # m/s 横向
        theta_cmd = 0.0  # deg/s 旋转

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def configure(self):
        pass

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """命令 lekiwi 移动到目标关节配置。转换为电机空间 + 通过 ZMQ 发送

        Args:
            action (np.ndarray): 包含电机目标位置的数组。

        Raises:
            RobotDeviceNotConnectedError: 如果机器人未连接。

        Returns:
            np.ndarray: 发送到电机的动作，可能经过裁剪。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        self.zmq_cmd_socket.send_string(json.dumps(action))  # action 是在电机空间中

        # TODO(Steven): 当可以记录非 numpy 数组值时，删除 np 转换
        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)

        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent[ACTION] = actions
        return action_sent

    def disconnect(self):
        """清理 ZMQ 通信"""

        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeKiwi is not connected. You need to run `robot.connect()` before disconnecting."
            )
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
