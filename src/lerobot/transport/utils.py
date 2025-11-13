#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import io
import json
import logging
import pickle  # nosec B403: Safe usage for internal serialization only
from multiprocessing import Event
from queue import Queue
from typing import Any

import torch

from lerobot.transport import services_pb2
from lerobot.utils.transition import Transition

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    """获取字节缓冲区的大小

    Args:
        buffer: BytesIO 缓冲区对象

    Returns:
        缓冲区的字节大小
    """
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    """分块发送字节数据

    Args:
        buffer: 要发送的字节数据
        message_class: 消息类
        log_prefix: 日志前缀
        silent: 是否静默模式

    Yields:
        消息对象，包含传输状态和数据块
    """
    buffer = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = services_pb2.TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = services_pb2.TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = services_pb2.TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(iterator, queue: Queue | None, shutdown_event: Event, log_prefix: str = ""):
    """分块接收字节数据

    Args:
        iterator: 数据迭代器
        queue: 可选的队列，用于存储接收到的数据
        shutdown_event: 关闭事件，用于优雅停止接收
        log_prefix: 日志前缀

    Returns:
        如果 queue 为 None，则返回接收到的字节数据
    """
    bytes_buffer = io.BytesIO()
    step = 0

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == services_pb2.TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == services_pb2.TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == services_pb2.TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """将模型状态字典转换为字节数组以进行传输

    Args:
        state_dict: 模型状态字典

    Returns:
        序列化后的字节数据
    """
    buffer = io.BytesIO()

    torch.save(state_dict, buffer)

    return buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    """将字节数据转换为模型状态字典

    Args:
        buffer: 序列化的字节数据

    Returns:
        模型状态字典
    """
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def python_object_to_bytes(python_object: Any) -> bytes:
    """将 Python 对象序列化为字节数据

    Args:
        python_object: 要序列化的 Python 对象

    Returns:
        序列化后的字节数据
    """
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    """将字节数据反序列化为 Python 对象

    Args:
        buffer: 序列化的字节数据

    Returns:
        反序列化后的 Python 对象
    """
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    obj = pickle.load(buffer)  # nosec B301: Safe usage of pickle.load
    # 在此处添加验证检查
    return obj


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    """将字节数据转换为 Transition 列表

    Args:
        buffer: 序列化的字节数据

    Returns:
        Transition 对象列表
    """
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    transitions = torch.load(buffer, weights_only=True)
    return transitions


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    """将 Transition 列表转换为字节数据

    Args:
        transitions: Transition 对象列表

    Returns:
        序列化后的字节数据
    """
    buffer = io.BytesIO()
    torch.save(transitions, buffer)
    return buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    """配置 gRPC 通道选项

    Args:
        max_receive_message_length: 最大接收消息长度
        max_send_message_length: 最大发送消息长度
        enable_retries: 是否启用重试
        initial_backoff: 初始退避时间
        max_attempts: 最大尝试次数
        backoff_multiplier: 退避乘数
        max_backoff: 最大退避时间

    Returns:
        gRPC 通道选项列表
    """
    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # 应用于所有服务的所有方法
                "retryPolicy": {
                    "maxAttempts": max_attempts,  # 最大重试次数（总尝试次数 = 5）
                    "initialBackoff": initial_backoff,  # 首次重试在 0.1 秒后
                    "maxBackoff": max_backoff,  # 重试之间的最大等待时间
                    "backoffMultiplier": backoff_multiplier,  # 指数退避因子
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # 在网络故障时重试
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]
