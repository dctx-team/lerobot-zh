#!/usr/bin/env python

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

# ruff: noqa: N802
# 此noqa标记用于Protocol类：PortHandler、PacketHandler、GroupSyncRead/Write
# TODO(aliberts): 当以下功能可用时添加块级noqa
# https://github.com/astral-sh/ruff/issues/3711

import abc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import Protocol, TypeAlias

import serial
from deepdiff import DeepDiff
from tqdm import tqdm

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

NameOrID: TypeAlias = str | int
Value: TypeAlias = int | float

logger = logging.getLogger(__name__)


def get_ctrl_table(model_ctrl_table: dict[str, dict], model: str) -> dict[str, tuple[int, int]]:
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for {model=} not found.")
    return ctrl_table


def get_address(model_ctrl_table: dict[str, dict], model: str, data_name: str) -> tuple[int, int]:
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(f"Address for '{data_name}' not found in {model} control table.")
    return addr_bytes


def assert_same_address(model_ctrl_table: dict[str, dict], motor_models: list[str], data_name: str) -> None:
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = get_address(model_ctrl_table, model, data_name)
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_addr, strict=False))})."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_bytes, strict=False))})."
        )


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class PortHandler(Protocol):
    def __init__(self, port_name):
        self.is_open: bool
        self.baudrate: int
        self.packet_start_time: float
        self.packet_timeout: float
        self.tx_time_per_byte: float
        self.is_using: bool
        self.port_name: str
        self.ser: serial.Serial

    def openPort(self): ...
    def closePort(self): ...
    def clearPort(self): ...
    def setPortName(self, port_name): ...
    def getPortName(self): ...
    def setBaudRate(self, baudrate): ...
    def getBaudRate(self): ...
    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class PacketHandler(Protocol):
    def getTxRxResult(self, result): ...
    def getRxPacketError(self, error): ...
    def txPacket(self, port, txpacket): ...
    def rxPacket(self, port): ...
    def txRxPacket(self, port, txpacket): ...
    def ping(self, port, id): ...
    def action(self, port, id): ...
    def readTx(self, port, id, address, length): ...
    def readRx(self, port, id, length): ...
    def readTxRx(self, port, id, address, length): ...
    def read1ByteTx(self, port, id, address): ...
    def read1ByteRx(self, port, id): ...
    def read1ByteTxRx(self, port, id, address): ...
    def read2ByteTx(self, port, id, address): ...
    def read2ByteRx(self, port, id): ...
    def read2ByteTxRx(self, port, id, address): ...
    def read4ByteTx(self, port, id, address): ...
    def read4ByteRx(self, port, id): ...
    def read4ByteTxRx(self, port, id, address): ...
    def writeTxOnly(self, port, id, address, length, data): ...
    def writeTxRx(self, port, id, address, length, data): ...
    def write1ByteTxOnly(self, port, id, address, data): ...
    def write1ByteTxRx(self, port, id, address, data): ...
    def write2ByteTxOnly(self, port, id, address, data): ...
    def write2ByteTxRx(self, port, id, address, data): ...
    def write4ByteTxOnly(self, port, id, address, data): ...
    def write4ByteTxRx(self, port, id, address, data): ...
    def regWriteTxOnly(self, port, id, address, length, data): ...
    def regWriteTxRx(self, port, id, address, length, data): ...
    def syncReadTx(self, port, start_address, data_length, param, param_length): ...
    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length): ...


class GroupSyncRead(Protocol):
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.last_result: bool
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id): ...
    def removeParam(self, id): ...
    def clearParam(self): ...
    def txPacket(self): ...
    def rxPacket(self): ...
    def txRxPacket(self): ...
    def isAvailable(self, id, address, data_length): ...
    def getData(self, id, address, data_length): ...


class GroupSyncWrite(Protocol):
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id, data): ...
    def removeParam(self, id): ...
    def changeParam(self, id, data): ...
    def clearParam(self): ...
    def txPacket(self): ...


class MotorsBus(abc.ABC):
    """
    MotorsBus允许高效地读取和写入已连接的电机。
    它表示通过串行端口菊花链式连接在一起的多个电机。
    此抽象类目前有两个实现：
        - DynamixelMotorsBus
        - FeetechMotorsBus

    注意：如果我们添加其他类型总线的支持，此类可能会在未来发生变化。

    MotorsBus子类实例需要一个端口（例如 `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`）。
    要查找端口，可以运行我们的实用脚本：
    ```bash
    lerobot-find-port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ["/dev/tty.usbmodem575E0032081", "/dev/tty.usbmodem575E0031751"]
    >>> Remove the usb cable from your MotorsBus and press Enter when done.
    >>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    连接到总线的1个Feetech sts3215电机的使用示例：
    ```python
    bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"my_motor": (1, "sts3215")},
    )
    bus.connect()

    position = bus.read("Present_Position", "my_motor", normalize=False)

    # 作为示例，移动几个电机步数
    few_steps = 30
    bus.write("Goal_Position", "my_motor", position + few_steps, normalize=False)

    # 完成后，使用以下方法正确断开端口连接
    bus.disconnect()
    ```
    """

    apply_drive_mode: bool
    available_baudrates: list[int]
    default_baudrate: int
    default_timeout: int
    model_baudrate_table: dict[str, dict]
    model_ctrl_table: dict[str, dict]
    model_encoding_table: dict[str, dict]
    model_number_table: dict[str, int]
    model_resolution_table: dict[str, int]
    normalized_data: list[str]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.sync_reader: GroupSyncRead
        self.sync_writer: GroupSyncWrite
        self._comm_success: int
        self._no_error: int

        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        if len(self.models) < 2:
            return False

        first_table = self.model_ctrl_table[self.models[0]]
        return any(
            DeepDiff(first_table, get_ctrl_table(self.model_ctrl_table, model)) for model in self.models[1:]
        )

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motor_model(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors.copy()
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> list[str]:
        if isinstance(values, (int, float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        else:
            raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Some motors have the same id!\n{self}")

        # Ensure ctrl table available for all models
        for model in self.models:
            get_ctrl_table(self.model_ctrl_table, model)

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self.model_number_table[m.model] for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            error_lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]

            if missing_ids:
                error_lines.append("\nMissing motor IDs:")
                error_lines.extend(
                    f"  - {id_} (expected model: {expected_models[id_]})" for id_ in missing_ids
                )

            if wrong_models:
                error_lines.append("\nMotors with incorrect model numbers:")
                error_lines.extend(
                    f"  - {id_} ({self._id_to_name(id_)}): expected {expected}, found {found}"
                    for id_, (expected, found) in wrong_models.items()
                )

            error_lines.append("\nFull expected motor list (id: model_number):")
            error_lines.append(pformat(expected_models, indent=4, sort_dicts=False))
            error_lines.append("\nFull found motor list (id: model_number):")
            error_lines.append(pformat(found_models, indent=4, sort_dicts=False))

            raise RuntimeError("\n".join(error_lines))

    @abc.abstractmethod
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        """bool: 如果底层串口已打开则为 `True`。"""
        return self.port_handler.is_open

    def connect(self, handshake: bool = True) -> None:
        """打开串口并初始化通信。

        Args:
            handshake (bool, optional): 对每个预期的电机执行ping操作，并执行特定于实现的
                其他完整性检查。默认为 `True`。

        Raises:
            DeviceAlreadyConnectedError: 端口已打开。
            ConnectionError: 底层SDK无法打开端口或握手失败。
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected. Do not call `{self.__class__.__name__}.connect()` twice."
            )

        self._connect(handshake)
        self.set_timeout()
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            elif handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

    @abc.abstractmethod
    def _handshake(self) -> None:
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        """关闭串口（可选择先禁用扭矩）。

        Args:
            disable_torque (bool, optional): 如果为 `True`（默认），在关闭端口之前
                禁用所有电机的扭矩。这可以防止电机在断开连接后仍然施加阻力扭矩而损坏。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. Try running `{self.__class__.__name__}.connect()` first."
            )

        if disable_torque:
            self.port_handler.clearPort()
            self.port_handler.is_using = False
            self.disable_torque(num_retry=5)

        self.port_handler.closePort()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_port(cls, port: str, *args, **kwargs) -> dict[int, list[int]]:
        """在每个支持的波特率下探测 *port* 并列出响应的ID。

        Args:
            port (str): 要扫描的串口/USB端口（例如 ``"/dev/ttyUSB0"``）。
            *args, **kwargs: 转发到子类构造函数。

        Returns:
            dict[int, list[int]]: 映射 *波特率 → 电机ID列表*，
            包含至少产生一个响应的每个波特率。
        """
        bus = cls(port, {}, *args, **kwargs)
        bus._connect(handshake=False)
        baudrate_ids = {}
        for baudrate in tqdm(bus.available_baudrates, desc="Scanning port"):
            bus.set_baudrate(baudrate)
            ids_models = bus.broadcast_ping()
            if ids_models:
                tqdm.write(f"Motors found for {baudrate=}: {pformat(ids_models, indent=4)}")
                baudrate_ids[baudrate] = list(ids_models)

        bus.port_handler.closePort()
        return baudrate_ids

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """为单个电机分配正确的ID和波特率。

        此辅助函数临时切换到电机的当前设置，禁用扭矩，设置所需的ID，
        最后编程总线的默认波特率。

        Args:
            motor (str): :pyattr:`motors` 中电机的键。
            initial_baudrate (int | None, optional): 当前波特率（提供时跳过扫描）。
                默认为None。
            initial_id (int | None, optional): 当前ID（提供时跳过扫描）。默认为None。

        Raises:
            RuntimeError: 找不到电机或其型号编号与预期不匹配。
            ConnectionError: 与电机的通信失败。
        """
        if not self.is_connected:
            self._connect(handshake=False)

        if initial_baudrate is None:
            initial_baudrate, initial_id = self._find_single_motor(motor)

        if initial_id is None:
            _, initial_id = self._find_single_motor(motor, initial_baudrate)

        model = self.motors[motor].model
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        self._disable_torque(initial_id, model)

        # Set ID
        addr, length = get_address(self.model_ctrl_table, model, "ID")
        self._write(addr, length, initial_id, target_id)

        # Set Baudrate
        addr, length = get_address(self.model_ctrl_table, model, "Baud_Rate")
        baudrate_value = self.model_baudrate_table[model][self.default_baudrate]
        self._write(addr, length, target_id, baudrate_value)

        self.set_baudrate(self.default_baudrate)

    @abc.abstractmethod
    def _find_single_motor(self, motor: str, initial_baudrate: int | None) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def configure_motors(self) -> None:
        """向每个电机写入特定于实现的推荐设置。

        典型更改包括缩短返回延迟、增加加速度限制或禁用安全锁。
        """
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        """禁用选定电机的扭矩。

        禁用扭矩允许写入电机的永久存储区域（EPROM/EEPROM）。

        Args:
            motors (int | str | list[str] | None, optional): 目标电机。接受电机名称、ID、
                名称列表或 `None` 以影响所有已注册的电机。默认为 `None`。
            num_retry (int, optional): 通信失败时的额外重试次数。默认为0。
        """
        pass

    @abc.abstractmethod
    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """启用选定电机的扭矩。

        Args:
            motor (int): 语义与 :pymeth:`disable_torque` 相同。默认为 `None`。
            num_retry (int, optional): 通信失败时的额外重试次数。默认为0。
        """
        pass

    @contextmanager
    def torque_disabled(self, motors: int | str | list[str] | None = None):
        """保证重新启用扭矩的上下文管理器。

        此辅助函数对于在配置电机时临时禁用扭矩很有用。

        Examples:
            >>> with bus.torque_disabled():
            ...     # 这里执行安全操作
            ...     pass
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):
        """更改SDK使用的数据包超时时间。

        Args:
            timeout_ms (int | None, optional): 超时时间（*毫秒*）。如果为 `None`（默认），
                该方法回退到 :pyattr:`default_timeout`。
        """
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:
        """返回端口上配置的当前波特率。

        Returns:
            int: 波特率（位/秒）。
        """
        return self.port_handler.getBaudRate()

    def set_baudrate(self, baudrate: int) -> None:
        """在端口上设置新的UART波特率。

        Args:
            baudrate (int): 期望的波特率（位/秒）。

        Raises:
            RuntimeError: SDK无法应用更改。
        """
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            logger.info(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise RuntimeError("Failed to write bus baud rate.")

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """bool: 如果缓存的校准与电机匹配则为 ``True``。"""
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """从电机读取校准参数。

        Returns:
            dict[str, MotorCalibration]: 映射 *电机名称 → 校准*。
        """
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """将校准参数写入电机并可选择缓存它们。

        Args:
            calibration_dict (dict[str, MotorCalibration]): 从 :pymeth:`read_calibration` 获得
                或由用户创建的校准。
            cache (bool, optional): 将校准保存到 :pyattr:`calibration`。默认为True。
        """
        pass

    def reset_calibration(self, motors: NameOrID | list[NameOrID] | None = None) -> None:
        """恢复选定电机的出厂校准。

        归位偏移设置为 ``0``，最小/最大位置限制设置为完整可用范围。
        内存中的 :pyattr:`calibration` 被清除。

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): 电机选择。`None`（默认）
                重置每个电机。
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        for motor in motors:
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None) -> dict[NameOrID, Value]:
        """将每个电机的范围居中于其当前位置。

        该函数计算并写入归位偏移，使当前位置恰好为半圈（例如12位编码器上的 `2047`）。

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): 要调整的电机。默认为所有电机（`None`）。

        Returns:
            dict[NameOrID, Value]: 映射 *电机 → 写入的归位偏移*。
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        self.reset_calibration(motors)
        actual_positions = self.sync_read("Present_Position", motors, normalize=False)
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        pass

    def record_ranges_of_motion(
        self, motors: NameOrID | list[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
        """交互式记录每个电机的最小/最大编码器值。

        手动移动关节（禁用扭矩的情况下），同时该方法流式传输实时位置。按 :kbd:`Enter` 完成。

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): 要记录的电机。
                默认为每个电机（`None`）。
            display_values (bool, optional): 当为 `True`（默认）时，将实时表格打印到控制台。

        Returns:
            tuple[dict[NameOrID, Value], dict[NameOrID, Value]]: 两个字典 *mins* 和 *maxes*，
                包含每个电机观察到的极值。
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        start_positions = self.sync_read("Present_Position", motors, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motors, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motors:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(motors) + 3)

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @abc.abstractmethod
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    @abc.abstractmethod
    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    def _serialize_data(self, value: int, length: int) -> list[int]:
        """
        将无符号整数值转换为字节大小整数列表，以通过通信协议发送。
        根据协议，拆分的值可以是大端或小端顺序。

        Feetech和Dynamixel都支持的数据长度：
            - 1（用于0到255的值）
            - 2（用于0到65,535的值）
            - 4（用于0到4,294,967,295的值）
        """
        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {length} bytes ({max_value}).")

        return self._split_into_byte_chunks(value, length)

    @abc.abstractmethod
    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """将整数转换为字节大小整数列表。"""
        pass

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """Ping单个电机并返回其型号编号。

        Args:
            motor (NameOrID): 目标电机（名称或ID）。
            num_retry (int, optional): 放弃前的额外尝试次数。默认为 `0`。
            raise_on_error (bool, optional): 如果为 `True`，通信错误会引发异常而不是
                返回 `None`。默认为 `False`。

        Returns:
            int | None: 电机型号编号，失败时返回 `None`。
        """
        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
            if self._is_comm_success(comm):
                break
            logger.debug(f"ping failed for {id_=}: {n_try=} got {comm=} {error=}")

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            else:
                return
        if self._is_error(error):
            if raise_on_error:
                raise RuntimeError(self.packet_handler.getRxPacketError(error))
            else:
                return

        return model_number

    @abc.abstractmethod
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """使用广播地址ping总线上的每个ID。

        Args:
            num_retry (int, optional): 重试尝试次数。默认为 `0`。
            raise_on_error (bool, optional): 当为 `True` 时，失败会引发异常而不是返回
                `None`。默认为 `False`。

        Returns:
            dict[int, int] | None: 映射 *id → 型号编号*，如果调用失败则返回 `None`。
        """
        pass

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """从电机读取寄存器。

        Args:
            data_name (str): 控制表键（例如 `"Present_Position"`）。
            motor (str): 电机名称。
            normalize (bool, optional): 当为 `True`（默认）时，将值缩放到由校准定义的
                用户友好范围。
            num_retry (int, optional): 重试尝试次数。默认为 `0`。

        Returns:
            Value: 根据 *normalize* 参数返回原始值或归一化值。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        value, _, _ = self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

        id_value = self._decode_sign(data_name, {id_: value})

        if normalize and data_name in self.normalized_data:
            id_value = self._normalize(id_value)

        return id_value[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        if length == 1:
            read_fn = self.packet_handler.read1ByteTxRx
        elif length == 2:
            read_fn = self.packet_handler.read2ByteTxRx
        elif length == 4:
            read_fn = self.packet_handler.read4ByteTxRx
        else:
            raise ValueError(length)

        for n_try in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to read @{address=} ({length=}) on {motor_id=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return value, comm, error

    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """向单个电机的寄存器写入值。

        与 :pymeth:`sync_write` 不同，此方法期望电机发出响应状态数据包，
        这提供了值成功写入寄存器的保证。因此，它比 :pymeth:`sync_write` 慢，
        但更可靠。通常应在配置电机时使用。

        Args:
            data_name (str): 寄存器名称。
            motor (str): 电机名称。
            value (Value): 要写入的值。如果 *normalize* 为 `True`，该值首先转换为原始单位。
            normalize (bool, optional): 启用或禁用归一化。默认为 `True`。
            num_retry (int, optional): 重试尝试次数。默认为 `0`。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            value = self._unnormalize({id_: value})[id_]

        value = self._encode_sign(data_name, {id_: value})[id_]

        err_msg = f"Failed to write '{data_name}' on {id_=} with '{value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        data = self._serialize_data(value, length)
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, length, data)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) on id={motor_id} with {value=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return comm, error

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """一次从多个电机读取同一寄存器。

        Args:
            data_name (str): 寄存器名称。
            motors (str | list[str] | None, optional): 要查询的电机。`None`（默认）读取每个电机。
            normalize (bool, optional): 归一化标志。默认为 `True`。
            num_retry (int, optional): 重试尝试次数。默认为 `0`。

        Returns:
            dict[str, Value]: 映射 *电机名称 → 值*。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        self._assert_protocol_is_compatible("sync_read")

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        ids_values, _ = self._sync_read(
            addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

        ids_values = self._decode_sign(data_name, ids_values)

        if normalize and data_name in self.normalized_data:
            ids_values = self._normalize(ids_values)

        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[dict[int, int], int]:
        self._setup_sync_reader(motor_ids, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync read @{addr=} ({length=}) on {motor_ids=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
        return values, comm

    def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

    # TODO(aliberts, pkooij): 如果需要，实现类似这样的功能可以获得更快的读取时间。
    # 虽然必须处理检查数据包是否已发送的逻辑，但这是可行的。
    # 这可能会以增加数据产生时刻和策略使用时刻之间的延迟为代价。
    # def _async_read(self, motor_ids: list[int], address: int, length: int):
    #     if self.sync_reader.start_address != address or self.sync_reader.data_length != length or ...:
    #         self._setup_sync_reader(motor_ids, address, length)
    #     else:
    #         self.sync_reader.rxPacket()
    #         self.sync_reader.txPacket()

    #     for id_ in motor_ids:
    #         value = self.sync_reader.getData(id_, address, length)

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """向多个电机写入同一寄存器。

        与 :pymeth:`write` 不同，此方法*不*期望电机发出响应状态数据包，
        这可能导致数据包丢失。它比 :pymeth:`write` 快，通常应在频率很重要且
        可以接受丢失一些数据包的情况下使用（例如遥操作循环）。

        Args:
            data_name (str): 寄存器名称。
            values (Value | dict[str, Value]): 单个值（应用于每个电机）或映射
                *电机名称 → 值*。
            normalize (bool, optional): 如果为 `True`（默认），将值从用户范围转换为原始单位。
            num_retry (int, optional): 重试尝试次数。默认为 `0`。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in ids_values]
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            ids_values = self._unnormalize(ids_values)

        ids_values = self._encode_sign(data_name, ids_values)

        err_msg = f"Failed to sync write '{data_name}' with {ids_values=} after {num_retry + 1} tries."
        self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int:
        self._setup_sync_writer(ids_values, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) with {ids_values=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        return comm

    def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)
