"""
机器人模块

该模块提供了LeRobot框架的核心机器人接口和配置类。
包含机器人配置、基础机器人类以及工具函数。
"""

from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config
