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
import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from collections.abc import Sequence
from functools import wraps
from pathlib import Path

import draccus

from lerobot.utils.utils import has_method

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """在给定的嵌套属性级别解析命令行参数。

    例如,假设主脚本调用方式为:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    如果在执行 myscript.py 期间调用,get_cli_overrides("arg2") 将返回:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict:
    """从命令行参数中解析插件相关的参数。

    此函数从命令行参数中提取匹配指定后缀模式的参数。
    它处理 '--key=value' 格式的参数并将其作为字典返回。

    参数:
        plugin_arg_suffix (str): 用于识别插件相关参数的后缀。
        cli_args (Sequence[str]): 要解析的命令行参数序列。

    返回:
        dict: 包含已解析的插件参数的字典,其中:
            - 键为参数名称(如果存在,则移除 '--' 前缀)
            - 值为对应的参数值

    示例:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """插件加载失败时引发此异常。"""


def load_plugin(plugin_path: str) -> None:
    """从给定的 Python 包路径加载并初始化插件。

    此函数尝试通过导入其包和任何子模块来加载插件。
    插件注册预期在包初始化期间发生,即当导入包时,
    gym 环境应该被注册,配置类应使用 `register_subclass` 装饰器
    注册到其父类。

    参数:
        plugin_path (str): 插件的 Python 包路径(例如 "mypackage.plugins.myplugin")

    异常:
        PluginLoadError: 如果由于导入错误或包路径无效而无法加载插件。

    示例:
        >>> load_plugin("external_plugin.core")  # 从外部包加载插件

    注意:
        - 插件包应在导入期间处理自己的注册
        - 插件包中的所有子模块都将被导入
        - 实现遵循 Python 打包指南中的插件发现模式

    另请参阅:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg):
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    过滤与特定路径参数相关的字段的命令行参数。

    参数:
        fields_to_filter (str | list[str]): 需要过滤参数的单个字符串或字符串列表。
        args (Sequence[str] | None): 要过滤的命令行参数序列。
            默认为 None。

    返回:
        list[str]: 过滤后的参数列表,其中与指定字段相关的参数已被移除。

    异常:
        ArgumentError: 如果为同一字段同时指定了路径参数(例如 `--field_name.path`)
            和类型参数(例如 `--field_name.type`)。
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def wrap(config_path: Path | None = None):
    """
    HACK: 类似于 draccus.wrap,但额外做了三件事:
        - 将从 CLI 中移除 '.path' 参数以便稍后处理。
        - 如果传递了 'config_path' 且主配置类具有 'from_pretrained' 方法,
          将从那里初始化它以允许直接从 hub 获取配置
        - 将加载在 CLI 参数中指定的插件。这些插件通常会注册自己的配置类的子类,
          以便 draccus 可以从 CLI '.type' 参数中找到正确的类来实例化
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
