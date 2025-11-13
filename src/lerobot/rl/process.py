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

import logging
import os
import signal
import sys


class ProcessSignalHandler:
    """用于附加优雅关闭信号处理器的工具类。

    该类暴露一个 shutdown_event 属性，当接收到关闭信号时会被设置。
    一个计数器跟踪已捕获的关闭信号数量。在第二个信号时，进程将以状态 1 退出。
    """

    _SUPPORTED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")

    def __init__(self, use_threads: bool, display_pid: bool = False):
        # TODO: 检查我们是否可以使用 threading 的 Event，因为 multiprocessing 的 Event
        # 是 threading.Event 的克隆。
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Event
        if use_threads:
            from threading import Event
        else:
            from multiprocessing import Event

        self.shutdown_event = Event()
        self._counter: int = 0
        self._display_pid = display_pid

        self._register_handlers()

    @property
    def counter(self) -> int:  # pragma: no cover – simple accessor
        """已拦截的关闭信号数量。"""
        return self._counter

    def _register_handlers(self):
        """将内部 _signal_handler 附加到 POSIX 信号的子集。"""

        def _signal_handler(signum, frame):
            pid_str = ""
            if self._display_pid:
                pid_str = f"[PID: {os.getpid()}]"
            logging.info(f"{pid_str} Shutdown signal {signum} received. Cleaning up…")
            self.shutdown_event.set()
            self._counter += 1

            # 在第二次 Ctrl-C（或任何支持的信号）时强制退出，以
            # 模拟之前的行为，同时给调用者一次优雅关闭的机会。
            # TODO: 调查我们稍后是否需要它
            if self._counter > 1:
                logging.info("Force shutdown")
                sys.exit(1)

        for sig_name in self._SUPPORTED_SIGNALS:
            sig = getattr(signal, sig_name, None)
            if sig is None:
                # 该信号在此平台上不可用（例如 Windows 不提供 SIGHUP、SIGQUIT 等）。跳过它。
                continue
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):  # pragma: no cover – unlikely but safe
                # 不支持的信号或我们在非主线程中。
                continue
