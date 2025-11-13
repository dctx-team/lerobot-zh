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
import multiprocessing
import queue
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch


def safe_stop_image_writer(func):
    """装饰器：在函数抛出异常时安全地停止图像写入器。"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("等待图像写入器终止...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    """将图像数组转换为 PIL 图像。

    Args:
        image_array: 输入的图像数组
        range_check: 是否检查值范围

    Returns:
        PIL.Image.Image: 转换后的 PIL 图像
    """
    # TODO(aliberts): 处理 1 个通道和 4 个通道的深度图像
    if image_array.ndim != 3:
        raise ValueError(f"数组有 {image_array.ndim} 个维度，但图像预期为 3 维。")

    if image_array.shape[0] == 3:
        # 从 PyTorch 约定 (C, H, W) 转置到 (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3:
        raise NotImplementedError(
            f"图像有 {image_array.shape[-1]} 个通道，但目前只支持 3 个通道。"
        )

    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "图像数据类型为浮点型，需要值在 [0.0, 1.0] 范围内。"
                    f"但是提供的范围是 [{min_}, {max_}]。请调整范围或"
                    "提供值在 [0, 255] 范围内的 uint8 图像。"
                )

        image_array = (image_array * 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path):
    """将图像写入磁盘。

    Args:
        image: numpy 数组或 PIL 图像
        fpath: 文件保存路径
    """
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")
        img.save(fpath)
    except Exception as e:
        print(f"写入图像 {fpath} 时出错：{e}")


def worker_thread_loop(queue: queue.Queue):
    """工作线程循环，从队列中获取图像并写入磁盘。"""
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath = item
        write_image(image_array, fpath)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    """工作进程，启动多个工作线程。"""
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    此类抽象了用于异步在磁盘上保存图像的进程或/和线程的初始化，
    这对于控制机器人和以高帧率记录数据至关重要。

    当 `num_processes=0` 时，它创建一个大小为 `num_threads` 的线程池。
    当 `num_processes>0` 时，它创建一个大小为 `num_processes` 的进程池，
    其中每个子进程启动自己的大小为 `num_threads` 的线程池。

    最佳的进程数和线程数取决于您的计算机能力。
    我们建议每个摄像头使用 4 个线程，进程数为 0。如果 fps 不稳定，
    尝试增加或减少线程数。如果仍然不稳定，尝试使用 1 个子进程或更多。
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("线程数和进程数必须大于零。")

        if self.num_processes == 0:
            # 使用线程
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # 使用多进程
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path):
        """将图像添加到保存队列。

        Args:
            image: 要保存的图像（torch.Tensor、numpy 数组或 PIL 图像）
            fpath: 保存路径
        """
        if isinstance(image, torch.Tensor):
            # 将张量转换为 numpy 数组以减少主进程时间
            image = image.cpu().numpy()
        self.queue.put((image, fpath))

    def wait_until_done(self):
        """等待队列中的所有图像写入完成。"""
        self.queue.join()

    def stop(self):
        """停止所有工作线程和进程。"""
        if self._stopped:
            return

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True
