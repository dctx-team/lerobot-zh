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

import functools
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, REWARD
from lerobot.utils.transition import Transition


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    以向量化方式对一批图像执行逐图像随机裁剪。
    （与之前显示的相同。）
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # 收集像素
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """向量化的随机移位，imgs: (B,C,H,W)，pad: 像素数量"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        用于存储转移的回放缓冲区。
        当添加第一个转移时，它将在指定的设备上分配张量。
        注意：如果遇到内存问题，可以尝试使用 `optimize_memory` 标志来节省内存，或者
        使用 `storage_device` 标志将缓冲区存储在不同的设备上。
        参数：
            capacity (int)：缓冲区中存储转移的最大数量。
            device (str)：采样时张量将被移动到的设备（"cuda:0" 或 "cpu"）。
            state_keys (List[str])：出现在 `state` 和 `next_state` 中的键列表。
            image_augmentation_function (Optional[Callable])：一个接受一批图像并返回一批增强图像的函数。
                如果为 None，则使用默认的增强函数。
            use_drq (bool)：在缓冲区中采样时是否使用默认的 DRQ 图像增强风格。
            storage_device：数据将存储在的设备（例如 "cpu" 或 "cuda:0"）。
                使用 "cpu" 可以帮助节省 GPU 内存。
            optimize_memory (bool)：如果为 True，通过不存储重复的 next_states 来优化内存
                （当它们可以从 states 派生时）。这对于大型数据集很有用，其中 next_state[i] = state[i+1]。
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # 跟踪回合边界以进行内存优化
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # 如果未提供 state_keys，默认为空列表
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """根据第一个转移初始化存储张量。"""
        # 从第一个转移确定形状
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # 为存储预分配张量
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # 标准方法：分别存储 states 和 next_states
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # 内存优化方法：不分配 next_states 缓冲区
            # 只创建一个对 states 的引用以保持 API 一致性
            self.next_states = self.states  # 只是为了 API 一致性的引用

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # 初始化 complementary_info 的存储
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # 为 complementary_info 中的每个键预分配张量
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int, float)):
                    # 处理类似于 reward 的标量值
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """保存一个转移，确保张量存储在指定的存储设备上。"""
        # 如果这是第一个转移，则初始化存储
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # 在预分配的张量中存储转移
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # 仅当不优化内存时才存储 next_states
                self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        # 如果提供了 complementary_info 且存储已初始化，则处理它
        if complementary_info is not None and self.has_complementary_info:
            # 存储 complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][self.position].copy_(value.squeeze(dim=0))
                    elif isinstance(value, (int, float)):
                        self.complementary_info[key][self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """采样一个随机批次的转移并将它们整理为批次张量。"""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

        # 用于采样的随机索引 - 在与存储相同的设备上创建
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # 识别需要增强的图像键
        image_keys = [k for k in self.states if k.startswith(OBS_IMAGE)] if self.use_drq else []

        # 创建批次 state 和 next_state
        batch_state = {}
        batch_next_state = {}

        # 第一遍：将所有状态张量加载到目标设备
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # 标准方法 - 直接加载 next_states
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # 内存优化方法 - 从下一个索引获取 next_state
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # 如果需要，以批次方式应用图像增强
        if self.use_drq and image_keys:
            # 连接来自 state 和 next_state 的所有图像
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # 优化：批处理所有图像并应用一次增强
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # 将增强后的图像拆分回它们的来源
            for i, key in enumerate(image_keys):
                # 计算当前图像键的偏移量：
                # 对于每个键，我们有 2*batch_size 张图像（batch_size 用于 states，batch_size 用于 next_states）
                # States 从索引 i*2*batch_size 开始，占用 batch_size 个槽位
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states 从索引 (i*2+1)*batch_size 开始，也占用 batch_size 个槽位
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # 采样其他张量
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        # 如果可用，采样 complementary_info
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        创建一个无限迭代器，生成转移批次。
        当内部迭代器耗尽时将自动重启。

        参数：
            batch_size (int)：要采样的批次大小
            async_prefetch (bool)：是否使用线程进行异步预取（默认：True）
            queue_size (int)：要预取的批次数量（默认：2）

        生成：
            BatchTransition：批次转移
        """
        while True:  # 创建一个无限循环
            if async_prefetch:
                # 获取标准迭代器
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # 从迭代器中生成所有项
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        创建一个迭代器，在后台线程中持续生成预取的批次。
        设计故意简单，避免忙等待/复杂的状态管理。

        参数：
            batch_size (int)：要采样的批次大小。
            queue_size (int)：要在内存中保留的预取批次的最大数量。

        生成：
            BatchTransition：从回放缓冲区采样的批次。
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """持续将采样的批次放入队列，直到关闭。"""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    # 超时确保如果队列已满，线程可以解除阻塞
                    # 同时关闭事件被设置。
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # 队列已满 - 再次循环（将重新检查 shutdown_event）
                    continue
                except Exception:
                    # 显示任何意外错误并终止生产者。
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # 如果生产者已经设置了关闭标志，我们退出。
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # 快速排空队列以帮助线程退出（如果它在 `put` 上被阻塞）。
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # 给生产者线程一些时间来完成。
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        创建一个简单的非线程迭代器，生成批次。

        参数：
            batch_size (int)：要采样的批次大小
            queue_size (int)：要预取的初始批次数量

        生成：
            BatchTransition：批次转移
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        将 LeRobotDataset 转换为 ReplayBuffer。

        参数：
            lerobot_dataset (LeRobotDataset)：要转换的数据集。
            device (str)：用于采样张量的设备。默认为 "cuda:0"。
            state_keys (Sequence[str] | None)：出现在 `state` 和 `next_state` 中的键列表。
            capacity (int | None)：缓冲区容量。如果为 None，则使用数据集长度。
            action_mask (Sequence[int] | None)：要保留的动作维度的索引。
            image_augmentation_function (Callable | None)：用于图像增强的函数。
                如果为 None，则使用默认的 pad=4 的随机移位。
            use_drq (bool)：在采样时是否使用 DrQ 图像增强。
            storage_device (str)：用于存储张量数据的设备。使用 "cpu" 可节省 GPU 内存。
            optimize_memory (bool)：如果为 True，通过不复制状态数据来减少内存使用。

        返回：
            ReplayBuffer：包含数据集转移的回放缓冲区。
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # 使用图像增强和 DrQ 设置创建回放缓冲区
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # 将数据集转换为转移
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # 使用第一个转移初始化缓冲区以设置存储张量
        if list_transition:
            first_transition = list_transition[0]
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition[ACTION].to(device)

            # 如果可用，获取 complementary info
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # 用所有转移填充缓冲区
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data[ACTION]

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # 注意：lerobot 数据集尚不支持截断
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        将此 ReplayBuffer 中的所有转移转换为单个 LeRobotDataset 对象。
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # 为数据集创建特征字典
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # 跨回合的全局索引
            "episode_index": {"dtype": "int64", "shape": [1]},  # 哪个回合
            "frame_index": {"dtype": "int64", "shape": [1]},  # 回合内的索引
            "timestamp": {"dtype": "float32", "shape": [1]},  # 现在我们存储虚拟值
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # 添加 "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name=ACTION)
        features[ACTION] = act_info

        # 添加 "reward" 和 "done"
        features[REWARD] = {"dtype": "float32", "shape": (1,)}
        features[DONE] = {"dtype": "bool", "shape": (1,)}

        # 添加状态键
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # 如果可用，添加 complementary_info 键
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # 创建一个空的 LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # 如果需要，开始写入图像
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # 将转移转换为回合和帧

        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # 填充状态键的数据
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # 填充 action、reward、done
            frame_dict[ACTION] = self.actions[actual_idx].cpu()
            frame_dict[REWARD] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()
            frame_dict["task"] = task_name

            # 如果可用，添加 complementary_info
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # 将张量转换为 CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # 非张量值可以直接使用
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # 添加到数据集的缓冲区
            lerobot_dataset.add_frame(frame_dict)

            # 如果到达回合边界，调用 save_episode，重置计数器
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()

        # 保存缓冲区中任何剩余的帧
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[Transition]:
        """
        将 LeRobotDataset 转换为 RL (s, a, r, s', done) 转移列表。

        参数：
            dataset (LeRobotDataset)：
                要转换的数据集。数据集中的每个项预期至少有以下键：
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                加上您的 'state_keys' 指定的任何内容。

            state_keys (Sequence[str] | None)：
                要包含在 'state' 和 'next_state' 中的数据集键。它们的名称
                将在输出转移中保持原样。例如
                ["observation.state", "observation.environment_state"]。
                如果为 None，您必须处理或定义默认键。

        返回：
            transitions (List[Transition])：
                与 `dataset` 长度相同的 Transition 字典列表。
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # 检查数据集是否有 "next.done" 键
        sample = dataset[0]
        has_done_key = DONE in sample

        # 检查 complementary_info 键
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # 如果没有，我们需要从回合边界推断
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) 当前状态 -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # 添加批次维度

            # ----- 2) 动作 -----
            action = current_sample[ACTION].unsqueeze(0)  # 添加批次维度

            # ----- 3) 奖励和 done -----
            reward = float(current_sample[REWARD].item())  # 确保为 float

            # 确定 done 标志 - 如果可用则使用 next.done，否则从回合边界推断
            if has_done_key:
                done = bool(current_sample[DONE].item())  # 确保为 bool
            else:
                # 如果这是最后一帧或下一帧在不同的回合中，则标记为 done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) 处理截断（现在使用与 done 相同的值）
            truncated = done

            # ----- 4) 下一个状态 -----
            # 如果不是 done 且下一个样本在同一回合中，我们提取下一个样本的状态。
            # 否则（done=True 或下一个样本跨到新回合），next_state = current_state。
            next_state = current_state  # 默认值
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # 从相同的键构建 next_state
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # 添加批次维度
                    next_state = next_state_data

            # ----- 5) 补充信息（如果可用）-----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # 剥离 "complementary_info." 前缀以获取实际键
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # 对张量和非张量值进行不同处理
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # 添加批次维度
                    else:
                        # TODO: (azouitine) 检查是否有必要转换为张量
                        # 对于非张量值，直接使用
                        complementary_info[clean_key] = val

            # ----- 构造 Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions


# 用于从张量猜测形状/数据类型的实用函数
def guess_feature_info(t, name: str):
    """
    返回一个包含给定张量或标量值的 'dtype' 和 'shape' 的字典。
    如果它看起来像一个 3D (C,H,W) 形状，我们可能将其视为 'image'。
    否则默认为适当的数字类型。
    """

    shape = tuple(t.shape)
    # 基本猜测：如果我们恰好有 3 个维度且 shape[0] 在 {1, 3} 中，猜测为 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # 否则视为数字类型
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    将两个 BatchTransition 对象连接为一个。

    此函数通过沿维度 0 连接所有对应的张量，将右侧 BatchTransition 合并到左侧。
    该操作会就地修改 left_batch_transitions 并返回它。

    参数：
        left_batch_transitions (BatchTransition)：要连接的第一个批次，也是将被就地修改的批次。
        right_batch_transition (BatchTransition)：要附加到第一个批次的第二个批次。

    返回：
        BatchTransition：连接后的批次（与 left_batch_transitions 是同一对象）。

    警告：
        此函数会就地修改 left_batch_transitions 对象。
    """
    # 连接状态字段
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # 连接基本字段
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # 连接 next_state 字段
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # 连接 done 和 truncated 字段
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # 处理 complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # 仅当 right_info 存在时才处理
    if right_info is not None:
        # 如果需要，初始化左侧 complementary_info
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # 连接每个字段
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
