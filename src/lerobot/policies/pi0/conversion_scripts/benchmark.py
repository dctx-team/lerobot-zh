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

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

torch.backends.cudnn.benchmark = True


def main():
    device = "cuda"
    dataset_repo_id = "danaaubakirova/koch_test"
    # model_name = "pi0_base"
    # ckpt_torch_dir = Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}_pytorch"
    ckpt_torch_dir = "lerobot/pi0"

    dataset = LeRobotDataset(dataset_repo_id, episodes=[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
    )

    batch = next(iter(dataloader))

    # 移动到设备
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)

    cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir)
    cfg.pretrained_path = ckpt_torch_dir
    policy = make_policy(cfg, ds_meta=dataset.meta)

    # policy = torch.compile(policy, mode="reduce-overhead")

    warmup_iters = 10
    benchmark_iters = 30

    # 预热
    for _ in range(warmup_iters):
        torch.cuda.synchronize()
        policy.select_action(batch)
        policy.reset()
        torch.cuda.synchronize()

    # 基准测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_iters):
        policy.select_action(batch)
        policy.reset()
    end_event.record()

    # 同步并测量时间
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    avg_time_per_iter = elapsed_time_ms / benchmark_iters
    print(f"每次迭代的平均执行时间: {avg_time_per_iter:.3f} ms")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
