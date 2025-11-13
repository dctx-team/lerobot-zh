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
import glob
import importlib
import logging
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import av
import fsspec
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    使用指定的后端解码视频帧。

    参数:
        video_path (Path): 视频文件的路径。
        timestamps (list[float]): 要提取的帧的时间戳列表。
        tolerance_s (float): 帧检索允许的容差(秒)。
        backend (str, optional): 用于解码的后端。平台可用时默认为"torchcodec"；否则默认为"pyav"。

    返回:
        torch.Tensor: 解码的帧。

    当前支持 torchcodec (CPU) 和 pyav。
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """加载视频中与请求时间戳相关联的帧

    后端可以是 "pyav"(默认)或 "video_reader"。
    "video_reader" 需要从源代码安装 torchvision，参见:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (注意需要针对 ffmpeg<4.3 编译)

    虽然两者都使用 CPU，但 "video_reader" 据说比 "pyav" 更快，但需要额外的设置。
    有关视频解码的更多信息，请参见 `benchmark/video/README.md`

    有关这两个后端的更多信息，请参见 torchvision 文档:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    注意: 视频受益于帧间压缩。编码器不是单独存储每一帧，而是存储一个参考帧(或关键帧)，
    后续帧存储为相对于该关键帧的差异。因此，要访问请求的帧，我们需要加载前面的关键帧，
    以及所有后续帧直到达到请求的帧。视频中的关键帧数量可以在编码期间调整，
    以考虑解码时间和视频大小(字节)。
    """
    video_path = str(video_path)

    # 设置后端
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav 不支持精确查找

    # 设置视频流读取器
    # TODO(rcadene): 同时加载音频流
    reader = torchvision.io.VideoReader(video_path, "video")

    # 设置第一个和最后一个请求的时间戳
    # 注意: 由于我们需要访问前一个关键帧，通常会加载之前的时间戳
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # 访问第一个请求帧的最近关键帧
    # 注意: 最近关键帧的时间戳通常小于 `first_ts`(例如关键帧可能是视频的第一帧)
    # 有关 `seek` 的详细信息，请参见: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # 加载所有帧直到最后一个请求的帧
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # 计算每个查询时间戳与所有已加载帧的时间戳之间的距离
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"一个或多个查询时间戳意外超出容差 ({min_[~is_within_tol]} > {tolerance_s=})。"
        "这意味着可以从视频加载的最近帧在时间上距离过远。"
        "这可能是由于数据收集期间时间戳同步问题造成的。"
        "为了安全起见，我们建议在训练期间忽略此项。"
        f"\n查询的时间戳: {query_ts}"
        f"\n已加载的时间戳: {loaded_ts}"
        f"\n视频: {video_path}"
        f"\n后端: {backend}"
    )

    # 获取与查询时间戳最接近的帧
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # 转换为 pytorch 格式，即 [0,1] 范围内的 float32(并且通道在前)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class VideoDecoderCache:
    """视频解码器的线程安全缓存，用于避免昂贵的重新初始化。"""

    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        """获取缓存的解码器或创建新的解码器。"""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[video_path] = (decoder, file_handle)

            return self._cache[video_path][0]

    def clear(self):
        """清空缓存并关闭文件句柄。"""
        with self._lock:
            for _, file_handle in self._cache.values():
                file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        """返回缓存的解码器数量。"""
        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):
    """辅助错误，用于指示检索到的时间戳超过查询的时间戳"""

    pass


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
) -> torch.Tensor:
    """使用 torchcodec 加载视频中与请求时间戳相关联的帧。

    参数:
        video_path: 视频文件的路径。
        timestamps: 要提取帧的时间戳列表。
        tolerance_s: 帧检索允许的偏差(秒)。
        log_loaded_timestamps: 是否记录已加载的时间戳。
        decoder_cache: 可选的解码器缓存实例。如果为 None 则使用默认值。

    注意: 在主进程之外设置 device="cuda"，例如在数据加载器工作进程中，将导致 CUDA 初始化错误。

    注意: 视频受益于帧间压缩。编码器不是单独存储每一帧，而是存储一个参考帧(或关键帧)，
    后续帧存储为相对于该关键帧的差异。因此，要访问请求的帧，我们需要加载前面的关键帧，
    以及所有后续帧直到达到请求的帧。视频中的关键帧数量可以在编码期间调整，
    以考虑解码时间和视频大小(字节)。
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # Use cached decoder instead of creating new one each time
    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    # 获取帧信息的元数据
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # 将时间戳转换为帧索引
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # 根据索引检索帧
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # 计算每个查询时间戳与已加载时间戳之间的距离
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"一个或多个查询时间戳意外超出容差 ({min_[~is_within_tol]} > {tolerance_s=})。"
        "这意味着可以从视频加载的最近帧在时间上距离过远。"
        "这可能是由于数据收集期间时间戳同步问题造成的。"
        "为了安全起见，我们建议在训练期间忽略此项。"
        f"\n查询的时间戳: {query_ts}"
        f"\n已加载的时间戳: {loaded_ts}"
        f"\n视频: {video_path}"
    )

    # 获取与查询时间戳最接近的帧
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # 转换为 [0,1] 范围内的 float32
    closest_frames = (closest_frames / 255.0).type(torch.float32)

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """有关 ffmpeg 参数调优的更多信息，请参见 `benchmark/video/README.md`"""
    # 检查编码器可用性
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # 编码器/像素格式不兼容性检查
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"像素格式 'yuv444p' 与编解码器 {vcodec} 不兼容，自动选择格式 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # 获取输入帧
    template = "frame-" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    # 定义视频输出帧大小(假设所有输入帧大小相同)
    if len(input_list) == 0:
        raise FileNotFoundError(f"在 {imgs_dir} 中未找到图像。")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # 定义视频编解码器选项
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # 设置日志级别
    if log_level is not None:
        # "虽然效率较低，但通常最好使用 Python 的 logging 来修改日志"
        logging.getLogger("libav").setLevel(log_level)

    # 创建并打开输出文件(默认覆盖)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # 遍历输入帧并编码
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # 刷新编码器
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # 重置日志级别
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def concatenate_video_files(
    input_video_paths: list[Path | str], output_video_path: Path, overwrite: bool = True
):
    """
    使用 pyav 将多个视频文件连接成单个视频文件。

    此函数接受视频输入文件路径列表，并将它们连接成单个输出视频文件。
    它使用 ffmpeg 的 concat 解复用器和流复制模式进行快速连接，无需重新编码。

    参数:
        input_video_paths: 要连接的输入视频文件路径的有序列表。
        output_video_path: 输出视频文件的路径。
        overwrite: 如果输出视频文件已存在，是否覆盖。默认为 True。

    注意:
        - 为中间文件创建临时目录，使用后会清理。
        - 使用 ffmpeg 的 concat 解复用器，要求所有输入视频具有相同的编解码器、
          分辨率和帧率，以正确连接。
    """

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("未提供输入视频路径。")

    # 创建临时 .ffconcat 文件以列出输入视频路径
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp_concatenate_file:
        tmp_concatenate_file.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp_concatenate_file.write(f"file '{str(input_path.resolve())}'\n")
        tmp_concatenate_file.flush()
        tmp_concatenate_path = tmp_concatenate_file.name

    # 创建输入和输出容器
    input_container = av.open(
        tmp_concatenate_path, mode="r", format="concat", options={"safe": "0"}
    )  # safe = 0 允许绝对路径和相对路径

    tmp_output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    output_container = av.open(
        tmp_output_video_path, mode="w", options={"movflags": "faststart"}
    )  # faststart 是将元数据移动到文件开头以加快加载速度

    # 在输出容器中复制输入流
    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):  # 仅复制兼容的流
            stream_map[input_stream.index] = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )
            stream_map[
                input_stream.index
            ].time_base = (
                input_stream.time_base
            )  # 将时基设置为输入流的时基(编解码器上下文中缺少)

    # 解复用 + 重新复用数据包(无重新编码)
    for packet in input_container.demux():
        # 跳过未映射流的数据包
        if packet.stream.index not in stream_map:
            continue

        # 跳过解复用刷新数据包
        if packet.dts is None:
            continue

        output_stream = stream_map[packet.stream.index]
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_output_video_path, output_video_path)
    Path(tmp_concatenate_path).unlink()


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): 移动到 Hugging Face `datasets` 仓库
    """
    为包含视频帧的数据集提供类型。

    示例:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # 使 VideoFrame 在 HuggingFace `datasets` 中可用
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # 设置日志级别
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # 获取音频流信息
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # 重置日志级别
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # 在理想的无损情况下：位深度 x 采样率 x 通道数 = 比特率。
        # 在实际压缩情况下，比特率根据压缩级别设置：比特率越低，压缩越多。
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # 每秒采样数
        # 在理想的无损情况下：每个采样的固定位数。
        # 在实际压缩情况下：每个采样的可变位数(通常减少以匹配给定的深度率)。
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # 重置日志级别
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # 设置日志级别
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # 获取视频流信息
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # 重置日志级别
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # 从 r_frame_rate 计算 fps
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # 重置日志级别
    av.logging.restore_default_callback()

    # 添加音频流信息
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # 灰度
    elif image.mode == "LA":
        return 2  # 灰度 + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("未知格式")


def get_video_duration_in_s(video_path: Path | str) -> float:
    """
    使用 PyAV 获取视频文件的持续时间(秒)。

    参数:
        video_path: 视频文件的路径。

    返回:
        视频的持续时间(秒)。
    """
    with av.open(str(video_path)) as container:
        # 获取第一个视频流
        video_stream = container.streams.video[0]
        # 计算持续时间：stream.duration * stream.time_base 得到持续时间(秒)
        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # 如果流持续时间不可用，则回退到容器持续时间
            duration = float(container.duration / av.time_base)
    return duration


class VideoEncodingManager:
    """
    上下文管理器，确保即使发生异常也能正确进行视频编码和数据清理。

    此管理器处理:
    - 当录制中断时对任何剩余片段进行批量编码
    - 清理中断片段的临时图像文件
    - 删除空的图像目录

    参数:
        dataset: LeRobotDataset 实例
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 处理尚未批量编码的剩余片段
        if self.dataset.episodes_since_last_encoding > 0:
            if exc_type is not None:
                logging.info("发生异常。在退出前编码剩余片段...")
            else:
                logging.info("录制已停止。编码剩余片段...")

            start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
            end_ep = self.dataset.num_episodes
            logging.info(
                f"编码剩余 {self.dataset.episodes_since_last_encoding} 个片段, "
                f"从片段 {start_ep} 到 {end_ep - 1}"
            )
            self.dataset._batch_save_episode_video(start_ep, end_ep)

        # 如果录制被中断则清理片段图像
        if exc_type is not None:
            interrupted_episode_index = self.dataset.num_episodes
            for key in self.dataset.meta.video_keys:
                img_dir = self.dataset._get_image_file_path(
                    episode_index=interrupted_episode_index, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    logging.debug(
                        f"清理片段 {interrupted_episode_index} 的中断图像，相机 {key}"
                    )
                    shutil.rmtree(img_dir)

        # 如果图像目录为空则清理
        img_dir = self.dataset.root / "images"
        # 检查是否有剩余的 PNG 文件
        png_files = list(img_dir.rglob("*.png"))
        if len(png_files) == 0:
            # 只有在没有剩余 PNG 文件时才删除图像目录
            if img_dir.exists():
                shutil.rmtree(img_dir)
                logging.debug("清理了空的图像目录")
        else:
            logging.debug(f"图像目录不为空，包含 {len(png_files)} 个 PNG 文件")

        return False  # 不要抑制原始异常
