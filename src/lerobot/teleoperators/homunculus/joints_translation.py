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

# 食指展开系数
INDEX_SPLAY = 0.3
# 中指展开系数
MIDDLE_SPLAY = 0.3
# 无名指展开系数
RING_SPLAY = 0.3
# 小指展开系数
PINKY_SPLAY = 0.5


def get_ulnar_flexion(flexion: float, abduction: float, splay: float):
    """获取尺侧屈曲值。

    Args:
        flexion: 屈曲角度
        abduction: 外展角度
        splay: 展开系数

    Returns:
        尺侧屈曲值
    """
    return -abduction * splay + flexion * (1 - splay)


def get_radial_flexion(flexion: float, abduction: float, splay: float):
    """获取桡侧屈曲值。

    Args:
        flexion: 屈曲角度
        abduction: 外展角度
        splay: 展开系数

    Returns:
        桡侧屈曲值
    """
    return abduction * splay + flexion * (1 - splay)


def homunculus_glove_to_hope_jr_hand(glove_action: dict[str, float]) -> dict[str, float]:
    """将 Homunculus 手套动作转换为 Hope Jr 机械手动作。

    该函数将手套传感器的原始数据转换为机械手的关节位置命令。
    对于每个手指，它将屈曲和外展动作转换为机械手的桡侧和尺侧屈肌控制。

    Args:
        glove_action: 包含手套传感器数据的字典，键为关节名称，值为位置

    Returns:
        包含机械手关节位置命令的字典
    """
    return {
        "thumb_cmc.pos": glove_action["thumb_cmc.pos"],
        "thumb_mcp.pos": glove_action["thumb_mcp.pos"],
        "thumb_pip.pos": glove_action["thumb_pip.pos"],
        "thumb_dip.pos": glove_action["thumb_dip.pos"],
        "index_radial_flexor.pos": get_radial_flexion(
            glove_action["index_mcp_flexion.pos"], glove_action["index_mcp_abduction.pos"], INDEX_SPLAY
        ),
        "index_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["index_mcp_flexion.pos"], glove_action["index_mcp_abduction.pos"], INDEX_SPLAY
        ),
        "index_pip_dip.pos": glove_action["index_dip.pos"],
        "middle_radial_flexor.pos": get_radial_flexion(
            glove_action["middle_mcp_flexion.pos"], glove_action["middle_mcp_abduction.pos"], MIDDLE_SPLAY
        ),
        "middle_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["middle_mcp_flexion.pos"], glove_action["middle_mcp_abduction.pos"], MIDDLE_SPLAY
        ),
        "middle_pip_dip.pos": glove_action["middle_dip.pos"],
        "ring_radial_flexor.pos": get_radial_flexion(
            glove_action["ring_mcp_flexion.pos"], glove_action["ring_mcp_abduction.pos"], RING_SPLAY
        ),
        "ring_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["ring_mcp_flexion.pos"], glove_action["ring_mcp_abduction.pos"], RING_SPLAY
        ),
        "ring_pip_dip.pos": glove_action["ring_dip.pos"],
        "pinky_radial_flexor.pos": get_radial_flexion(
            glove_action["pinky_mcp_flexion.pos"], glove_action["pinky_mcp_abduction.pos"], PINKY_SPLAY
        ),
        "pinky_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["pinky_mcp_flexion.pos"], glove_action["pinky_mcp_abduction.pos"], PINKY_SPLAY
        ),
        "pinky_pip_dip.pos": glove_action["pinky_dip.pos"],
    }
