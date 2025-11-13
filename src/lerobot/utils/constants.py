# 版权所有 2024 The HuggingFace Inc. team。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 有关许可证下权限和限制的具体语言，请参阅许可证。
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

ACTION = "action"
REWARD = "next.reward"
TRUNCATED = "next.truncated"
DONE = "next.done"

ROBOTS = "robots"
ROBOT_TYPE = "robot_type"
TELEOPERATORS = "teleoperators"

# 文件和目录
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"您设置了 'LEROBOT_HOME' 环境变量为 '{os.getenv('LEROBOT_HOME')}'。\n"
        "'LEROBOT_HOME' 已弃用，请改用 'HF_LEROBOT_HOME'。"
    )

# 缓存目录
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

# 校准目录
default_calibration_path = HF_LEROBOT_HOME / "calibration"
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", default_calibration_path)).expanduser()


# 流式数据集
LOOKBACK_BACKTRACKTABLE = 100
LOOKAHEAD_BACKTRACKTABLE = 100
