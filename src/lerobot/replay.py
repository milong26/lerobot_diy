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

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = False
    # skip设置
    skip_step: int=1


# @draccus.wrap()
# def replay(cfg: ReplayConfig):
#     init_logging()
#     logging.info(pformat(asdict(cfg)))

#     robot = make_robot_from_config(cfg.robot)
#     dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
#     actions = dataset.hf_dataset.select_columns("action")
#     robot.connect()
#     # 前多少次计数
#     """
#     replay的时候可以跳步
#     """
#     skip_step = cfg.skip_step
#     warmup_merge_steps = 5
#     send_count = 0
#     frame_idx = 0

#     log_say("Replaying episode", cfg.play_sounds, blocking=True)
#     # for idx in range(dataset.num_frames):
#     #     start_episode_t = time.perf_counter()

#     #     action_array = actions[idx]["action"]
#     #     action = {}
#     #     for i, name in enumerate(dataset.features["action"]["names"]):
#     #         action[name] = action_array[i]

#     #     robot.send_action(action)

#     #     dt_s = time.perf_counter() - start_episode_t
#     #     busy_wait(1 / dataset.fps - dt_s)

#     # robot.disconnect()

#     while frame_idx < dataset.num_frames:
#         start_episode_t = time.perf_counter()

#         if skip_step > 1 and send_count < warmup_merge_steps:
#             # Warmup 合并阶段：取 skip_step 个 action 加权平均
#             merged_action = {}
#             valid_skip = min(skip_step, dataset.num_frames - frame_idx)
#             weight = 1.0 / valid_skip

#             # 初始化 merged_action
#             for name in dataset.features["action"]["names"]:
#                 merged_action[name] = 0.0

#             for offset in range(valid_skip):
#                 action_array = actions[frame_idx + offset]["action"]
#                 for i, name in enumerate(dataset.features["action"]["names"]):
#                     merged_action[name] += action_array[i] * weight

#             frame_idx += valid_skip
#             send_count += 1
#             robot.send_action(merged_action)

#         else:
#             print("正常")
#             # 正常逐帧
#             action_array = actions[frame_idx]["action"]
#             action = {}
#             for i, name in enumerate(dataset.features["action"]["names"]):
#                 action[name] = action_array[i]

#             robot.send_action(action)
#             frame_idx += 1

#         # 控制帧率
#         dt_s = time.perf_counter() - start_episode_t
#         busy_wait(1 / dataset.fps - dt_s)

#     robot.disconnect()

@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")
    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action_array = actions[idx]["action"]
        action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            action[name] = action_array[i]

        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / dataset.fps - dt_s)

    robot.disconnect()


def main():
    replay()


if __name__ == "__main__":
    main()
