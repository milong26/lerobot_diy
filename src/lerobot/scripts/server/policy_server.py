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

"""
Example:
```shell
python src/lerobot/scripts/server/policy_server.py \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch

# 确保能收到客户端发来的depth image
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import SUPPORTED_POLICIES
from lerobot.scripts.server.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
# 根据模型路径判断
from pathlib import Path

# mtask需要目标识别
from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor

class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        
        # self.obj_detector=None
        # 新增 统计推理次数
        self.inference_count = 0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    # 接收本地传过来的policy setting
    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        # 应该是包含'observation.images.side_depth
        print("服务器接收到的feature",self.lerobot_features)
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.perf_counter()

        # 根据模型的路径选择合适的modify_task策略
        model_path = policy_specs.pretrained_name_or_path
        model_dirname = Path(model_path).parts  # or use Path(model_path).name for last part
        model_name=exp_name = str(model_dirname[3])

        # 处理三种模型：baseline，mtask，mstate
        self.obj_detector = None
        self.add_location_to_state = ""
        self.language_tip_mode=""
        # 在推理之前处理
        if model_name.startswith("baseline"):
            print("baseline模型，不做额外处理")
        elif model_name.startswith("mtask_"):
            # task 模式
            lang_mode = model_name.split("mtask_")[1]  # 例如 relative / grid_2cm / grid_5cm
            self.language_tip_mode = lang_mode
            self.obj_detector = VisionProcessor(language_tip_mode=lang_mode)
            print(f"采用的 task 模式: {lang_mode}")
        elif model_name.startswith("mstate_"):
            # state 模式
            state_mode = model_name.split("mstate_")[1]  # pure / 5cm
            self.add_location_to_state = state_mode
            self.obj_detector = VisionProcessor()
            print(f"采用的 state 模式: {state_mode}")


        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()


    # 服务器接收并反序列化observation
    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.info(f"Observation #{obs_timestep} has been filtered out")
            # pass

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        # 当前timestamp已经预测过
        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        # observation和上一个近似。
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        # 如果需要用depth，self.policy_image_features里面就要加
        # 根据 obj_detector 是否存在选择 feature
        if self.obj_detector:
            # 新增 side_depth 的 PolicyFeature
            side_depth_feature = {
                "observation.images.side_depth": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 480, 640)
                )
            }
            # 合并原来的 policy_image_features 和新增 feature
            policy_image_features_to_use = {**self.policy_image_features, **side_depth_feature}
        else:
            policy_image_features_to_use = self.policy_image_features

        # print("使用的 image feature:", policy_image_features_to_use)

        # 调用 raw_observation_to_observation
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            policy_image_features_to_use,
            self.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    # 处理state的函数
    def _add_location_to_state(self, item,unit_mter=1):
        """从 item 中计算 gripper-object 相对坐标并拼接到 state"""
        orig_state = item["observation.state"]
        dx, dy, dz, flag = 0.0, 0.0, 0.0, 0.0
        if self.obj_detector:
            rgb = item["observation.images.side"]
            depth = item.get("observation.images.side_depth", None)
            points_3d = self.obj_detector.process_sample(rgb, depth)
            if points_3d and len(points_3d) >= 2:
                gripper_pos, object_pos = self.obj_detector.transform_camera_to_custom_coordsystem(points_3d)[:2]
                if gripper_pos is not None and object_pos is not None:
                    if unit_mter==1:
                        dx = object_pos[0] - gripper_pos[0]
                        dy = object_pos[1] - gripper_pos[1]
                        dz = object_pos[2] - gripper_pos[2]
                        flag = 1.0
                    else:
                        dx = round((object_pos[0] - gripper_pos[0])/unit_mter)
                        dy = round((object_pos[1] - gripper_pos[1])/unit_mter)
                        dz = round((object_pos[2] - gripper_pos[2])/unit_mter)
                        flag = 1.0
    
        merged_state = torch.cat([orig_state, torch.tensor([[dx, dy, dz, flag]], dtype=orig_state.dtype).to(self.device)], dim=1)
        item["observation.state"] = merged_state
        return item
    

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        preprocessing_time = time.perf_counter() - start_time

        self.last_processed_obs: TimedObservation = observation_t

        # 推理之前，处理task和state
        # 处理obs中的task
        # 在visionprocessor已经用language_tip_mode进行初始化了
        if self.language_tip_mode:
            task=observation["task"]
            colored_image=observation["observation.images.side"]
            depth_image=observation["observation.images.side_depth"]
            task_batch = [task]
            tasks=self.obj_detector.add_depth_info_to_task(colored_image,depth_image,task_batch)
            print("返回的tasks",tasks)
            task=tasks[0]
            observation["task"]=task
        self.logger.info(f'mtask修改obs.task内容变成：{observation["task"]}')
        # 处理obs中的state
        if self.add_location_to_state:
            if self.add_location_to_state=="pure_step2":
                observation = self._add_location_to_state(observation)
            elif self.add_location_to_state=="grid_5cm_step2_restart":
                observation =self._add_location_to_state(observation,unit_mter=0.05)
        self.logger.info(f'mstate修改obs.state内容：{observation["observation.state"]}')

        # 去掉多余的side_depth
        observation.pop("observation.images.side_depth", None)

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        # 输出总共的推理次数
        self.inference_count += 1
        self.logger.info(f"总共的推理次数: {self.inference_count}")

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms | "
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms | "
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms | "
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
