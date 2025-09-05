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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor

# https://github.com/huggingface/lerobot/issues/1377
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # train.py 开头
# import os

# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)


# 为了filter raw dataset
import json
from pathlib import Path


class FilteredBatchLoader:
    def __init__(self, dataloader, exclude_keys: list, obj_detector:VisionProcessor=None, save_task_path='modified_tasks_pure.jsonl'):
        self.dataloader = dataloader
        self.exclude_keys = set(exclude_keys)
        self.obj_detector = obj_detector
        self.save_task_path = Path(save_task_path) if save_task_path else None

        # # 暂时不用了
        if self.save_task_path:
            self.save_task_path.parent.mkdir(parents=True, exist_ok=True)
            self.task_f = open(self.save_task_path, "a")

    def __del__(self):
        if hasattr(self, "task_f") and not self.task_f.closed:
            self.task_f.close()

    def __iter__(self):
        for batch in self.dataloader:
            # Step 1: apply obj_detector before excluding keys
            if self.obj_detector is not None:
                images = batch.get("observation.images.side").cpu()
                depths = batch.get("observation.images.side_depth").cpu()
                tasks = batch.get("task")

                if images is not None and depths is not None and tasks is not None:
                    # 需要手动修改
                    new_tasks = self.obj_detector.add_depth_info_to_task(images, depths, tasks,["router","sticker"])
                    batch["task"] = new_tasks

                    if self.save_task_path:
                        ep_indices = batch.get("episode_index", [])
                        frame_indices = batch.get("frame_index", [])
                        for ep, frame, task in zip(ep_indices, frame_indices, new_tasks):
                            record = {
                                "episode_index": int(ep.item()),
                                "frame_index": int(frame.item()),
                                "task": task,
                            }
                            self.task_f.write(json.dumps(record) + "\n")

                    self.obj_detector.print_statistics()

            # Step 2: now filter excluded keys
            filtered_batch = {k: v for k, v in batch.items() if k not in self.exclude_keys}

            yield filtered_batch




def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    cfg.num_workers=1
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    # make_dataset接收的就是cfg
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # 开始
    logging.info("Creating policy")
    # policy = make_policy(
    #     cfg=cfg.policy,
    #     ds_meta=dataset.meta,
    # )

    logging.info("Creating optimizer and scheduler")
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    # 结束

    step = 0  # number of policy updates (forward + backward + optim)

    # 开始
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # num_total_params = sum(p.numel() for p in policy.parameters())

    # logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    # if cfg.env is not None:
    #     logging.info(f"{cfg.env.task=}")
    # logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    # logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    # logging.info(f"{dataset.num_episodes=}")
    # logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    # logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    # 结束

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    raw_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    #  构造 exclude list
    exclude_features = []
    # if not cfg.use_depth_image:
    #     exclude_features += ["observation.images.side_depth", "observation.images.side_depth_is_pad"]
    # if not cfg.use_force:
    #     exclude_features += ["observation.force", "observation.force_is_pad"]
    # obj_detector = None

    # if cfg.use_language_tip:
    
    obj_detector = VisionProcessor(language_tip_mode="training")

    # 包装 dataloader
    dataloader = FilteredBatchLoader(raw_dataloader, exclude_features, obj_detector=obj_detector)

    # 检查
    # peek_batch = next(iter(dataloader))
    # print("真正训练的时候甬道的feature：", list(peek_batch.keys()))
    # print("task示例",peek_batch["task"][0])
    # raise KeyError("输出检查")


    # start train
    dl_iter = cycle(dataloader)

    # policy.train()

    # train_metrics = {
    #     "loss": AverageMeter("loss", ":.3f"),
    #     "grad_norm": AverageMeter("grdn", ":.3f"),
    #     "lr": AverageMeter("lr", ":0.1e"),
    #     "update_s": AverageMeter("updt_s", ":.3f"),
    #     "dataloading_s": AverageMeter("data_s", ":.3f"),
    # }

    # train_tracker = MetricsTracker(
    #     cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    # )

    seen_99 = False
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        # start_time = time.perf_counter()
        batch = next(dl_iter)
        # train_tracker.dataloading_s = time.perf_counter() - start_time
        ep_indices = batch.get("episode_index", None)
        if ep_indices is not None:
                ep_np = ep_indices.cpu().numpy() if isinstance(ep_indices, torch.Tensor) else ep_indices

                if not seen_99 and 99 in ep_np:
                    seen_99 = True  # 第一次遇到99
                    print("第一次遇到 episode_index=99，开始关注后续的0")

                if seen_99 and 0 in ep_np:
                    print("在遇到99之后遇到0，停止训练")
                    break


        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        # print("task示例",batch["task"])

        # train_tracker, output_dict = update_policy(
        #     train_tracker,
        #     policy,
        #     batch,
        #     optimizer,
        #     cfg.optimizer.grad_clip_norm,
        #     grad_scaler=grad_scaler,
        #     lr_scheduler=lr_scheduler,
        #     use_amp=cfg.policy.use_amp,
        # )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
    #     train_tracker.step()
    #     is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
    #     is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
    #     is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

    #     if is_log_step:
    #         logging.info(train_tracker)
    #         if wandb_logger:
    #             wandb_log_dict = train_tracker.to_dict()
    #             if output_dict:
    #                 wandb_log_dict.update(output_dict)
    #             wandb_logger.log_dict(wandb_log_dict, step)
    #         train_tracker.reset_averages()

    #     if cfg.save_checkpoint and is_saving_step:
    #         logging.info(f"Checkpoint policy after step {step}")
    #         checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
    #         save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
    #         update_last_checkpoint(checkpoint_dir)
    #         if wandb_logger:
    #             wandb_logger.log_policy(checkpoint_dir)

    #     if cfg.env and is_eval_step:
    #         step_id = get_step_identifier(step, cfg.steps)
    #         logging.info(f"Eval policy at step {step}")
    #         with (
    #             torch.no_grad(),
    #             torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
    #         ):
    #             eval_info = eval_policy(
    #                 eval_env,
    #                 policy,
    #                 cfg.eval.n_episodes,
    #                 videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
    #                 max_episodes_rendered=4,
    #                 start_seed=cfg.seed,
    #             )

    #         eval_metrics = {
    #             "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
    #             "pc_success": AverageMeter("success", ":.1f"),
    #             "eval_s": AverageMeter("eval_s", ":.3f"),
    #         }
    #         eval_tracker = MetricsTracker(
    #             cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
    #         )
    #         eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
    #         eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
    #         eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
    #         logging.info(eval_tracker)
    #         if wandb_logger:
    #             wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
    #             wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
    #             wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # if eval_env:
    #     eval_env.close()
    # logging.info("End of training")

    # if cfg.policy.push_to_hub:
    #     policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    
    train()