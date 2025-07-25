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

# https://github.com/huggingface/lerobot/issues/1377
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 为了filter raw dataset
class FilteredBatchLoader:
    def __init__(self, dataloader, exclude_keys: list, obj_detector=None):
        self.dataloader = dataloader
        self.exclude_keys = set(exclude_keys)
        self.obj_detector = obj_detector  # 新增

    def __iter__(self):
        for batch in self.dataloader:
            # print("第几个batch")
            # 先过滤掉无用字段
            filtered_batch = {k: v for k, v in batch.items() if k not in self.exclude_keys}

            # 如果有 obj_detector，就处理每张图
            if self.obj_detector is not None:
                # images都是tensor格式的
                images = filtered_batch["observation.images.side"]            # [B, C, H, W]
                depths = filtered_batch["observation.images.side_depth"]      # [B, 1, H, W]
                tasks = filtered_batch["task"]                                 # list[str] or tensor of strings
                # new_tasks: list[str]，和原始 tasks 长度一致
                new_tasks = self.obj_detector.add_depth_info_to_task(images, depths, tasks)
                filtered_batch["task"] = new_tasks  # 覆盖原 task
                self.obj_detector.print_statistics()

            yield filtered_batch

    def __len__(self):
        return len(self.dataloader)



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
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # logging.info("Creating policy")
    # logging.info("Creating policy")
    # policy = make_policy(
    #     cfg=cfg.policy,
    #     ds_meta=dataset.meta,
    # )

    # logging.info("Creating optimizer and scheduler")
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    # if cfg.resume:
    #     step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

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
    # exclude_features = []
    # if not cfg.use_depth_image:
    #     exclude_features += ["observation.images.side_depth", "observation.images.side_depth_is_pad"]
    # if not cfg.use_force:
    #     exclude_features += ["observation.force", "observation.force_is_pad"]
    # obj_detector = None
    # if cfg.use_language_tip:
    #     from simplify_work.obj_dection.detector_api import GroundingDINOProcessor
    #     obj_detector = GroundingDINOProcessor(
    #         text_prompt="The Gripper And The Pyramid-Shaped Sachet",
    #         device=device.type,
    #     )

    """
    保存图片到本地
    """

    # # 创建保存目录
    # os.makedirs("outputs/check_channel", exist_ok=True)
    # import numpy as np

    # img_counter = 0
    # for batch_idx, batch in enumerate(raw_dataloader):
    #     if "observation.images.side_depth" in batch:
    #         images = batch["observation.images.side_depth"]  # [B, C, H, W]
            
    #         for i in range(images.shape[0]):
    #             img_tensor = images[i].detach().cpu()  # [C, H, W]
                
    #             # 1. 转换为NumPy数组并调整通道顺序 (C,H,W) -> (H,W,C)
    #             if img_tensor.shape[0] == 3:  # RGB图像
    #                 img_np = img_tensor.permute(1, 2, 0).numpy()
    #             else:  # 灰度图像或特殊情况
    #                 img_np = img_tensor.squeeze(0).numpy()  # [H, W]
                
    #             # 2. 检查并修复异常形状 (1,1,3)
    #             if img_np.shape == (1, 1, 3):
    #                 # 尝试恢复为原始尺寸（根据您的数据集实际情况修改尺寸）
    #                 original_height, original_width = 224, 224  # 示例尺寸，需替换为真实值
    #                 try:
    #                     img_np = img_np.reshape(original_height, original_width, 3)
    #                 except ValueError:
    #                     logging.warning(f"无法重塑图像 {img_counter}，跳过")
    #                     continue
                
    #             # 3. 归一化处理（支持float32和float64）
    #             if img_np.dtype in [np.float32, np.float64]:
    #                 if np.min(img_np) < 0:  # 值范围[-1,1]
    #                     img_np = (img_np + 1) * 127.5
    #                 elif np.max(img_np) <= 1.0:  # 值范围[0,1]
    #                     img_np = img_np * 255
    #                 # 确保值在0-255范围内
    #                 img_np = np.clip(img_np, 0, 255)
                
    #             # 4. 转换为uint8（PIL要求的数据类型）
    #             img_np = img_np.astype(np.uint8)
                
    #             # 5. 处理单通道灰度图
    #             if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
    #                 img_np = img_np.squeeze()  # 移除单通道维度
                
    #             # 6. 保存为PNG
    #             try:
    #                 img_path = os.path.join("outputs/foryolo", f"{img_counter:06d}.png")
    #                 Image.fromarray(img_np).save(img_path)
    #                 img_counter += 1
    #             except Exception as e:
    #                 logging.error(f"保存图像失败: {e}, 形状: {img_np.shape}, 类型: {img_np.dtype}")
    #                 continue
                    
    #     # 每10批次打印一次进度
    #     if batch_idx % 10 == 0:
    #         logging.info(f"已保存 {img_counter} 张图像...")
    # raise KeyError("yolo准备工作")




    # 包装 dataloader
    dataloader = FilteredBatchLoader(raw_dataloader, exclude_features, obj_detector=obj_detector)
    peek_batch = next(iter(dataloader))
    print("task示例",peek_batch["task"])
    # print("真正训练的时候甬道的feature：", list(peek_batch.keys()))
    raise KeyError("测试task")
    print(peek_batch["observation.images.side_depth"].shape)
    import cv2
    import numpy as np
    from PIL import Image
    r_count=0
    b_count=0
    for i in range(64):
        image_tensor=peek_batch["observation.images.side_depth"][i]
        image_np = (image_tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rgb_image=image_np
    #     # rgb_image 是形状为 (H, W, 3)，类型为 uint8，且通道顺序是 RGB
    #     img = Image.fromarray(rgb_image)  # 自动识别为 RGB 模式
    #     img.save(f"from_videos/{i}.png")
    #     print(f"保存成功{i}.png")
    # depth_encoded_path='frame_000000.png'
    
    # cv2_img = cv2.imread(depth_encoded_path, cv2.IMREAD_UNCHANGED)

    # 通道调换顺序比较
    # image_np_bgr = image_np[..., ::-1]  # RGB to BGR

    # 直接比较差异
    # 依次比较每个通道
    

        r = rgb_image[:, :, 0].astype(np.uint16)
        g = rgb_image[:, :, 1].astype(np.uint16)
        b = rgb_image[:, :, 2].astype(np.uint16)
        print("r=",r[220][230])
        print("g=",b[220][230])
    # g_corrected = (g // 8) * 4  # 对低位进行4的倍数截断，抵消抖动
    # depth_smooth = (r << 8) | g_corrected

    # depth_uint16 =  g.astype(np.uint16)
    # depth_r_only1 = (rgb_image[:, :, 0].astype(np.uint16)) << 8


    # b, g, r = cv2.split(cv2_img)
    # print(r[220][230])
    # print(b[220][230])
    # depth_uint162 = (r << 8) | g.astype(np.uint16)
    # depth_r_only2 = (cv2_img[:, :, 0].astype(np.uint16)) << 8
    # import numpy as np

    # # 1. 绝对误差图
    # diff = np.abs(cv2_img.astype(np.int16) - rgb_image.astype(np.int16))
    # diff_gray = np.mean(diff, axis=2)  # 转灰度看差异强度
    # abs_diff = np.abs(depth_uint16.astype(np.int32) - depth_uint162.astype(np.int32))
    # abs_diffabs_diff_r_only=np.abs(depth_smooth.astype(np.int32) - depth_uint162.astype(np.int32))

    # 2. 打印统计量
    print("绝对误差统计:")
    print(f"最大误差: {abs_diff.max()}")
    print(f"平均误差: {abs_diff.mean():.2f}")
    print(f"误差为0的像素占比: {(abs_diff == 0).sum() / abs_diff.size:.2%}")
    print(f"误差大于10的像素占比: {(abs_diff > 10).sum() / abs_diff.size:.2%}")
    
    raise KeyError("stop")

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

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        continue
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()