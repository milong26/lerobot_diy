# 测试scaled_relative，如果成功率很低的话，那就是不能用relative做
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
import time

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor
print("路径",gym_pusht.__path__)
##===================环境准备====================
#输出目录，用来保存输出(我用的绝对路径)
# output_directory = Path("for_pusht/relative/output")
# baseline的测试
# raise KeyError("没改")
output_directory = Path("for_pusht/926output/old_scaled_120pixel") 
# relative的测试
# output_directory = Path("for_pusht/mytrain_result_400time/relative") 
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# pretrained_policy_path = "lerobot/diffusion_pusht"
# OR a path to a local outputs/train folder.
# 选择模型文件夹(我也用的绝对路径)
# pretrained_policy_path = Path("for_pusht/train/checkpoints/last/pretrained_model")
# raise KeyError("没改")
pretrained_policy_path = Path("for_pusht/train_0916/mtask_old_scaled_120pixel/checkpoints/026000/pretrained_model")
policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 500 interactions/steps.
#创建pusht环境，env详细参数信息在：https://github.com/huggingface/gym-pusht/
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos", # 在此type下，可以从env得到 ：[pixels]->图像 + [agent_pos]->位置(x,y)
    max_episode_steps=1000,    #最大步数

    #observation_width = 96     #img宽，默认96
    #observation_height = 96    #img高，默认96
)
# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print("input_feature: ")
print(policy.config.input_features)
print(env.observation_space)
# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print("output_feature: ")
print(policy.config.output_features)
print(env.action_space)
#===========================================
# raise KeyError("没改")
obj_detector= VisionProcessor(language_tip_mode="120pixel")

##==================评估=====================
# 运行的轮数(评估次数)
num_rollouts = 100
#设置task
# task = "Push the T-shaped block onto the T-shaped target"
task= "Push the grey T-shaped block onto the green T-shaped target"
task=[task]
#设置info频率(比如：Rollout 19 | step=300  reward=np.float64(0.87)  terminated=False)
log_freq=100
# 创建成功和失败的保存目录
true_dir = output_directory / "True"
false_dir = output_directory / "False"
true_dir.mkdir(parents=True, exist_ok=True)
false_dir.mkdir(parents=True, exist_ok=True)
# 结果统计文件
results_file = output_directory / "results.txt"

with open(results_file, "w") as f:
    # 在txt文件内输出提示信息
    f.write(f"Rollout Results (Total: {num_rollouts})\n")
    f.write("=" * 50 + "\n")
    # 成功和失败计数
    success_count = 0
    failure_count = 0
    printed_task=False



    #开始num_rollouts轮测试
    for i in range(num_rollouts):
        # torch.cuda.empty_cache()
        print(f"Starting rollout {i + 1}/{num_rollouts}")
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        numpy_observation, info = env.reset()

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []
        frames = []

        # Render frame of the initial state
        frames.append(env.render())

        step = 0
        done = False
        
with open(results_file, "w") as f:
    f.write(f"Rollout Results (Total: {num_rollouts})\n")
    f.write("=" * 50 + "\n")
    success_count = 0
    failure_count = 0
    printed_task = False

    # 全局时间统计
    total_img_proc_time = 0.0
    total_infer_time = 0.0
    total_steps = 0

    for i in range(num_rollouts):
        policy.reset()
        numpy_observation, info = env.reset()

        rewards = []
        frames = []
        frames.append(env.render())

        step = 0
        done = False

        # rollout 内时间统计
        rollout_img_proc_time = 0.0
        rollout_infer_time = 0.0

        while not done:
            state = torch.from_numpy(numpy_observation["agent_pos"])
            image = torch.from_numpy(numpy_observation["pixels"])
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # ==== 图像处理计时 ====
            t0 = time.perf_counter()
            new_tasks = obj_detector.add_2d_position_to_task(
                image, task, state, ["grey", "green"]
            )
            t1 = time.perf_counter()
            img_proc_time = (t1 - t0) * 1000  # ms

            # ==== 推理计时 ====
            new_task = new_tasks[0]
            state = state.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            observation = {
                "observation.state": state,
                "observation.image": image,
                "task": [new_task],
            }
            t2 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(observation)
            t3 = time.perf_counter()
            infer_time = (t3 - t2) * 1000  # ms

            # 累积时间
            rollout_img_proc_time += img_proc_time
            rollout_infer_time += infer_time
            total_img_proc_time += img_proc_time
            total_infer_time += infer_time
            total_steps += 1



            numpy_action = action.squeeze(0).to("cpu").numpy()
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            # 日志
            if step % log_freq == 0:
                print(
                    f"Rollout {i} | step={step} reward={reward:.3f} terminated={terminated} "
                    f"| img_proc={img_proc_time:.2f} ms, infer={infer_time:.2f} ms"
                )
            rewards.append(reward)
            frames.append(env.render())
            done = terminated | truncated | done
            step += 1

        # rollout 结束，计算平均耗时
        avg_img_proc = rollout_img_proc_time / step
        avg_infer = rollout_infer_time / step
        f.write(f"Rollout {i}: avg img_proc={avg_img_proc:.2f} ms, avg infer={avg_infer:.2f} ms\n")

        if terminated:
            result_str = f"Rollout {i}: Success! Total reward: {sum(rewards)}"
            success_count += 1
            video_path = true_dir / f"rollout_{i}.mp4"
        else:
            result_str = f"Rollout {i}: Failure! Total reward: {sum(rewards)}"
            failure_count += 1
            video_path = false_dir / f"rollout_{i}.mp4"

        print(result_str)
        f.write(result_str + "\n")
        imageio.mimsave(str(video_path), numpy.stack(frames), fps=env.metadata["render_fps"])

    # 全局平均时间
    global_avg_img_proc = total_img_proc_time / total_steps
    global_avg_infer = total_infer_time / total_steps

    f.write("\n" + "=" * 50 + "\n")
    f.write("Final Statistics:\n")
    f.write(f"Total Rollouts: {num_rollouts}\n")
    f.write(f"Successful Rollouts: {success_count}\n")
    f.write(f"Failed Rollouts: {failure_count}\n")
    f.write(f"Global Avg img_proc: {global_avg_img_proc:.2f} ms\n")
    f.write(f"Global Avg infer: {global_avg_infer:.2f} ms\n")

    print("\n" + "=" * 50)
    print("Final Statistics:")
    print(f"Total Rollouts: {num_rollouts}")
    print(f"Successful Rollouts: {success_count}")
    print(f"Failed Rollouts: {failure_count}")
    print(f"Global Avg img_proc: {global_avg_img_proc:.2f} ms")
    print(f"Global Avg infer: {global_avg_infer:.2f} ms")

env.close()
##=======================================