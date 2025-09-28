from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor

##===================环境准备====================
#输出目录，用来保存输出(我用的绝对路径)
# output_directory = Path("for_pusht/relative/output")
# baseline的测试
# raise KeyError("没改")
output_directory = Path("for_pusht/mytrain_result_final_just/10pixel") 
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
pretrained_policy_path = Path("for_pusht/train_just/mtask_10pixel/checkpoints/026000/pretrained_model")
# pretrained_policy_path = Path("for_pusht/train/relative/checkpoints/026000/pretrained_model")
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
obj_detector= VisionProcessor(language_tip_mode="10pixel")

##==================评估=====================
# 运行的轮数(评估次数)
num_rollouts = 500
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

        while not done:
            # Prepare observation for the policy running in Pytorch
            state = torch.from_numpy(numpy_observation["agent_pos"])
            image = torch.from_numpy(numpy_observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)



            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)


            # relative
            new_tasks = obj_detector.add_2d_position_to_task(image,task,state,["grey","green"],)
            # new_task=new_tasks[0]
            # baseline
            new_task=new_tasks[0]
            # Send data tensors from CPU to GPU
            state = state.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image": image,
                "task": [new_task],
            }

            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare the action for the environment
            numpy_action = action.squeeze(0).to("cpu").numpy()

            # Step through the environment and receive a new observation
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            if step % log_freq == 0:
                print(f"Rollout {i} | {step=} {reward=} {terminated=}")
            # Keep track of all the rewards and frames
            rewards.append(reward)
            frames.append(env.render())

            # The rollout is considered done when the success state is reached (i.e. terminated is True),
            # or the maximum number of iterations is reached (i.e. truncated is True)
            done = terminated | truncated | done
            step += 1

        if terminated:  #成功时处理
            result_str = f"Rollout {i}: Success! Total reward: {sum(rewards)}"
            print(result_str)
            f.write(result_str + "\n")
            video_path = true_dir / f"rollout_{i}.mp4"
            success_count+=1
        else:           #失败时处理
            result_str = f"Rollout {i}: Failure! Total reward: {sum(rewards)}"
            print(result_str)
            f.write(result_str + "\n")
            video_path = false_dir / f"rollout_{i}.mp4"
            failure_count+=1
        print(f"成功{success_count}，失败{failure_count}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.metadata["render_fps"]

        # Encode all frames into a mp4 video.
        imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

        print(f"Saved video for rollout {i} at '{video_path}'")
    #统计结果
    success_rate = success_count / num_rollouts * 100.0
    failure_rate = failure_count / num_rollouts * 100.0
    #打印结果到txt
    f.write("\n" + "=" * 50 + "\n")
    f.write("Final Statistics:\n")
    f.write(f"Total Rollouts: {num_rollouts}\n")
    f.write(f"Successful Rollouts: {success_count} ({success_rate:.2f}%)\n")
    f.write(f"Failed Rollouts: {failure_count} ({failure_rate:.2f}%)\n")
    #打印结果到控制台
    print("\n" + "=" * 50)
    print("Final Statistics:")
    print(f"Total Rollouts: {num_rollouts}")
    print(f"Successful Rollouts: {success_count} ({success_rate:.2f}%)")
    print(f"Failed Rollouts: {failure_count} ({failure_rate:.2f}%)")

env.close()
##=======================================