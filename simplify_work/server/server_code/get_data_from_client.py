# server/server_main.py

from copy import copy
from image_receiver import start_image_server
from handler import start_data_server
from model_loader import load_policy
import torch
import time

# 预热
def load_and_warmup_model():
    print("[Init] Loading model...")
    model = load_policy()

    # 模拟 warm-up
    import numpy as np

    observation = {
        "observation.state": np.random.rand(6).astype(np.float32),  # shape: [6]
        "observation.images.side": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),  # shape: [480, 640, 3]
        "observation.images.wrist": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),  # shape: [480, 640, 3]
        "task": "pick up the orange tomato and place it into the box.",  # string
        "robot_type": "so100_follower"  # string
    }

    # optional: 添加 images 列表（如果模型需要）
    observation["observation.images"] = [
        observation["observation.images.side"],
        observation["observation.images.wrist"]
    ]
    
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda") 
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if isinstance(observation[name], np.ndarray):  # ✅ 只处理 numpy 数组
                observation[name] = torch.from_numpy(observation[name])
                if "image" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = observation[name].unsqueeze(0)
                observation[name] = observation[name].to("cuda")
        



        # Compute the next action with the policy
        # based on the current observation


        """
        使用服务器推理。给服务器传observation，接收服务器的action
        """

        # 原来本地调用polciy

    with torch.no_grad():
        _ = model(observation)

    print("[Init] Model loaded and warmed up.")
    return model


if __name__ == '__main__':
    model = load_and_warmup_model()
    host = '10.10.1.35'
    start_image_server(host=host, port=9100)
    start_data_server(host=host, port=9101, model=model)
    print("[Main] Server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")
