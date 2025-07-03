import torch
import time
from lerobot.common.policies.act.modeling_act import ACT
from model_loader import get_model

# 加载模型
model = get_model()
model.eval().to("cuda")

# 构造假的输入 batch，结构要与真实数据一致
fake_batch = {
    "observation.state": torch.rand(1, 6).to("cuda"),
    "observation.force": torch.rand(1, 15).to("cuda"),
    "observation.images.scene": torch.rand(1, 3, 480, 640).to("cuda"),
    "observation.images.wrist": torch.rand(1, 3, 480, 640).to("cuda"),
    "observation.images.sceneDepth": torch.rand(1, 3,  480, 640).to("cuda"),
    "observation.images": [torch.rand(1,3,  480, 640).to("cuda") for _ in range(3)],  # list of images
}

# 测试 N 次推理时间
N = 10
total_time = 0.0

with torch.no_grad():
    for _ in range(N):
        start = time.time()
        output = model(fake_batch)
        end = time.time()
        print(_,"次数:",end - start)
        total_time += (end - start)
        fake_batch = {
            "observation.state": torch.rand(1, 6).to("cuda"),
            "observation.force": torch.rand(1, 15).to("cuda"),
            "observation.images.scene": torch.rand(1, 3, 480, 640).to("cuda"),
            "observation.images.wrist": torch.rand(1, 3, 480, 640).to("cuda"),
            "observation.images.sceneDepth": torch.rand(1, 3,  480, 640).to("cuda"),
            "observation.images": [torch.rand(1,3,  480, 640).to("cuda") for _ in range(3)],  # list of images
        }
        time.sleep(2)
        

average_time = total_time / N
print(f"平均推理耗时: {average_time:.4f} 秒 (共 {N} 次)")
