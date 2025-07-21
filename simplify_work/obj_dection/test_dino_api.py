import torch
import torchvision.transforms as transforms
from PIL import Image
from simplify_work.obj_dection.detector_api import GroundingDINOProcessor

# Step 1: 加载本地图片
img_path = "image_debug.png"  # 替换为你图片的路径
img = Image.open(img_path).convert("RGB")

# Step 2: 转换为Tensor并归一化 (0~1)
transform = transforms.Compose([
    transforms.ToTensor(),  # 自动归一化为[0, 1]
])
img_tensor = transform(img)  # shape: (3, H, W)

# Step 3: 构造batch（B=1）
rgb_batch = img_tensor.unsqueeze(0)         # shape: (1, 3, H, W)
B, C, H, W = rgb_batch.shape
depth_batch = torch.rand(B, 3, H, W)  # 随机浮点数 ∈ [0,1]，模拟深度图


task_batch = ["put the gripper near the sachet"]  # 示例任务描述

# Step 4: 初始化 GroundingDINOProcessor
processor = GroundingDINOProcessor(
    text_prompt="The Gripper And The Pyramid-Shaped Sachet",
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device="cpu"
)

# Step 5: 调用函数测试
result = processor.add_depth_info_to_task(rgb_batch, depth_batch, task_batch)
print("输出任务字符串：", result)
