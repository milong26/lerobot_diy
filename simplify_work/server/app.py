# import time
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Dict, List
# import base64
# from PIL import Image
# import io
# import torch
# import numpy as np
# from model_loader import get_model

# app = FastAPI()



# # """
# # 导入模型
# # """
# model=get_model()
# model.eval().to("cuda")

# # 先推理一次
# batch = {
#     "observation.state": torch.rand(1, 6).to("cuda"),
#     "observation.force": torch.rand(1, 15).to("cuda"),
#     "observation.images.scene": torch.rand(1, 3, 480, 640).to("cuda"),
#     "observation.images.wrist": torch.rand(1, 3, 480, 640).to("cuda"),
#     "observation.images.sceneDepth": torch.rand(1, 3,  480, 640).to("cuda"),
# }
# action = model(batch)



# class RequestData(BaseModel):
#     tensors: Dict[str, List[float]]  # 无 batch 维度，直接普通 list
#     image_wrist: str  # base64 编码列表
#     image_sceneDepth: str  # base64 编码列表
#     image_scene: str  # base64 编码列表

# def decode_tensor_image(base64_str: str) -> torch.Tensor:
#     binary = base64.b64decode(base64_str)
#     buffer = io.BytesIO(binary)
#     # 使用 torch.load 还原
#     tensor_list = torch.load(buffer)
#     return tensor_list


# def serialize_tensor_tuple(tensor):
#     # Save tensor to bytes
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)
#     buffer.seek(0)
    
#     # Encode to base64 string
#     b64_data = base64.b64encode(buffer.read()).decode('utf-8')
#     return b64_data

# def detach_and_cpu(obj):
#     if isinstance(obj, torch.Tensor):
#         return obj.detach().cpu()
#     elif isinstance(obj, (list, tuple)):
#         return type(obj)(detach_and_cpu(item) for item in obj)
#     elif isinstance(obj, dict):
#         return {k: detach_and_cpu(v) for k, v in obj.items()}
#     else:
#         return obj  # e.g. None, numbers, strings...

# @app.post("/predict")
# async def predict(request_data: RequestData):
#     # tensors 转成无 batch 维度 tensor
#     tensors = {
#         k: torch.tensor(v).unsqueeze(0)  # 加回 batch 维度 (1, D)
#         for k, v in request_data.tensors.items()
#     }
#     # images 解码成 list of (3,H,W)
#     print("images之前:",time.time())
#     wrist_images = decode_tensor_image(request_data.image_wrist)
#     scene_images = decode_tensor_image(request_data.image_scene)
#     sceneDepth_images = decode_tensor_image(request_data.image_sceneDepth)
#     print("images之后:",time.time())
    

#     # 组装 batch，保持跟客户端一样格式
#     batch = {
#         "observation.state": tensors["observation.state"].to("cuda"),  # (1, 6)
#         "observation.force": tensors["observation.force"].to("cuda"), # (1, 15)
#         "observation.images.scene":scene_images.to("cuda"),
#         "observation.images.wrist":wrist_images.to("cuda"),
#         "observation.images.sceneDepth":sceneDepth_images.to("cuda"),                 # list of (3,H,W)
#     }
#     image_features=["observation.images.scene","observation.images.wrist","observation.images.sceneDepth"]
#     # batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
#     batch["observation.images"] = [batch[key] for key in image_features]

#     # print(batch["observation.images.scene"])

#     # 这里调用你的模型，示例：
#     # with torch.no_grad():
#     # print("模型之前",time.time())
#     with torch.no_grad():
#         action = model(batch)
#     # print("模型之后",time.time())

#     action = detach_and_cpu(action)
  
#     action=serialize_tensor_tuple(action)
#     return {"action": action}


import socket
import threading
import torch
from PIL import Image
import io
import time
import base64
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
from lerobot_diy.simplify_work.server.server_code.model_loader import get_model

app = FastAPI()

# 模型准备
model = get_model()
model.eval().to("cuda")

# 模拟初始推理（warm-up）
batch = {
    "observation.state": torch.rand(1, 6).to("cuda"),
    "observation.force": torch.rand(1, 15).to("cuda"),
    "observation.images.scene": torch.rand(1, 3, 480, 640).to("cuda"),
    "observation.images.wrist": torch.rand(1, 3, 480, 640).to("cuda"),
    "observation.images.sceneDepth": torch.rand(1, 3, 480, 640).to("cuda"),
}
batch["observation.images"] = [batch["observation.images.scene"], batch["observation.images.wrist"], batch["observation.images.sceneDepth"]]

action = model(batch)

# 用于缓存图片
image_cache = {}
image_lock = threading.Lock()

# FastAPI 结构体
class RequestData(BaseModel):
    tensors: Dict[str, List[float]]

def deserialize_action(action: torch.Tensor):
    buffer = io.BytesIO()
    torch.save(action, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# def load_image_from_bytes(data: bytes) -> torch.Tensor:
#     image = Image.open(io.BytesIO(data)).convert('RGB')
#     tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
#     tensor = tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
#     return tensor

# import numpy as np

# def load_image_from_bytes(data: bytes) -> torch.Tensor:
#     image = Image.open(io.BytesIO(data)).convert('RGB')
#     np_image = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
#     tensor = torch.from_numpy(np_image).permute(2, 0, 1)  # -> (3, H, W)
#     return tensor

import numpy as np

def load_image_from_bytes(data: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(data)).convert('RGB')
    np_img = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)   # (3, H, W)
    return tensor


@app.post("/predict")
async def predict(request_data: RequestData):
    print("[服务器] 接收到 force/state 时间:", time.time())
    tensors = {
        k: torch.tensor(v).unsqueeze(0).to("cuda")
        for k, v in request_data.tensors.items()
    }

    # 等待图像到达
    print("[服务器] 等待图片锁...", time.time())
    with image_lock:
        if len(image_cache) < 3:
            print("[服务器] 图片尚未全部接收，返回失败")
            return {"error": "images not ready"}

        print("[服务器] 所有图片接收完毕，准备组装 batch")
        scene = image_cache.pop("scene")
        wrist = image_cache.pop("wrist")
        sceneDepth = image_cache.pop("sceneDepth")

    batch = {
        "observation.state": tensors["observation.state"],
        "observation.force": tensors["observation.force"],
        "observation.images.scene": scene.unsqueeze(0).to("cuda"),
        "observation.images.wrist": wrist.unsqueeze(0).to("cuda"),
        "observation.images.sceneDepth": sceneDepth.unsqueeze(0).to("cuda"),
    }
    batch["observation.images"] = [batch["observation.images.scene"], batch["observation.images.wrist"], batch["observation.images.sceneDepth"]]

    with torch.no_grad():
        action = model(batch)
    print("[服务器] 模型推理完成", time.time())

    return {"action": deserialize_action(action)}

def tcp_image_server(host='0.0.0.0', port=9000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[TCP 服务器] 正在监听: {host}:{port}")

    while True:
        conn, addr = server_socket.accept()
        print(f"[TCP 服务器] 客户端连接: {addr}")
        try:
            for name in ["scene", "wrist", "sceneDepth"]:
                length_bytes = conn.recv(4)
                if not length_bytes:
                    raise ConnectionError("接收长度失败")
                length = int.from_bytes(length_bytes, 'big')
                print(f"[TCP 服务器] 即将接收 {name}, 大小: {length}")
                buffer = b''
                while len(buffer) < length:
                    data = conn.recv(length - len(buffer))
                    if not data:
                        raise ConnectionError("接收数据失败")
                    buffer += data
                with image_lock:
                    image_cache[name] = load_image_from_bytes(buffer)
                print(f"[TCP 服务器] {name} 图片接收完毕")
            conn.close()
        except Exception as e:
            print(f"[TCP 服务器] 错误: {e}")
            conn.close()

if __name__ == '__main__':
    tcp_thread = threading.Thread(target=tcp_image_server, daemon=True)
    tcp_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)