
# client_multi_thread.py
import socket
import torch
import threading
import io
import json
import time
import requests
from torchvision.transforms.functional import to_pil_image

# def encode_tensor_image(tensor: torch.Tensor) -> bytes:
#     buffer = io.BytesIO()
#     to_pil_image(tensor).save(buffer, format='PNG')  # PNG 无损压缩
#     return buffer.getvalue()

def encode_tensor_as_image(tensor: torch.Tensor, format='JPEG') -> bytes:
    """将 (3, H, W) 的 tensor 转为压缩图像格式的 bytes"""
    from torchvision.transforms.functional import to_pil_image
    image = to_pil_image(tensor)
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=85)  # JPEG/WebP 质量可以调
    return buffer.getvalue()

def send_images(image_dict, host='10.10.1.35', port=9000):
    try:
        start = time.time()
        with socket.create_connection((host, port)) as s:
            for key in ["scene", "wrist", "sceneDepth"]:
                image_bytes = encode_tensor_as_image(image_dict[key])
                s.send(len(image_bytes).to_bytes(4, 'big'))
                s.sendall(image_bytes)
                print(f"[客户端] 第 {key} 张图片发送完成, 大小: {len(image_bytes)} 字节")
        end = time.time()
        print(f"[客户端] 所有图片发送完毕，用时: {end - start:.3f} 秒")
    except Exception as e:
        print("[客户端] 发送图片失败:", e)

def send_tensors(tensor_dict, server_url="http://10.10.1.35:8000/predict"):
    try:
        payload = {
            "tensors": {
                "observation.state": tensor_dict["state"].tolist(),
                "observation.force": tensor_dict["force"].tolist(),
            }
        }
        headers = {"Content-Type": "application/json"}
        start = time.time()
        response = requests.post(server_url, headers=headers, data=json.dumps(payload))
        end = time.time()
        if response.status_code == 200:
            print(f"[客户端] 收到响应, 用时: {end - start:.3f} 秒")
            action = torch.load(io.BytesIO(base64.b64decode(response.json()["action"])))
            print("[客户端] 解码后的 action:", action)
            return action
        else:
            print(f"[客户端] 请求失败: {response.status_code} {response.text}")
    except Exception as e:
        print("[客户端] 发送 tensors 失败:", e)

def main():
    # 示例数据
    batch = {
        "observation.state": torch.rand(6),
        "observation.force": torch.rand(15),
        "observation.images.scene": torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8),
        "observation.images.wrist": torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8),
        "observation.images.sceneDepth": torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8),
    }
    
    image_dict = {
        "scene": batch["observation.images.scene"],
        "wrist": batch["observation.images.wrist"],
        "sceneDepth": batch["observation.images.sceneDepth"],
    }

    tensor_dict = {
        "state": batch["observation.state"],
        "force": batch["observation.force"]
    }

    # 启动两个线程并行发送
    t1 = threading.Thread(target=send_images, args=(image_dict,))
    t2 = threading.Thread(target=send_tensors, args=(tensor_dict,))

    print("[客户端] 开始发送数据")
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("[客户端] 本轮数据发送完毕")

if __name__ == '__main__':
    main()
