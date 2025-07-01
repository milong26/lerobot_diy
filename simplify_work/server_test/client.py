# client.py
import socket
import torch
from torchvision.transforms.functional import to_pil_image
import io
import time
def create_image_tensor(seed):
    torch.manual_seed(seed)
    return torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)

def tensor_to_png_bytes(tensor):
    buffer = io.BytesIO()
    to_pil_image(tensor).save(buffer, format='PNG')
    return buffer.getvalue()

HOST = '10.10.1.35'  # 替换为服务器 IP
PORT = 9000
print("本地，之前",time.time())
with socket.create_connection((HOST, PORT)) as s:
    for i in range(3):
        image_tensor = create_image_tensor(i)
        image_bytes = tensor_to_png_bytes(image_tensor)
        # 发送长度
        s.send(len(image_bytes).to_bytes(4, byteorder='big'))
        # 发送内容
        s.sendall(image_bytes)
        print(f"第 {i+1} 张图片发送完成")

print("所有图片发送完毕")
print("本地，之后",time.time())
