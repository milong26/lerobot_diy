# server.py
import socket
import torch
from PIL import Image
import io
from torchvision.transforms.functional import to_tensor

def receive_all(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError("连接中断")
        data += packet
    return data

HOST = '0.0.0.0'
PORT = 9000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"服务器监听中: {HOST}:{PORT}")

conn, addr = server_socket.accept()
print(f"客户端连接: {addr}")

try:
    image_index = 0
    while True:
        # 读取 4 字节长度
        length_bytes = conn.recv(4)
        if not length_bytes:
            print("客户端关闭连接")
            break

        image_len = int.from_bytes(length_bytes, byteorder='big')
        print(f"即将接收第 {image_index+1} 张图片，大小: {image_len} 字节")

        image_data = receive_all(conn, image_len)
        print(f"第 {image_index+1} 张图片接收完成")

        # 转换为 Tensor
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = to_tensor(image) * 255
        image_tensor = image_tensor.to(torch.uint8)

        print(f"还原 Tensor 形状: {image_tensor.shape}")
        image_index += 1

except Exception as e:
    print("发生异常:", e)

finally:
    conn.close()
    server_socket.close()
