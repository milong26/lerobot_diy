# control_utils里面调用，代替原来的action = policy.select_action(observation)
# 传入policy,observation
"""
policy暂时传不过来，以后再搞，最好是能把path和type传过来

{'observation.state': tensor([[ ]]),
 'observation.force': tensor([[]]), 
'observation.images.side': tensor([[[[]]]]),
'observation.images.side_depth': tensor([[[[]]]]), 
'observation.images.wrist': tensor([[[[]]]]),
 'task': '', 
 'robot_type': ''
}
"""

import socket
import json
import torch
import numpy as np
import threading
from server.local_code.image_sender import start_image_sender, image_queue, stop_image_sender
from queue import Queue

server_ip = '10.10.1.35'
image_port = 9100
text_port =9101

# 创建共享的非图像请求队列和响应队列
request_queue = Queue()
response_queue = Queue()

def start_non_image_thread():
    def send_non_image_data():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, text_port))
        print("[DataSender] Connected to server for non-image data")

        buffer = b""
        while True:
            payload = request_queue.get()
            if payload is None:
                break

            try:
                # 发送json数据
                json_str = json.dumps(payload)
                sock.sendall(json_str.encode('utf-8'))

                # 这里改为循环接收，直到完整收到一条JSON消息
                while True:
                    data = sock.recv(4096)
                    if not data:
                        # 连接关闭
                        raise ConnectionError("连接关闭")
                    buffer += data
                    try:
                        # 尝试解json
                        result = json.loads(buffer.decode('utf-8'))
                        buffer = b""  # 清空缓存
                        break  # 成功解析跳出循环
                    except json.JSONDecodeError:
                        # JSON未完整，继续收
                        continue

                response_queue.put(result)

            except Exception as e:
                print(f"[DataSender] Error: {e}")
                response_queue.put(None)
                break

        sock.close()


    threading.Thread(target=send_non_image_data, daemon=True).start()

# 启动图像线程
start_image_sender(host=server_ip, port=image_port)
# 启动非图像线程
start_non_image_thread()


def predict_from_server(observation):
    # 抽取图像数据并发送
    for cam_key in ['observation.images.side', 
                    # 'observation.images.side_depth', 
                    'observation.images.wrist']:
        img_tensor = observation.pop(cam_key)
        img_array = img_tensor.squeeze().cpu().numpy()
        img_array = np.transpose(img_array, (1, 2, 0))
        cam_name = cam_key.split('.')[-1]
        image_queue.put({"name": cam_name, "data": img_array})

    # 转换非图像数据为 JSON 兼容格式
    payload = {}
    for key, val in observation.items():
        if isinstance(val, torch.Tensor):
            payload[key] = val.cpu().numpy().tolist()
        else:
            payload[key] = val

    request_queue.put(payload)
    result = response_queue.get()
    # print("result=", result)  # 这里你已经打印了服务器返回的数据
    if result is None:
        raise RuntimeError("服务器返回无效数据")

    action_list = result["action"]
    action_tensor = torch.tensor(action_list, dtype=torch.float32, device='cpu')
    return action_tensor


def shutdown_clients():
    request_queue.put(None)
    stop_image_sender()
