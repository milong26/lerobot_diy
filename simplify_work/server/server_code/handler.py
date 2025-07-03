# server/handler.py
import socket
import threading
import json
import torch
import numpy as np
from image_receiver import image_buffer, lock

def dummy_policy(observation):
    # 模拟返回动作：你可以替换为实际模型调用
    return {"action": [0.1, 0.2, 0.3]}

def handle_data_connection(conn):
    while True:
        try:
            data = conn.recv(8192)
            if not data:
                break

            payload = json.loads(data.decode('utf-8'))

            # 重建 observation
            observation = {}
            for key, val in payload.items():
                if key.startswith('observation.'):
                    observation[key] = torch.tensor(val, dtype=torch.float32)
                else:
                    observation[key] = val

            # 合并图像数据
            with lock:
                for cam in ['scene', 'scene_depth', 'wrist']:
                    key = f'observation.images.{cam}'
                    if cam in image_buffer:
                        np_img = image_buffer[cam]
                        tensor_img = torch.tensor(np_img, dtype=torch.float32)
                        observation[key] = tensor_img
                    else:
                        print(f"[DataReceiver] Warning: Missing image '{cam}'")

            # 调用模型
            action = dummy_policy(observation)
            conn.sendall(json.dumps(action).encode('utf-8'))

        except Exception as e:
            print(f"[DataReceiver] Error: {e}")
            break

    conn.close()

def start_data_server(host='0.0.0.0', port=9000):
    threading.Thread(target=_start_data_thread, args=(host, port), daemon=True).start()

def _start_data_thread(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((host, port))
    server_sock.listen(1)
    print(f"[DataReceiver] Listening on {host}:{port}")

    conn, addr = server_sock.accept()
    print(f"[DataReceiver] Connected by {addr}")
    handle_data_connection(conn)
