# server/handler.py
import socket
import threading
import json
import torch
import numpy as np
from image_receiver import image_buffer, lock
import time

def dummy_policy(observation):
    # 模拟返回动作：你可以替换为实际模型调用
    return {"action": [0.1, 0.2, 0.3]}


def wait_for_images(required_keys, timeout=2.0):
    """等待图像都到达缓存，最多 timeout 秒"""
    start = time.time()
    while time.time() - start < timeout:
        with lock:
            if all(k in image_buffer for k in required_keys):
                return {k: image_buffer[k] for k in required_keys}
        time.sleep(0.01)
    return None  # 超时未到齐

def handle_data_connection(conn):
    while True:
        try:
            data = conn.recv(8192)
            if not data:
                break

            payload = json.loads(data.decode('utf-8'))

            # 重建 observation（非图像部分）
            observation = {}
            for key, val in payload.items():
                if key.startswith('observation.'):
                    observation[key] = torch.tensor(val, dtype=torch.float32)
                else:
                    observation[key] = val

            # 等待图像数据到齐
            images = wait_for_images(['scene', 'scene_depth', 'wrist'], timeout=2.0)
            if images is None:
                print("[DataReceiver] Warning: Missing some images after waiting.")
                # 你可以决定继续执行或返回默认图像
                # 例如填充默认图像：
                for cam in ['scene', 'scene_depth', 'wrist']:
                    key = f'observation.images.{cam}'
                    observation[key] = torch.zeros(3, 480, 640)  # 根据你实际图像尺寸设置
            else:
                # 合并图像数据进 observation
                for cam, np_img in images.items():
                    key = f'observation.images.{cam}'
                    observation[key] = torch.tensor(np_img, dtype=torch.float32)

            # 调用模型
            action = dummy_policy(observation)
            conn.sendall(json.dumps(action).encode('utf-8'))

        except Exception as e:
            print(f"[DataReceiver] Error: {e}")
            break

    conn.close()


def start_data_server(host='0.0.0.0', port=9001):
    threading.Thread(target=_start_data_thread, args=(host, port), daemon=True).start()

def _start_data_thread(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((host, port))
    server_sock.listen(1)
    print(f"[DataReceiver] Listening on {host}:{port}")

    conn, addr = server_sock.accept()
    print(f"[DataReceiver] Connected by {addr}")
    handle_data_connection(conn)
