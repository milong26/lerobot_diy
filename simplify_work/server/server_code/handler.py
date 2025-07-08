# server/handler.py
from copy import copy
import socket
import threading
import json
from model_loader import load_policy
import torch
import numpy as np
from image_receiver import image_buffer, lock
import time


def wait_for_images(required_keys, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        with lock:
            if all(k in image_buffer for k in required_keys):
                return {k: image_buffer[k] for k in required_keys}
        time.sleep(0.01)
    return None


def preprocess_observation_server(observation, device="cuda"):
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
        
    return observation



def handle_data_connection(conn, model):
    while True:
        try:
            data = conn.recv(8192)
            if not data:
                break

            payload = json.loads(data.decode('utf-8'))
            print("payload内容",payload)

            observation = {}
            for key, val in payload.items():
                # if key.startswith('observation.'):
                #     observation[key] = torch.tensor(val, dtype=torch.float32)
                # else:
                if isinstance(val, list):
                    val = np.array(val, dtype=np.float32).squeeze()
                observation[key] = val
            # 每个episode才reset一次，但是reset只是跟queue有关
            # model.reset() 
            images = wait_for_images(['side', 'wrist'], timeout=2.0)
            if images is None:
                raise KeyError("图片没传送过来")
            else:
                for cam, np_img in images.items():
                    key = f'observation.images.{cam}'
                    # np_img = observation[key]
                    if np_img.shape[-1] == 3:  # 如果是 HWC
                        observation[key] = np.transpose(np_img, (2, 0, 1))  # → CHW
                    observation[key] = np_img
            observation["observation.images"] = [
                observation["observation.images.side"],
                observation["observation.images.wrist"],
                # batch["observation.images.sideDepth"],
            ]
            # 需要确保这里的observation格式一致
            batch=preprocess_observation_server(observation=observation)
            # 整理 batch 格式供模型推理
            # 参考_get_action_chunk函数
            images, img_masks = model.prepare_images(batch)
            state = model.prepare_state(batch)
            lang_tokens, lang_masks = model.prepare_language(batch)
            # 看了下好像没有noise？
            noise=None

            actions = model.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            print("推理后",time.time())
            print(actions)

            # 返回结果
            conn.sendall(json.dumps({"action": actions.tolist()}).encode('utf-8'))

        except Exception as e:
            print(f"[DataReceiver] Error: {e}")
            break

    conn.close()


def start_data_server(host='0.0.0.0', port=9101, model=None):
    threading.Thread(target=_start_data_thread, args=(host, port, model), daemon=True).start()


def _start_data_thread(host, port, model):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    print(f"[DataReceiver] Listening on {host}:{port}")

    while True:
        conn, addr = server_sock.accept()
        print(f"[DataReceiver] Connected by {addr}")
        threading.Thread(target=handle_data_connection, args=(conn, model), daemon=True).start()

