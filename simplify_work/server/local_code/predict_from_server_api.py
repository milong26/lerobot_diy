# control_utils里面调用，代替原来的action = policy.select_action(observation)
# 传入policy,observation
"""
policy暂时传不过来，以后再搞，最好是能把path和type传过来

{'observation.state': tensor([[ ]]),
 'observation.force': tensor([[]]), 
'observation.images.scene': tensor([[[[]]]]),
'observation.images.scene_depth': tensor([[[[]]]]), 
'observation.images.wrist': tensor([[[[]]]]),
 'task': '', 
 'robot_type': ''
}
"""
import asyncio
import json
import torch
import numpy as np
from ..shared_state import image_queue

server_ip = '10.10.1.35'
image_port = 9100
text_port = 9101

async def predict_from_server(observation):
    reader, writer = await asyncio.open_connection(server_ip, text_port)

    # 图像提取与发送
    imgs = {
        'scene': observation.pop('observation.images.side').cpu().numpy(),
        'scene_depth': observation.pop('observation.images.side_depth').cpu().numpy(),
        'wrist': observation.pop('observation.images.wrist').cpu().numpy()
    }
    await image_queue.put(imgs)

    # 非图像部分序列化
    payload = {
        k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
        for k, v in observation.items()
    }
    writer.write(json.dumps(payload).encode('utf-8'))
    await writer.drain()

    # 接收结果
    resp = await reader.read(4096)
    result = json.loads(resp.decode('utf-8'))
    writer.close()
    await writer.wait_closed()
    return result

