import torch
import asyncio
from predict_from_server_api import predict_from_server
from image_sender import start_image_sender_async, stop_image_sender_async

async def main():
    observation = {
        'observation.state': torch.rand(1, 6),
        'observation.force': torch.rand(1, 15),
        'observation.images.scene': torch.rand(3, 480, 640),
        'observation.images.scene_depth': torch.rand(3, 480, 640),
        'observation.images.wrist': torch.rand(3, 480, 640),
        'task': 'pick apple',
        'robot_type': 'so100',
    }

    start_image_sender_async('10.10.1.35', 9100)
    result = await predict_from_server(observation)
    stop_image_sender_async()
    print("Result:", result)

if __name__ == '__main__':
    asyncio.run(main())