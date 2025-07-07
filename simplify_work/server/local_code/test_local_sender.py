import torch
import time
from predict_from_server_api import predict_from_server,shutdown_clients

# 构造 observation
observation = {
    'observation.state': torch.rand(1, 6),
    # 'observation.force': torch.rand(1, 15),
    'observation.images.side': torch.rand(3, 480, 640),
    # 'observation.images.scene_depth': torch.rand(3, 480, 640),
    'observation.images.wrist': torch.rand(3, 480, 640),
    'task': 'pick apple',
    'robot_type': 'so100',
}

start_time = time.time()

# 预测调用
result = predict_from_server(observation)

# 程序退出前调用
shutdown_clients()

end_time = time.time()

print("Prediction result:", result)
print(f"Total round-trip time: {(end_time - start_time)*1000:.2f} ms")
