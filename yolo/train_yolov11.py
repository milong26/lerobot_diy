# train_yolo11nano.py

from ultralytics import YOLO

# 加载YOLOv11-nano模型（确保 ultralytics 包支持该模型）
model = YOLO("yolo11n.pt")

# 开始训练
model.train(
    data="yoloimages/data.yaml",  # 数据集配置文件路径
    epochs=100,                   # 训练轮次
    imgsz=640,                    # 输入图像大小
    batch=16,                     # 批大小
    name="yolo11nano_custom",     # 实验名称/输出文件夹
    project="runs/train",         # 训练输出保存路径
    device=0                      # 使用GPU（0）或CPU（'cpu'）
)
