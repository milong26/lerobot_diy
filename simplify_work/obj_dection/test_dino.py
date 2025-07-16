# 为了方便导入groundingdino
import sys
import os

# 添加GroundingDINO模块的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
groundingdino_path = os.path.join(current_dir, "GroundingDINO")
sys.path.append(groundingdino_path)
# 以后再重构代码

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
CONFIG_PATH = "simplify_work/obj_dection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

CHECKPOINT_PATH = "models/objdection/dinoground/groundingdino_swint_ogc.pth"   #下载的权重文件
DEVICE = "cpu"   #可以选择cpu/cuda
IMAGE_PATH = "simplify_work/obj_dection/refer.png"    #用户设置的需要读取image的路径
TEXT_PROMPT = "The Gripper And The Pyramid-Shaped Sachet"    #用户给出的文本提示
BOX_TRESHOLD = 0.25     #源码给定的边界框判定阈值
TEXT_TRESHOLD = 0.25    #源码给定的文本端获取关键属性阈值
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
print(boxes,phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)