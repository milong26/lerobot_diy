import os
import sys
import cv2

# 添加 GroundingDINO 到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
groundingdino_path = os.path.join(current_dir, "GroundingDINO")
sys.path.append(groundingdino_path)

import pyrealsense2 as rs
import numpy as np



# 手动构造 intrinsics 对象
intrinsics = rs.intrinsics()
intrinsics.width = 640
intrinsics.height = 480
intrinsics.ppx = 304.7939453125
intrinsics.ppy = 234.874755859375
intrinsics.fx = 616.6113891601562
intrinsics.fy = 616.5948486328125
intrinsics.model = rs.distortion.inverse_brown_conrady
intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
depth_scale=0.0010000000474974513




# input image,boxed,phrases,depth_image



# obj_detection/api.py
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

class GroundingDINOProcessor:
    def __init__(self, 
                 config_path="simplify_work/obj_dection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                 checkpoint_path="models/objdection/dinoground/groundingdino_swint_ogc.pth", 
                 text_prompt="The Gripper And The Pyramid-Shaped Sachet", 
                 device="cpu"):
        self.device = device
        self.text_prompt = text_prompt
        self.model = load_model(config_path, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.fail_counter=0
    

    def _transform_image(self, image_np):
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        image_tensor, _ = self.transform(image_pil, None)
        return image_tensor

    def _get_bbox_centers(self, boxes_cxcywh, image_w, image_h):
        centers = []
        for box in boxes_cxcywh:
            cx, cy, _, _ = box.tolist()
            x_center = cx * image_w
            y_center = cy * image_h
            centers.append((int(round(x_center)), int(round(y_center))))
        return centers



    # 此时的side_img是tensor
    @torch.no_grad()
    def process_sample(self, side_img: torch.Tensor, side_depth: torch.Tensor):
        """
        Args:
            side_img: Tensor (3, H, W) RGB image
            side_depth: Tensor (1, H, W) depth map

        Returns:
            depth_values: List[float] sampled at bbox centers
        """
        # Convert to numpy for predict
        image_np = (side_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image_tensor = self._transform_image(image_np)

        boxes, logits,phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=0.25,
            text_threshold=0.25,
            device=self.device,
        )

        # 1. 把相同类别的索引分组
        from collections import defaultdict

        grouped = defaultdict(list)
        for i, phrase in enumerate(phrases):
            grouped[phrase].append(i)

        # 2. 对每组按 logits 选最高的索引
        selected_indices = []
        for phrase, indices in grouped.items():
            # 取这组的logits
            group_logits = logits[indices]
            max_idx_in_group = indices[group_logits.argmax()]
            selected_indices.append(max_idx_in_group)

        # 3. 根据 selected_indices 过滤结果
        filtered_logits = logits[selected_indices]
        filtered_phrases = [phrases[i] for i in selected_indices]
        filtered_boxes = boxes[selected_indices]
        if len(filtered_boxes) < 2:
            self.fail_counter+=1
            print(self.fail_counter,":没有识别到2个物体",filtered_boxes,filtered_phrases)
            return None  # fallback if detection fails
        image_h, image_w = image_np.shape[:2]
        centers = self._get_bbox_centers(filtered_boxes, image_w, image_h)
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
        # # 使用转换后的图像生成标注帧
        # annotated_frame = annotate(
        #     image_source=image_bgr,  # 使用转换后的BGR图像
        #     boxes=filtered_boxes,
        #     logits=filtered_logits,
        #     phrases=filtered_phrases
        # )
        # # 保存标注后的图像
        # cv2.imwrite("annotated_image.jpg", annotated_frame)

        # 按照gripper、object的顺序
        if "gripper" not in filtered_phrases[0]:
            centers[0], centers[1] = centers[1], centers[0]


        threed_pos=self.pixel_to_3d(side_depth,centers)
        # depth_tasks = self._sample_depth(side_depth, centers,)
        return threed_pos
    
    
    def add_depth_info_to_task(self, rgb_batch, depth_batch, task_batch):

        updated_tasks = []

        for rgb, depth, task in zip(rgb_batch, depth_batch, task_batch):
            # 获取两个目标的深度值
            points_3d = self.process_sample(rgb, depth)  # 返回两个 float

            # 用深度返回物体的3维坐标

            valid_points = [p for p in points_3d if p is not None]

            if len(valid_points) >= 2:
                a = valid_points[0]
                b = valid_points[1]
                task_str = f"{task} | gripper at ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}), the Pyramid-Shaped Sachet at ({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f})"
            else:
                task_str = f"{task} | insufficient valid 3D points"

            updated_tasks.append(task_str)

        return updated_tasks
    
    def revert_colorful_depth_image(self,colored_depth_image):
        # 输入是3通道的彩色图（伪彩色深度图）

        # 先转为灰度图
        gray_depth = cv2.cvtColor(colored_depth_image, cv2.COLOR_BGR2GRAY)  # shape: (H, W)

        # 反向缩放恢复接近原始深度值（注意不精确）
        depth_approx = gray_depth.astype(np.float32) / 0.03  # inverse of alpha
        return depth_approx

    def  pixel_to_3d(self, depth_image, pixels):
        # numpy只能在cpu上面
        if isinstance(depth_image, torch.Tensor):
            depth_image = depth_image.cpu().numpy()
            depth_image = depth_image.transpose(1, 2, 0)
        oned_depth_image=self.revert_colorful_depth_image(depth_image)
        points_3d = []

        # depth_image要从3 channel变成1 channel
        # 检验depth的size


        for (u, v) in pixels:
            if u < 0 or v < 0 or v >= depth_image.shape[0] or u >= depth_image.shape[1]:
                self.fail_counter+=1
                print(self.fail_counter,":bbox超出范围")
                points_3d.append(None)
                continue 

            depth_raw = oned_depth_image[u, v]

            if depth_raw == 0:
                points_3d.append(None)
                continue

            depth_in_meters = depth_raw * depth_scale
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_in_meters)
            points_3d.append(tuple(point))  # (x, y, z)
            

        return points_3d


