import math
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import pyrealsense2 as rs
from ultralytics import YOLO  # 导入YOLO库

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
depth_scale = 0.0010000000474974513

class YOLOProcessor:
    def __init__(self, 
                 model_path="yolo/training/runs/train/yolo11nano_custom2/weights/best.pt",  # 使用微调后的YOLO模型
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = YOLO(model_path).to(self.device)  # 加载YOLO模型
        self.fail_counter = 0
        self.class_names = {0: "gripper"}  # 只需要夹子的类别定义
        self.conf_threshold = 0.25  # 置信度阈值
        
        # 淡黄色物体的HSV颜色范围
        # self.lower_yellow = np.array([20, 50, 100])
        # self.upper_yellow = np.array([35, 150, 255])
        self.lower_yellow = np.array([15, 30, 80])
        self.upper_yellow = np.array([45, 255, 255])
        self.min_contour_area = 100  # 最小轮廓面积
        self.total_images = 0
        self.gripper_detected = 0
        self.object_detected = 0


    def _transform_image(self, image_tensor):
        """将(0,1)范围的浮点张量转换为YOLO所需的预处理图像"""
        # 1. 转换到0-255范围并转换为uint8
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return image_np


    def opencv_detect_yellow_object(self, color_image):
        """使用OpenCV检测淡黄色物体"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 创建黄色物体的掩码
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 寻找面积最大的轮廓
        max_area = 0
        best_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area and area > max_area:
                max_area = area
                best_contour = cnt
        
        # 计算边界框和中心点
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y), (x, y, w, h)
        
        return None, None

    # 相当于main,处理图像,得到坐标,加入task
    @torch.no_grad()
    def process_sample(self, side_img: torch.Tensor, side_depth: torch.Tensor):
        # 1. 图像预处理
        image_tensor = self._transform_image(side_img)
        depth_tensor=self._transform_image(side_depth)
        
        # # 2. YOLO推理 - 只检测夹子
        gripper_center = None
        object_center = None
        threed_pos = [None, None]
        
        # 检测夹子
        results = self.model.predict(
            image_tensor, 
            imgsz=640,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # 3. 解析YOLO结果 - 只取夹子
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 只处理夹子类别
                    if int(box.cls.item()) == 0 and box.conf.item() >= self.conf_threshold:
                        gripper_box = box.xyxy[0].cpu().numpy()
                        gripper_center = self._get_bbox_center(gripper_box)
                        break  # 只取第一个检测到的夹子
        
        # # 4. 使用OpenCV检测淡黄色物体
        # # 注意：OpenCV需要BGR格式，但我们的图像是RGB，需要转换
        bgr_image = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        object_center, object_bbox = self.opencv_detect_yellow_object(bgr_image)
        
        # # 5. 获取3D坐标
        centers = []
        if gripper_center is not None:
            centers.append(gripper_center)
        if object_center is not None:
            centers.append(object_center)
        
        #         # 统计逻辑
        # self.total_images += 1
        # if gripper_center is not None:
        #     self.gripper_detected += 1
        # if object_center is not None:
        #     self.object_detected += 1

        
        # # 6. 转换为3D坐标
        if centers:
            threed_pos = self.pixel_to_3d(side_depth, centers)

        """
        测试center是否正确
        """

        # 仅调试用：可视化检测点并保存图像
        # if gripper_center is not None:
        #     cv2.circle(debug_vis, gripper_center, 8, (0, 255, 0), -1)  # green
        # if object_center is not None:
        #     cv2.circle(debug_vis, object_center, 8, (0, 0, 255), -1)  # red
        # debug_vis = depth_tensor.copy()
        # debug_save_path='outputs/depth'
        # cv2.imwrite(debug_save_path, depth_tensor)

        
        return threed_pos



    def transform_camera_to_custom_coordsystem(points_3d):
        """
        将一组相机坐标转换为自定义坐标系（以机械臂底座为原点，布面为XY平面）
        Args:
            points_3d: List of (x, y, z) in camera coordinate system
        Returns:
            List of transformed (x, y, z) in custom coordinate system
        """

        # 已知平均局部坐标系（三个单位向量 + 原点）
        # 这个坐标系要重新修改
        origin = np.array([0.03217833, -0.01095684, 0.07867188])

        x_axis = np.array([-0.24203161,  0.33476907, -0.91068676])
        y_axis = np.array([ 0.40107384,  0.27464791,  0.87390406])
        z_axis = np.array([ 0.90643808, -0.25679492, -0.33530044])

        # 构建旋转矩阵（列向量为各轴）
        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3,3)

        # 执行变换
        converted = []
        for p in points_3d:
            if p is None:
                converted.append(None)
            else:
                p = np.array(p)
                local_p = np.dot(R.T, p - origin)
                converted.append(tuple(local_p))

        return converted

    
    def add_depth_info_to_task(self, rgb_batch, depth_batch, task_batch):
        """添加深度信息到任务描述"""
        updated_tasks = []
        
        for rgb, depth, task in zip(rgb_batch, depth_batch, task_batch):
            # 获取两个目标的3D坐标(相机坐标系)
            points_3d = self.process_sample(rgb, depth)

        
            if points_3d and len(points_3d) >= 2:
                # 变成机械臂底座的坐标系,因为这个函数没有处理None,就放在这儿了
                converted_3d = self.transform_camera_to_custom_coordsystem(points_3d)
                a = converted_3d[0]  # gripper
                b = converted_3d[1]  # object
                
                # 如果坐标无效，使用占位符
                a_str = f"({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})" if a is not None else "(N/A, N/A, N/A)"
                b_str = f"({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f})" if b is not None else "(N/A, N/A, N/A)"
                
                task_str = f"{task} | gripper at {a_str}, the Pyramid-Shaped Sachet at {b_str}"
            else:
                task_str = f"{task} | insufficient valid 3D points"
            
            updated_tasks.append(task_str)
        
        return updated_tasks



    def remove_edge_of_r_cahnnel(b_encoded):
        """
        输入:一个通道
        输出:给通道去边缘
        测试的成功率:(只有这一个通道),比较的对象是原图(原图->视频编码->提取图,输入是提取图)
        [Bilateral + medianBlur + 局部跳变补偿]
        最大误差: 100
        平均误差: 0.51
        误差为0的像素占比: 84.48%
        误差大于10的像素占比: 0.80%
        """
        # Step 1: Bilateral 滤波
        b_filtered = cv2.bilateralFilter(b_encoded, d=5, sigmaColor=75, sigmaSpace=75)

        # Step 2: Laplacian 边缘增强
        laplacian = cv2.Laplacian(b_filtered, cv2.CV_64F, ksize=1)
        b_restored = b_encoded - laplacian
        b_restored = np.clip(b_restored, 0, 255).astype(np.uint8)

        # Step 3: medianBlur
        b_median = cv2.medianBlur(b_restored, 3)

        # Step 4: Sobel 局部跳变补偿
        sobel_x = cv2.Sobel(b_median, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(b_median, cv2.CV_16S, 0, 1, ksize=3)
        gradient = cv2.convertScaleAbs(cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32)))
        jump_mask = (gradient > 40)
        b_local_avg = cv2.blur(b_median, (3, 3))
        b_final = b_median.copy()
        b_final[jump_mask] = b_local_avg[jump_mask]
        return b_final
        

    # 无损返回深度值 根本不可能实现
    def decode_depth_from_rgb(self,rgb_image: np.ndarray) -> np.ndarray:
        # 此时的rgb_image是彩色深度图.用rg复原depth_uint16
        r = rgb_image[:, :, 0].astype(np.uint16)
        g = rgb_image[:, :, 1].astype(np.uint16)
        b = rgb_image[:, :, 2].astype(np.uint16)
        # 先给图片去边缘(大概把)
        b_final = self.remove_edge_of_r_cahnnel(r)
        depth_uint16 = ((b_final.astype(np.uint16) << 8) | g.astype(np.uint16))
        return depth_uint16

    def pixel_to_3d(self, depth_image, pixels, radius=5):
        """将像素坐标转换为3D坐标，支持以像素为中心、半径为radius的平均深度"""
        
        # 将彩色深度图转换为原始深度值
        oned_depth_image = self.decode_depth_from_rgb(depth_image)
        
        height, width = oned_depth_image.shape
        points_3d = []
        
        for (u, v) in pixels:
            # 检查中心点是否在图像范围内
            if v < 0 or v >= height or u < 0 or u >= width:
                self.fail_counter += 1
                print(f"{self.fail_counter}: bbox超出范围")
                points_3d.append(None)
                continue

            # 获取局部区域的深度值
            u_min = max(0, u - radius)
            u_max = min(width - 1, u + radius)
            v_min = max(0, v - radius)
            v_max = min(height - 1, v + radius)

            depth_patch = oned_depth_image[v_min:v_max+1, u_min:u_max+1]
            valid_depths = depth_patch[depth_patch > 0]

            if valid_depths.size == 0:
                points_3d.append(None)
                continue

            # 计算平均深度
            depth_raw = valid_depths.mean()
            depth_in_meters = depth_raw * depth_scale

            # 转换为3D坐标
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_in_meters)
            points_3d.append(tuple(point))  # (x, y, z)

        return points_3d

    def count_distance(self, rgb_batch, depth_batch):
        """计算两个目标之间的距离"""
        distances = []
        for rgb, depth in zip(rgb_batch, depth_batch):
            points_3d = self.process_sample(rgb, depth)
            
            if points_3d and len(points_3d) >= 2 and all(p is not None for p in points_3d[:2]):
                a = points_3d[0]
                b = points_3d[1]
                distance = math.sqrt((a[0] - b[0])**2 + 
                                     (a[1] - b[1])**2 + 
                                     (a[2] - b[2])**2)
                distances.append(distance)
            else:
                distances.append(None)
        return distances

    def print_statistics(self):
        """打印识别成功率统计"""
        if self.total_images == 0:
            print("尚未处理任何图像。")
            return

        gripper_rate = self.gripper_detected / self.total_images * 100
        object_rate = self.object_detected / self.total_images * 100

        print(f"总图像数: {self.total_images}")
        print(f"Gripper 检测成功率: {gripper_rate:.2f}% ({self.gripper_detected}/{self.total_images})")
        print(f"Object 检测成功率: {object_rate:.2f}% ({self.object_detected}/{self.total_images})")
