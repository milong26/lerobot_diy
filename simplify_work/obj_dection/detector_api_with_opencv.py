import math
import cv2
import torch
import numpy as np
import pyrealsense2 as rs

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


class VisionProcessor:
    def __init__(self,language_tip_mode=""):
        self.fail_counter = 0

        # 红色识别 (夹子)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # # 黄色识别 (物体)
        # self.lower_yellow = np.array([15, 30, 80])
        # self.upper_yellow = np.array([45, 255, 255])
        self.min_contour_area = 50
        self.gripper_max_area = 200  # 根据实际 gripper 尺寸调节

        self.total_images = 0
        self.gripper_detected = 0
        self.object_detected = 0

        # task处理的mode
        self.language_tip_mode=language_tip_mode
        # 更改代码，适应更多颜色，红色不能用，蓝/黑尽量别用
        # 用颜色提取器获取，需要适应颜色变换
        self.color_ranges = {
            # Gripper: 深色，饱和度高，Hue 在 92–97
            # "gripper": [(np.array([90, 60, 50]), np.array([100, 130, 150]))],

            # Sponge: 蓝绿色，Hue 22–24，S 很高
            "sponge": [(np.array([20, 120, 100]), np.array([26, 255, 255]))],

            # Sachet: 灰蓝色，Hue 15–28，和 sponge 还是能区分开
            "sachet": [(np.array([15, 50, 60]), np.array([28, 130, 170]))],

            # Accessory: 浅灰绿，Hue 35–49
            "accessory": [(np.array([35, 15, 100]), np.array([50, 60, 230]))],

            # Router: 浅灰白，Hue 100–104，S 和 V 都比较高
            "router": [(np.array([98, 45, 200]), np.array([106, 70, 230]))],

            # Sticker: 蓝灰色，Hue 21–35，低饱和，高亮度
            "sticker": [(np.array([20, 20, 190]), np.array([36, 60, 230]))],
        }


    # 颜色检测函数，输入彩色图片和颜色名字，输出在彩色图片上的位置
    def opencv_detect_color(self, color_image, color_name):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        masks = []
        if color_name not in self.color_ranges:
            return None, None
        for lower, upper in self.color_ranges[color_name]:
            masks.append(cv2.inRange(hsv, lower, upper))
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area and area > max_area:
                max_area = area
                best_contour = cnt

        if best_contour is not None:
            bbox = cv2.boundingRect(best_contour)
            return self._get_bbox_center(bbox), bbox
        return None, None

    def _transform_image(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
        # [B, C, H, W] -> [C, H, W]
            image_tensor = image_tensor[0]
        if isinstance(image_tensor, torch.Tensor):
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(image_tensor, np.ndarray):
            if image_tensor.ndim == 3:
                # 如果是 [H, W, C]，直接用
                if image_tensor.shape[2] == 3:
                    image_np = image_tensor.astype(np.uint8)
                # 如果是 [C, H, W]，先转换
                elif image_tensor.shape[0] == 3:
                    image_np = np.transpose(image_tensor, (1, 2, 0)).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported numpy image shape: {image_tensor.shape}")
            elif image_tensor.ndim == 4:
                # [1, C, H, W] 或 [1, H, W, C]
                image_tensor = image_tensor[0]
                return self._transform_image(image_tensor)
            else:
                raise ValueError(f"Unsupported numpy image dimensions: {image_tensor.ndim}")

        return image_np

    def _get_bbox_center(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def opencv_detect_red_gripper(self, color_image):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area and area > max_area:
                max_area = area
                best_contour = cnt

        if best_contour is not None:
            bbox = cv2.boundingRect(best_contour)
            return self._get_bbox_center(bbox), bbox
        return None, None

    # def opencv_detect_yellow_object(self, color_image):
    #     hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     max_area = 0
    #     best_contour = None
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area > self.min_contour_area and area > max_area:
    #             max_area = area
    #             best_contour = cnt

    #     if best_contour is not None:
    #         bbox = cv2.boundingRect(best_contour)
    #         return self._get_bbox_center(bbox), bbox
    #     return None, None
    
    # 集中处理函数
    # 默认返回的只有3d的了
    @torch.no_grad()
    def process_sample(self, side_img, side_depth, colors_to_detect=None):
        if colors_to_detect is None:
            colors_to_detect = ["sponge"]  # 默认只检测 yellow
        image_tensor = self._transform_image(side_img)
        depth_tensor = self._transform_image(side_depth)
        bgr_image = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)

        results_2d = {}
        # gripper 单独用红色检测
        # gripper_center, _ = self.opencv_detect_color(bgr_image, "gripper")
        gripper_center, _ = self.opencv_detect_red_gripper(bgr_image)
        print("gripper的位置",gripper_center)
        results_2d["gripper"] = gripper_center

        # 其他颜色
        for c in colors_to_detect:
            center, _ = self.opencv_detect_color(bgr_image, c)
            results_2d[c] = center

        # 统计
        self.total_images += 1
        if results_2d["gripper"] is not None:
            self.gripper_detected += 1
        for c in colors_to_detect:
            if results_2d[c] is not None:
                self.object_detected += 1

        # 转3D
        centers = {k: v for k, v in results_2d.items() if v is not None}
        threed_points = {}
        if centers:
            points_3d = self.pixel_to_3d(depth_tensor, list(centers.values()))
            for key, pt in zip(centers.keys(), points_3d):
                threed_points[key] = pt



        # 测试识别对不对
        for name, center in results_2d.items():
            if center is None:
                continue
            cx, cy = center
            cv2.circle(bgr_image, (cx, cy), 10, (0, 255, 0), 2)  # 半径10圈
            cv2.putText(bgr_image, name, (cx + 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保存图片
            cv2.imwrite("debug_first_detection.jpg", bgr_image)

            # 打印3D坐标
            print("=== 识别到的3D坐标 ===")
            for k, v in threed_points.items():
                print(k, ":", v)

            # 强制退出
            # raise KeyError("Debug exit after saving first detection image.")
        return threed_points

    def transform_camera_to_custom_coordsystem(self, points_3d):
        origin = np.array([0.24163092, -0.08227619, 0.60075652])
        x_axis = np.array([-0.36651895, -0.77909696, 0.50859786])
        y_axis = np.array([-0.92731940, 0.26136948, -0.26788937])
        z_axis = np.array([0.07577983, -0.56981920, -0.81826860])
        R = np.stack([x_axis, y_axis, z_axis], axis=1)

        converted = []
        for p in points_3d:
            if p is None:
                converted.append(None)
            else:
                local_p = np.dot(R.T, np.array(p) - origin)
                converted.append(tuple(local_p))
        return converted

    def get_average_sevenpoints_3d_coords(self, depth_batch):
        sevenpoints = [
            (557, 149), (633, 261), (612, 236),
            (594, 212), (614, 326), (576, 340), (538, 350)
        ]
        collected_points = [[] for _ in sevenpoints]

        for depth_img in depth_batch:
            depth_tensor = self._transform_image(depth_img)
            points_3d = self.pixel_to_3d(depth_tensor, sevenpoints)
            for i, p in enumerate(points_3d):
                if p is not None:
                    collected_points[i].append(np.array(p))

        avg_points = []
        for pts in collected_points:
            avg_points.append(tuple(np.mean(pts, axis=0)) if pts else None)
        return avg_points

    def add_depth_info_to_task(self, rgb_batch, depth_batch, task_batch, colors_to_detect=None):
        updated_tasks = []
        for rgb, depth, task in zip(rgb_batch, depth_batch, task_batch):
            # 返回 dict, e.g. {"gripper": (x,y,z), "sachet": (..,..,..), ...}
            points_3d_dict = self.process_sample(rgb, depth, colors_to_detect)

            # 转换到自定义坐标系
            valid_pts = [pt for pt in points_3d_dict.values() if pt is not None]
            converted_pts = self.transform_camera_to_custom_coordsystem(valid_pts)

            # 映射回 key
            mapped = {}
            i = 0
            for k, v in points_3d_dict.items():
                if v is not None:
                    mapped[k] = converted_pts[i]
                    i += 1
                else:
                    mapped[k] = None
            print("对应",mapped)
            raise KeyError("Debug exit after saving first detection image.")

            task_str = task  # 默认原始

            if self.language_tip_mode:
                mode = self.language_tip_mode.lower()

                # === training 模式：输出所有目标的绝对位置 ===
                if mode == "training":
                    parts = [task]
                    for name, pos in mapped.items():
                        if pos is not None:
                            parts.append(f"{name} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    task_str = " | ".join(parts)

                # === relative / grid 模式：只拼接相对位置 ===
                else:
                    gripper_pos = mapped.get("gripper")
                    if gripper_pos is None:
                        # 没有 gripper，保持 task 不变
                        task_str = task
                    else:
                        rel_parts = []
                        for obj_name, obj_pos in mapped.items():
                            if obj_name == "gripper" or obj_pos is None:
                                continue
                            dx = obj_pos[0] - gripper_pos[0]
                            dy = obj_pos[1] - gripper_pos[1]
                            dz = obj_pos[2] - gripper_pos[2]

                            # 选择单位
                            if "5cm" in mode:
                                dx, dy, dz = round(dx / 0.05), round(dy / 0.05), round(dz / 0.05)
                                rel_parts.append(f"{obj_name} position relative to gripper is ({dx}, {dy}, {dz}) in 5cm grid units")
                            elif "2cm" in mode:
                                dx, dy, dz = round(dx / 0.02), round(dy / 0.02), round(dz / 0.02)
                                rel_parts.append(f"{obj_name} position relative to gripper is ({dx}, {dy}, {dz}) in 2cm grid units")
                            else:  # relative 模式
                                rel_parts.append(f"{obj_name} position relative to gripper is ({dx:.3f}, {dy:.3f}, {dz:.3f})")

                        if rel_parts:
                            task_str = f"{task}, " + ", ".join(rel_parts)

            updated_tasks.append(task_str)
            

        return updated_tasks






    def decode_depth_from_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        r = rgb_image[:, :, 0].astype(np.uint8)
        g = rgb_image[:, :, 1].astype(np.uint8)
        depth_uint16 = ((r.astype(np.uint16) << 8) | g.astype(np.uint16))
        return depth_uint16

    def pixel_to_3d(self, depth_image, pixels, radius=5):
        oned_depth_image = self.decode_depth_from_rgb(depth_image)
        height, width = oned_depth_image.shape
        points_3d = []

        for (u, v) in pixels:
            if v < 0 or v >= height or u < 0 or u >= width:
                self.fail_counter += 1
                print(f"{self.fail_counter}: bbox超出范围")
                points_3d.append(None)
                continue

            u_min = max(0, u - radius)
            u_max = min(width - 1, u + radius)
            v_min = max(0, v - radius)
            v_max = min(height - 1, v + radius)

            depth_patch = oned_depth_image[v_min:v_max + 1, u_min:u_max + 1]
            valid_depths = depth_patch[depth_patch > 0]

            if valid_depths.size == 0:
                points_3d.append(None)
                continue

            depth_raw = valid_depths.mean()
            depth_in_meters = depth_raw * depth_scale
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_in_meters)
            points_3d.append(tuple(point))

        return points_3d

    def count_distance(self, rgb_batch, depth_batch):
        is_single = (
            (isinstance(rgb_batch, torch.Tensor) and rgb_batch.dim() == 3)
            or
            (isinstance(rgb_batch, np.ndarray) and rgb_batch.ndim == 3)
        )


        if is_single:
            rgb_batch = [rgb_batch]
            depth_batch = [depth_batch]

        distances = []
        for rgb, depth in zip(rgb_batch, depth_batch):
            points_3d = self.process_sample(rgb, depth)
            if points_3d and len(points_3d) >= 2 and all(p is not None for p in points_3d[:2]):
                a, b = points_3d[0], points_3d[1]
                distance = math.sqrt((a[0] - b[0])**2 + 
                                     (a[1] - b[1])**2 + 
                                     (a[2] - b[2])**2)
                distances.append(distance)
            else:
                distances.append(0)

        return distances[0] if is_single else distances

    def print_statistics(self):
        if self.total_images == 0:
            print("尚未处理任何图像。")
            return
        if self.total_images % 1000 == 0:
            gripper_rate = self.gripper_detected / self.total_images * 100
            object_rate = self.object_detected / self.total_images * 100
            print(f"总图像数: {self.total_images}")
            print(f"Gripper 检测成功率: {gripper_rate:.2f}% ({self.gripper_detected}/{self.total_images})")
            print(f"Object 检测成功率: {object_rate:.2f}% ({self.object_detected}/{self.total_images})")
