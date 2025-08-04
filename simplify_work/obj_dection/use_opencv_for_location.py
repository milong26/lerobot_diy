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
    def __init__(self):
        self.fail_counter = 0

        # 红色识别 (夹子)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # 黄色识别 (物体)
        self.lower_yellow = np.array([15, 30, 80])
        self.upper_yellow = np.array([45, 255, 255])
        self.min_contour_area = 100

        self.total_images = 0
        self.gripper_detected = 0
        self.object_detected = 0

    def _transform_image(self, image_tensor):
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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

    def opencv_detect_yellow_object(self, color_image):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

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

    @torch.no_grad()
    def process_sample(self, side_img: torch.Tensor, side_depth: torch.Tensor):
        image_tensor = self._transform_image(side_img)
        depth_tensor = self._transform_image(side_depth)
        bgr_image = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)

        gripper_center, _ = self.opencv_detect_red_gripper(bgr_image)
        object_center, _ = self.opencv_detect_yellow_object(bgr_image)

        self.total_images += 1
        if gripper_center is not None:
            self.gripper_detected += 1
        if object_center is not None:
            self.object_detected += 1

        centers = [pt for pt in [gripper_center, object_center] if pt is not None]
        threed_pos = self.pixel_to_3d(depth_tensor, centers) if centers else [None, None]
        return threed_pos

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

    def add_depth_info_to_task(self, rgb_batch, depth_batch, task_batch):
        rgb_batch = rgb_batch.to("cuda" if torch.cuda.is_available() else "cpu")
        depth_batch = depth_batch.to("cuda" if torch.cuda.is_available() else "cpu")
        updated_tasks = []

        for rgb, depth, task in zip(rgb_batch, depth_batch, task_batch):
            if "|" in task or "gripper at" in task or "Pyramid-Shaped Sachet at" in task:
                print("处理过了,pass")
                updated_tasks.append(task)
                continue

            points_3d = self.process_sample(rgb, depth)
            if points_3d and len(points_3d) >= 2:
                converted_3d = self.transform_camera_to_custom_coordsystem(points_3d)
                a, b = converted_3d[0], converted_3d[1]
                a_str = f"({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})" if a else "(N/A, N/A, N/A)"
                b_str = f"({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f})" if b else "(N/A, N/A, N/A)"
                task_str = f"{task} | gripper at {a_str}m, the Pyramid-Shaped Sachet at {b_str}m"
            else:
                task_str = f"{task} |"
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
        is_single = isinstance(rgb_batch, torch.Tensor) and rgb_batch.dim() == 3

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
                distances.append(None)

        return distances[0] if is_single else distances

    def print_statistics(self):
        if self.total_images == 0:
            print("尚未处理任何图像。")
            return

        gripper_rate = self.gripper_detected / self.total_images * 100
        object_rate = self.object_detected / self.total_images * 100

        print(f"总图像数: {self.total_images}")
        print(f"Gripper 检测成功率: {gripper_rate:.2f}% ({self.gripper_detected}/{self.total_images})")
        print(f"Object 检测成功率: {object_rate:.2f}% ({self.object_detected}/{self.total_images})")
