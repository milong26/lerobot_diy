import cv2
import numpy as np
import torch

import pyrealsense2 as rs

def _transform_image(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return image_np

def _decode_depth_image(depth_img):
    R = depth_img[:, :, 0]
    G = depth_img[:, :, 1]
    depth_raw = ((R.astype(np.uint16) << 8) | G.astype(np.uint16))
    return depth_raw

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def transform_camera_to_custom_coordsystem(points_3d):
    origin = np.array([ 0.24163092, -0.08227619,  0.60075652])
    x_axis = np.array([-0.36651895, -0.77909696,  0.50859786])
    y_axis = np.array([-0.92731940,  0.26136948, -0.26788937])
    z_axis = np.array([ 0.07577983, -0.56981920, -0.81826860])
    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    converted = []
    for p in points_3d:
        if p is None:
            converted.append(None)
        else:
            p = np.array(p)
            local_p = np.dot(R.T, p - origin)
            converted.append(tuple(local_p))
    return converted

def get_3d_point(x, y, depth_raw, intrinsics, depth_scale):
    depth_val = depth_raw[y, x] * depth_scale  # 单位: 米
    if depth_val == 0:
        return None
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_val)
    return point_3d

# count_of_number=0
def count_distance(color_tensor, depth_tensor):
    # 转换图像
    color_img = _transform_image(color_tensor)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    depth_img_raw = _transform_image(depth_tensor)
    # cv2.imwrite("debug_side_color.png", color_img)
    # cv2.imwrite("debug_side_depth_rgb.png", depth_img_raw)




    depth_raw = _decode_depth_image(depth_img_raw)

    # 初始化 RealSense 内参
    intrinsics = rs.intrinsics()
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = 304.7939453125
    intrinsics.ppy = 234.874755859375
    intrinsics.fx = 616.6113891601562
    intrinsics.fy = 616.5948486328125
    intrinsics.model = rs.distortion.inverse_brown_conrady
    intrinsics.coeffs = [0, 0, 0, 0, 0]
    depth_scale = 0.0010000000474974513

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    # 红色范围
    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 黄色范围
    yellow_mask = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([40, 255, 255]))

    cv2.imwrite("debug_hsv.png", hsv)
    cv2.imwrite("debug_red_mask.png", red_mask)
    cv2.imwrite("debug_yellow_mask.png", yellow_mask)

    # 膨胀消噪
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # 获取中心点
    def get_center(mask):
        contour = find_largest_contour(mask)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            return x + w // 2, y + h // 2
        return None

    red_center = get_center(red_mask)
    yellow_center = get_center(yellow_mask)

    if red_center is None or yellow_center is None:
        print("红色或黄色未检测到")
        return None

    # 深度转 3D 点
    red_point_cam = get_3d_point(*red_center, depth_raw, intrinsics, depth_scale)
    yellow_point_cam = get_3d_point(*yellow_center, depth_raw, intrinsics, depth_scale)

    if red_point_cam is None or yellow_point_cam is None:
        return None

    # 转换到自定义坐标系
    red_custom, yellow_custom = transform_camera_to_custom_coordsystem([red_point_cam, yellow_point_cam])

    if red_custom is None or yellow_custom is None:
        print("坐标转换失败")
        return None
    print("黄色",yellow_custom,"红色",red_custom)

    # 可视化保存识别图像
    vis_img = color_img.copy()
    cv2.rectangle(vis_img, (red_center[0]-10, red_center[1]-10), (red_center[0]+10, red_center[1]+10), (0, 0, 255), 2)
    cv2.rectangle(vis_img, (yellow_center[0]-10, yellow_center[1]-10), (yellow_center[0]+10, yellow_center[1]+10), (0, 255, 255), 2)
    # global count_of_number
    

    # 返回两点间的欧氏距离
    red_np = np.array(red_custom)
    yellow_np = np.array(yellow_custom)
    distance = np.linalg.norm(red_np - yellow_np)
    # cv2.imwrite(f"saveimages/get_red_and_yellow_{count_of_number}.png", vis_img)
    # count_of_number+=1
    print(f"Distance between red and yellow targets: {distance:.3f} m")
    return distance