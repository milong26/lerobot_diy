"""
lerobot_dit$ python simplify_work/opencv_detect_object.py
打开三个窗口，标注黄色和红色物体并输出中心点相对机械臂坐标的3d位置
"""
import pyrealsense2 as rs
import cv2
import numpy as np

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 设置内参
intrinsics = rs.intrinsics()
intrinsics.width = 640
intrinsics.height = 480
intrinsics.ppx = 304.7939453125
intrinsics.ppy = 234.874755859375
intrinsics.fx = 616.6113891601562
intrinsics.fy = 616.5948486328125
intrinsics.model = rs.distortion.inverse_brown_conrady
intrinsics.coeffs = [0, 0, 0, 0, 0]

depth_scale = 0.0010000000474974513  # 深度单位转换比例

# 启动相机
pipeline.start(config)

# 点击查看 HSV
def print_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        print(f"HSV at ({x},{y}) = {pixel}")

cv2.namedWindow("Color Frame")
cv2.setMouseCallback("Color Frame", print_hsv_on_click)

def get_3d_point(x, y, depth_frame):
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        return None
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point_3d

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

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 红色 HSV 范围（两段，适配暗红/小区域）
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # 淡黄色 HSV 范围（宽松）
        lower_yellow = np.array([15, 30, 30])   # 适配浅黄
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 可选：去噪
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        centers_3d = []
        color_names = []
        box_colors = []

        for mask, color_name, box_color in [(red_mask, "Red", (0, 0, 255)), (yellow_mask, "Yellow", (0, 255, 255))]:
            contour = find_largest_contour(mask)
            if contour is not None and cv2.contourArea(contour) > 100:  # 原为500，支持小红色目标
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

                point_3d = get_3d_point(cx, cy, depth_frame)
                centers_3d.append(point_3d)
                color_names.append(color_name)
                box_colors.append(box_color)

                # 可视化框
                cv2.rectangle(color_image, (x, y), (x + w, y + h), box_color, 2)
                cv2.circle(color_image, (cx, cy), 5, (255, 255, 255), -1)

        # 坐标变换
        custom_coords = transform_camera_to_custom_coordsystem(centers_3d)

        # 输出
        for coord, name in zip(custom_coords, color_names):
            if coord:
                print(f"{name} (custom coord): ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) m")

        # 显示画面和掩膜
        cv2.imshow("Color Frame", color_image)
        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Yellow Mask", yellow_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
