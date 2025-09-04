import pyrealsense2 as rs
import numpy as np
import cv2
from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取 intrinsics
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# 初始化 VisionProcessor（这里不用传 language_tip_mode）
vp = VisionProcessor()
# vp.fail_counter = 0
# vp.lower_red1 = np.array([0, 70, 50])
# vp.upper_red1 = np.array([10, 255, 255])
# vp.lower_red2 = np.array([170, 70, 50])
# vp.upper_red2 = np.array([180, 255, 255])
# vp.min_contour_area = 50
# vp.gripper_max_area = 200

# 更新 intrinsics 和 depth_scale 到 class 内部
# vp.intrinsics = intrinsics
# vp.depth_scale = depth_scale

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Color")
cv2.setMouseCallback("Color", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        display_img = color_image.copy()
        if clicked_point:
            u, v = clicked_point
            cv2.circle(display_img, (u, v), 5, (0, 0, 255), -1)

            # 计算 3D 坐标
            depth_at_point = depth_frame.get_distance(u, v)
            if depth_at_point > 0:
                pt_camera = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_at_point)
                pt_custom = vp.transform_camera_to_custom_coordsystem([pt_camera])[0]
                print(f"点击点像素坐标: ({u}, {v})")
                print(f"相机坐标系 3D: {pt_camera}")
                print(f"自定义坐标系 3D: {pt_custom}")
            else:
                print("点击点深度无效")

            clicked_point = None  # 处理完清空

        cv2.imshow("Color", display_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
