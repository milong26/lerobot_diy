import cv2
import numpy as np
import pyrealsense2 as rs

# 相机内参
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


# 坐标系变换：定义单位向量和原点



x_unit = np.array([ 0.39238597,-0.08091992  ,0.91623426],)
y_unit = np.array([-0.78536786 , 0.05937675 ,-0.61617507],)
z_unit = np.array( [-0.0094724 , -0.99642532 ,-0.08394558])
origin = np.array([ 0.03217833,-0.01095684 ,0.07867188])
R = np.stack([x_unit, y_unit, z_unit], axis=1)  # 相机->自定义坐标变换矩阵

def transform_to_custom_coords(cam_point):
    """相机坐标 -> 自定义坐标"""
    cam_point = np.array(cam_point)
    return np.dot(R.T, cam_point - origin)

def encode_depth_to_rgb(depth_uint16):
    """将16位深度编码为RGB三通道图像"""
    depth_uint16 = depth_uint16.astype(np.uint16)
    r = (depth_uint16 >> 8).astype(np.uint8)
    g = (depth_uint16 & 0xFF).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    rgb_image = cv2.merge((r, g, b))  # BGR顺序
    return rgb_image

def decode_depth_from_rgb(rgb_image):
    """将RGB图像解码回原始16位深度图"""
    r, g, _ = cv2.split(rgb_image)
    depth_uint16 = ((r.astype(np.uint16) << 8) | g.astype(np.uint16))
    return depth_uint16

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame, color_frame = param['depth'], param['color']
        depth_image = np.asanyarray(depth_frame.get_data())  # 原始 uint16 深度图

        # ============ 保存为 RGB 编码图像 ============
        rgb_encoded = encode_depth_to_rgb(depth_image)
        cv2.imwrite("from_real.png", rgb_encoded)
        print("✅ 已保存 depth_image 为 from_real.png")

        # ============ 读取并解码 =============
        rgb_loaded = cv2.imread("from_real.png")
        depth_decoded = decode_depth_from_rgb(rgb_loaded)
        print("✅ 已读取并解码 from_real.png")

        # ============ 一致性检测 ============
        if np.array_equal(depth_image, depth_decoded):
            print("✅ 编码前后深度图完全一致！")
        else:
            diff = np.abs(depth_image.astype(np.int32) - depth_decoded.astype(np.int32))
            max_diff = np.max(diff)
            print("❌ 编码前后有差异，最大差值为：", max_diff)

        # 使用解码后的 depth_decoded 做后续处理
        depth = depth_decoded[y, x] * depth_scale
        print("depth =", depth_decoded[y, x])
        print("成 scale 之后 =", depth)
        if depth <= 0:
            print(f"点击像素 ({x}, {y}) 深度无效")
            return

        cam_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        print(f"点击像素 ({x}, {y})")
        print(f"相机坐标: {np.round(cam_point, 4)}")

        # 若要转换到自定义坐标，取消注释下面一行
        # custom_point = transform_to_custom_coords(cam_point)
        # print(f"自定义坐标: {np.round(custom_point, 4)}\n")

def run_click_3d_viewer():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    cv2.namedWindow("RealSense Viewer", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("RealSense Viewer", color_image)
            cv2.setMouseCallback("RealSense Viewer", mouse_callback, {'depth': depth_frame, 'color': color_frame})

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# 运行
if __name__ == "__main__":
    run_click_3d_viewer()
