


import cv2
import numpy as np
import pyrealsense2 as rs
import os

# 相机内参
intrinsics = rs.intrinsics()
intrinsics.width = 640
intrinsics.height = 480
intrinsics.ppx = 304.7939453125
intrinsics.ppy = 234.874755859375
intrinsics.fx = 616.6113891601562
intrinsics.fy = 616.5948486328125
intrinsics.model = rs.distortion.inverse_brown_conrady
intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

depth_scale = 0.0010000000474974513  # 深度单位转换为米

# 图像路径
image_dir = "from_videos"
sevenpoints = [
    (557, 149), (633, 261), (612, 236),
    (594, 212), (614, 326), (576, 340), (538, 350)
]

# 半径为2的区域采样
def get_local_depth_mean(depth_map, x, y, r=2):
    h, w = depth_map.shape
    x0 = max(x - r, 0)
    x1 = min(x + r + 1, w)
    y0 = max(y - r, 0)
    y1 = min(y + r + 1, h)
    roi = depth_map[y0:y1, x0:x1]
    valid = roi[roi > 0]
    return int(np.mean(valid)) if valid.size > 0 else 0

# 像素 -> 3D 点
def deproject_point(u, v, depth_value):
    depth_m = depth_value * depth_scale
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m)
    return tuple(point_3d)

# 深度还原处理流程
def restore_depth_from_b_channel(img):
    b_encoded = img[:, :, 0]
    g_encoded = img[:, :, 1]

    b_filtered = cv2.bilateralFilter(b_encoded, d=5, sigmaColor=75, sigmaSpace=75)
    laplacian = cv2.Laplacian(b_filtered, cv2.CV_64F, ksize=1)
    b_restored = b_encoded - laplacian
    b_restored = np.clip(b_restored, 0, 255).astype(np.uint8)

    b_median = cv2.medianBlur(b_restored, 3)

    sobel_x = cv2.Sobel(b_median, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(b_median, cv2.CV_16S, 0, 1, ksize=3)
    gradient = cv2.convertScaleAbs(cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32)))

    jump_mask = (gradient > 40)
    b_local_avg = cv2.blur(b_median, (3, 3))
    b_final = b_median.copy()
    b_final[jump_mask] = b_local_avg[jump_mask]

    depth_restored = ((b_final.astype(np.uint16) << 8) | g_encoded.astype(np.uint16))
    return depth_restored

# 存储每张图中 3D 点
all_points_3d = []

for i in range(64):
    path = os.path.join(image_dir, f"{i}.png")
    img = cv2.imread(path)
    if img is None:
        print(f"读取失败: {path}")
        continue

    depth_map = restore_depth_from_b_channel(img)

    points_3d = []
    for (u, v) in sevenpoints:
        d = get_local_depth_mean(depth_map, u, v, r=2)
        if d == 0:
            print(f"图像 {i} 点 ({u},{v}) 深度缺失")
        point = deproject_point(u, v, d)
        points_3d.append(point)

    all_points_3d.append(points_3d)
    print(f"图像 {i} 处理完成")

    # 累积原点、x方向点、y方向点
    origin_list = []
    x_vecs = []
    y_vecs = []

    for pts in all_points_3d:
        origin = np.array(pts[0])
        x_points = [np.array(pts[1]), np.array(pts[2]), np.array(pts[3])]
        y_points = [np.array(pts[4]), np.array(pts[5]), np.array(pts[6])]

        # 累积原点
        origin_list.append(origin)

        # 累积方向向量（每个都减去 origin）
        # 累积方向向量（基于点对之间的差值）
        x_diffs = [
            x_points[1] - x_points[0],  # pts[2] - pts[1]
            x_points[2] - x_points[0],  # pts[3] - pts[1]
            x_points[2] - x_points[1],  # pts[3] - pts[2]
        ]
        y_diffs = [
            y_points[1] - y_points[0],  # pts[5] - pts[4]
            y_points[2] - y_points[0],  # pts[6] - pts[4]
            y_points[2] - y_points[1],  # pts[6] - pts[5]
        ]
        x_vecs.append(np.mean(x_diffs, axis=0))
        y_vecs.append(np.mean(y_diffs, axis=0))


# 平均
origin_mean = np.mean(origin_list, axis=0)
x_vec_mean = np.mean(x_vecs, axis=0)
y_vec_mean = np.mean(y_vecs, axis=0)
z_vec = np.cross(x_vec_mean, y_vec_mean)

# 单位向量
x_axis = x_vec_mean / np.linalg.norm(x_vec_mean)
y_axis = y_vec_mean / np.linalg.norm(y_vec_mean)
z_axis = z_vec / np.linalg.norm(z_vec)

print("\n====== 平均局部坐标系（64张图）======")
print(f"原点 (O): {origin_mean}")
print(f"X轴方向: {x_axis}")
print(f"Y轴方向: {y_axis}")
print(f"Z轴方向: {z_axis}")
