import cv2
import numpy as np

# 读图
img = cv2.imread("1.png")  # BGR格式
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义颜色范围
color_ranges = {
    "green": [(np.array([50, 50, 50]), np.array([90, 255, 255]))],        # 绿色范围示例
    "grey": [(np.array([100, 10, 60]), np.array([130, 80, 200]))]    # 灰蓝色范围示例
}

# 创建掩码并可视化
vis_img = img.copy()

for color_name, ranges in color_ranges.items():
    mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.inRange(hsv_img, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)

    # 检查是否有像素符合要求
    if np.any(mask_total > 0):
        print(f"图像中存在 {color_name} 的像素点")
    else:
        print(f"图像中不存在 {color_name} 的像素点")

    # 用不同颜色绘制符合要求的像素点
    # green -> 红色标记，grey -> 蓝色标记
    color_bgr = (0, 0, 255) if color_name == "green" else (255, 0, 0)
    vis_img[mask_total > 0] = color_bgr

# 保存可视化结果
cv2.imwrite("color_detection_result.jpg", vis_img)
print("可视化结果已保存为 color_detection_result.jpg")
