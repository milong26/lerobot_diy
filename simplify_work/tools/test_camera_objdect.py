import pyrealsense2 as rs
import cv2
import numpy as np
import datetime

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("启动摄像头，识别蓝夹子与淡黄色物体，按 'q' 键退出...")

try:
    while True:
        # 获取彩色帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 转换为 BGR 图像
        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        found = False  # 是否检测到任一物体（用于保存）

        ### 🔵 蓝色夹子检测 ###
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色框
                cv2.putText(color_image, "Blue Clip", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                found = True

        ### 🟡 淡黄色物体检测 ###
        lower_yellow = np.array([20, 50, 100])  # 可根据实际调节
        upper_yellow = np.array([35, 150, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_yellow:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 黄色框
                cv2.putText(color_image, "Yellow Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                found = True

        # 显示图像
        cv2.imshow("Blue Clip + Yellow Object Detection", color_image)

        # 如果检测到物体，则保存
        if found:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.png"
            cv2.imwrite(filename, color_image)
            print(f"物体检测到，图像已保存为: {filename}")

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
