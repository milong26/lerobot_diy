import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# 引入 VisionProcessor 类（假设在 vision_processor.py 里）
from vision_processor import VisionProcessor


def main():
    # ====== 手动配置 ======
    # 你要识别的物体类别
    colors_to_detect = ["sachet", "router"]  
    # 初始任务描述
    task_batch = ["pick up objects"]  

    # 初始化 VisionProcessor
    vp = VisionProcessor(language_tip_mode="training")

    # ====== 配置 RealSense 相机 ======
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 转 numpy
            color_image = np.asanyarray(color_frame.get_data())   # BGR
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16 深度

            # ====== 处理 batch ======
            rgb_batch = [cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)]  # 转 RGB
            depth_batch = [depth_image]

            # 调用 add_depth_info_to_task
            results = vp.add_depth_info_to_task(
                rgb_batch,
                depth_batch,
                task_batch,
                colors_to_detect=colors_to_detect
            )

            # 打印结果
            print("=== 任务输出 ===")
            for res in results:
                print(res)

            # 显示彩色图像（可选）
            cv2.imshow("RealSense RGB", color_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
