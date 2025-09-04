import pyrealsense2 as rs
import numpy as np
import cv2
from simplify_work.obj_dection.detector_api_with_opencv import VisionProcessor

def main():
    # 初始化VisionProcessor，使用training模式
    processor = VisionProcessor(language_tip_mode="training")
    
    # 获取除了gripper以外的所有颜色名称
    colors_to_detect = [c for c in processor.color_ranges.keys() if c != "gripper"]
    
    # 配置RealSense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动相机
    pipeline.start(config)
    
    try:
        while True:
            # 等待并获取帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 将深度图编码为彩色图
            depth_r = (depth_image >> 8).astype(np.uint8)  # 高8位
            depth_g = (depth_image & 0xFF).astype(np.uint8)  # 低8位
            depth_b = np.zeros_like(depth_r)
            depth_encoded = cv2.merge([depth_r, depth_g, depth_b])
            
            # 创建显示图像（原始彩色图）
            display_image = color_image.copy()
            
            # 检测并绘制所有物体（包括gripper）
            # 1. 检测gripper - 修复：使用opencv_detect_color函数并正确处理bbox
            gripper_center, gripper_bbox = processor.opencv_detect_color(display_image, "gripper")
            if gripper_bbox:
                x, y, w, h = gripper_bbox
                # 绘制gripper框 - 修复：添加红色框
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # 添加标签 - 修复：添加"gripper"文本标签
                cv2.putText(display_image, "gripper", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 2. 检测其他物体
            for color_name in colors_to_detect:
                center, bbox = processor.opencv_detect_color(display_image, color_name)
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_image, color_name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('RealSense Detection', display_image)
            
            # 处理按键
            key = cv2.waitKey(1)
            if key == ord('a'):
                # 调用add_depth_info_to_task函数
                updated_tasks = processor.add_depth_info_to_task(
                    rgb_batch=[color_image],
                    depth_batch=[depth_encoded],
                    task_batch=["test"],
                    colors_to_detect=colors_to_detect
                )
                print("Updated Task:", updated_tasks[0])
            
            elif key == ord('q'):
                break
    
    finally:
        # 停止并关闭相机
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()