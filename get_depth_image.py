import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

def encode_depth_to_rgb(depth_uint16):
    """
    将16位深度编码为RGB三通道图像
    R通道：高8位
    G通道：低8位
    B通道：全为0
    """
    print(depth_uint16)
    depth_uint16 = depth_uint16.astype(np.uint16)
    r = (depth_uint16 >> 8).astype(np.uint8)     # 高8位
    g = (depth_uint16 & 0xFF).astype(np.uint8)   # 低8位
    b = np.zeros_like(r, dtype=np.uint8)         # B通道置0
    rgb_image = cv2.merge((r, g, b))  # B, G, R（OpenCV格式）
    print("Encoded depth channel stats:")
    print("R channel - max:", r.max(), "min:", r.min(), "mean:", r.mean())
    print("G channel - max:", g.max(), "min:", g.min(), "mean:", g.mean())
    print("B channel - max:", b.max(), "min:", b.min(), "mean:", b.mean())
    print("r:=",r)
    print("g:=",g)
    print("b:=",b)
    return rgb_image

def decode_depth_from_rgb(rgb_image):
    print(rgb_image.max(),rgb_image.min(),rgb_image.mean())
    """
    从编码的RGB图像恢复16位深度图
    输入：BGR格式图像，B通道全0，G通道低8位，R通道高8位
    """
    r, g, b = cv2.split(rgb_image)
    print("r=",r)
    print("g=",g)
    print("b=",b)
    print("解码")
    print("R channel - max:", r.max(), "min:", r.min(), "mean:", r.mean())
    print("G channel - max:", g.max(), "min:", g.min(), "mean:", g.mean())
    print("B channel - max:", b.max(), "min:", b.min(), "mean:", b.mean())
    # 理论上 b 通道应该全0，可用作校验
    depth_uint16 = (r.astype(np.uint16) << 8) | g.astype(np.uint16)
    return depth_uint16

def capture_and_save_images(output_dir="captures", wait_time=2.0):
    os.makedirs(output_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    print("[INFO] 摄像头已启动，等待曝光稳定中...")
    time.sleep(wait_time)

    align = rs.align(rs.stream.color)

    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("[ERROR] 无法捕获图像帧")
        pipeline.stop()
        return
    print("depth_frame.get_distance(320,240):", depth_frame.get_distance(320, 240))

    color_image = np.asanyarray(color_frame.get_data())
    depth_image_raw = np.asanyarray(depth_frame.get_data())
    print("depth_raw:",depth_image_raw)

    # 保存彩色图
    color_path = os.path.join(output_dir, "color.png")
    cv2.imwrite(color_path, color_image)

    # 保存原始16位深度图（单通道）
    depth_raw_path = os.path.join(output_dir, "depth_raw.png")
    cv2.imwrite(depth_raw_path, depth_image_raw)

    # 保存编码后的3通道深度图
    depth_encoded = encode_depth_to_rgb(depth_image_raw)
    depth_encoded_path = os.path.join(output_dir, "depth_encoded.png")
    cv2.imwrite(depth_encoded_path, depth_encoded)

    # 读回编码图，测试解码
    read_encoded = cv2.imread(depth_encoded_path)
    depth_decoded = decode_depth_from_rgb(read_encoded)

    # 对比原始深度和解码深度是否一致（允许无效深度值忽略）
    if np.array_equal(depth_image_raw, depth_decoded):
        print(depth_image_raw)
        print("[INFO] 解码成功，恢复深度与原始完全一致")
    else:
        diff = np.abs(depth_image_raw.astype(np.int32) - depth_decoded.astype(np.int32))
        print(f"[WARNING] 解码深度与原始有差异，最大差值: {diff.max()}")

    pipeline.stop()
    print(f"[SUCCESS] 图像已保存至：{output_dir}")

if __name__ == "__main__":
    capture_and_save_images()
