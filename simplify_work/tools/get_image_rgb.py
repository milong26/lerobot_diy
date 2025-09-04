# 打开realsense相机，获得鼠标点击到的像素的颜色
import cv2
import pyrealsense2 as rs
import numpy as np

def pick_color_from_realsense():
    # 配置 Realsense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    selected_color = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_color
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param
            if frame is not None:
                b, g, r = frame[y, x]
                hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                selected_color = {
                    "position": (x, y),
                    "bgr": (int(b), int(g), int(r)),
                    "hsv": (int(hsv[0]), int(hsv[1]), int(hsv[2]))
                }
                print(f"点击位置: {x},{y} | BGR: {selected_color['bgr']} | HSV: {selected_color['hsv']}")

    cv2.namedWindow("Realsense Color Picker", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 注册鼠标回调
            cv2.setMouseCallback("Realsense Color Picker", mouse_callback, color_image)

            cv2.imshow("Realsense Color Picker", color_image)
            key = cv2.waitKey(1)

            if key == 27:  # ESC 退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return selected_color


if __name__ == "__main__":
    color_info = pick_color_from_realsense()
    if color_info:
        print("\n最终选择的颜色信息：")
        print(f"坐标: {color_info['position']}")
        print(f"BGR: {color_info['bgr']}")
        print(f"HSV: {color_info['hsv']}")
