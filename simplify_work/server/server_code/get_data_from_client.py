# server/server_main.py
from image_receiver import start_image_server
from handler import start_data_server
import time

if __name__ == '__main__':
    start_image_server(port=9000)  # 图像与非图像共用端口
    start_data_server(port=9001)

    print("[Main] Server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")
