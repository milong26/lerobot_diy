# server/server_main.py
from image_receiver import start_image_server
from handler import start_data_server
import time

if __name__ == '__main__':
    host='10.10.1.35'
    start_image_server(host=host,port=9100)  # 图像与非图像共用端口
    start_data_server(host=host,port=9101)

    print("[Main] Server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")
