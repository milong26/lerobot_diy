# server/image_receiver.py
import socket
import struct
import numpy as np
import threading

image_buffer = {}
lock = threading.Lock()

def handle_image_connection(conn):
    while True:
        try:
            # 读取 header 长度
            header_len_bytes = conn.recv(4)
            if not header_len_bytes:
                break
            header_len = struct.unpack('I', header_len_bytes)[0]

            # 读取 header
            header = conn.recv(header_len).decode('utf-8')
            cam_name, shape_str = header.split(':')
            shape = tuple(map(int, shape_str.strip('()').split(',')))

            # 接收图像数据
            num_bytes = np.prod(shape) * 4  # float32
            data = b''
            while len(data) < num_bytes:
                packet = conn.recv(num_bytes - len(data))
                if not packet:
                    break
                data += packet

            img = np.frombuffer(data, dtype=np.float32).reshape(shape)

            # 保存到共享缓存
            with lock:
                image_buffer[cam_name] = img
        except Exception as e:
            print(f"[ImageReceiver] Error: {e}")
            break

    conn.close()

def start_image_server(host='0.0.0.0', port=9000):
    threading.Thread(target=_start_server_thread, args=(host, port), daemon=True).start()

def _start_server_thread(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((host, port))
    server_sock.listen(1)
    print(f"[ImageReceiver] Listening on {host}:{port}")

    conn, addr = server_sock.accept()
    print(f"[ImageReceiver] Connected by {addr}")
    handle_image_connection(conn)
