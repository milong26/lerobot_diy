# server/image_receiver.py
import socket
import struct
import numpy as np
import threading

image_buffer = {}
lock = threading.Lock()


def recv_all(sock, num_bytes):
    """确保从 socket 接收完整 num_bytes 字节"""
    buffer = b''
    while len(buffer) < num_bytes:
        packet = sock.recv(num_bytes - len(buffer))
        if not packet:
            raise ConnectionError("Socket closed while receiving data.")
        buffer += packet
    return buffer


def handle_image_connection(conn):
    while True:
        try:
            # 确保完整读取 header_len（4字节）
            header_len_bytes = recv_all(conn, 4)
            header_len = struct.unpack('I', header_len_bytes)[0]

            # 确保完整读取 header
            header_bytes = recv_all(conn, header_len)
            header = header_bytes.decode('utf-8')
            cam_name, shape_str = header.split(':')
            shape = tuple(map(int, shape_str.strip('()').split(',')))

            # 接收图像数据
            num_bytes = np.prod(shape) * 4  # float32
            data = recv_all(conn, num_bytes)

            img = np.frombuffer(data, dtype=np.float32).reshape(shape)

            # 保存到共享缓存
            with lock:
                image_buffer[cam_name] = img
        except Exception as e:
            print(f"[ImageReceiver] Error: {e}")
            break

    conn.close()


def start_image_server(host='0.0.0.0', port=9100):
    threading.Thread(target=_start_server_thread, args=(host, port), daemon=True).start()


def _start_server_thread(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    print(f"[ImageReceiver] Listening on {host}:{port}")

    while True:
        conn, addr = server_sock.accept()
        print(f"[ImageReceiver] Connected by {addr}")
        threading.Thread(target=handle_image_connection, args=(conn,), daemon=True).start()

