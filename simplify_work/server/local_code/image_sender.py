import threading
import queue
import socket
import numpy as np
import struct

image_queue = queue.Queue(maxsize=10)
_stop_signal = threading.Event()

def send_image_thread(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"[ImageSender] Connected to {host}:{port}")

    while not _stop_signal.is_set():
        try:
            item = image_queue.get(timeout=1)
        except queue.Empty:
            continue

        if item is None:
            break

        cam_name, img = item["name"], item["data"]
        img_bytes = img.astype(np.float32).tobytes()
        header = f"{cam_name}:{img.shape}".encode('utf-8')
        header_len = struct.pack('I', len(header))
        try:
            sock.sendall(header_len + header + img_bytes)
        except Exception as e:
            print(f"[ImageSender] Error sending image: {e}")
            break

    sock.close()

_thread = None

def start_image_sender(host, port):
    global _thread
    _thread = threading.Thread(target=send_image_thread, args=(host, port), daemon=True)
    _thread.start()

def stop_image_sender():
    _stop_signal.set()
    image_queue.put(None)
    if _thread:
        _thread.join()
