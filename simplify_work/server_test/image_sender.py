import socket
import threading
import io
import torch
import time

class ImageSender:
    def __init__(self, server_ip, port):
        self.server_ip = server_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))

    def send_tensor(self, tensor: torch.Tensor, name: str):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        binary = buffer.getvalue()
        length = len(binary)
        name_bytes = name.encode('utf-8')
        name_len = len(name_bytes)

        # Send header: [name_len][name][data_len][data]
        self.sock.sendall(name_len.to_bytes(4, 'big'))
        self.sock.sendall(name_bytes)
        self.sock.sendall(length.to_bytes(8, 'big'))
        self.sock.sendall(binary)

    def close(self):
        self.sock.close()