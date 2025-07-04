import asyncio
import struct
import numpy as np
from ..shared_state import image_buffer, lock

async def handle_image_connection(reader, writer):
    while True:
        try:
            header_len = struct.unpack("I", await reader.readexactly(4))[0]
            header = (await reader.readexactly(header_len)).decode()
            _, shape_str = header.split(":")
            shape = tuple(map(int, shape_str.strip("() ").split(",")))
            total_bytes = np.prod(shape) * 2  # float16
            data = await reader.readexactly(total_bytes)
            images = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)

            async with lock:
                image_buffer['scene'] = images[0]
                image_buffer['scene_depth'] = images[1]
                image_buffer['wrist'] = images[2]
        except Exception as e:
            print(f"[ImageReceiver] Error: {e}")
            break

    writer.close()
    await writer.wait_closed()

async def start_image_server_async(host='0.0.0.0', port=9100):
    server = await asyncio.start_server(handle_image_connection, host, port)
    print(f"[ImageReceiver] Listening on {host}:{port}")
    async with server:
        await server.serve_forever()