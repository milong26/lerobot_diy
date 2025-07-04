import asyncio
import numpy as np
import struct
from shared_state import image_queue, stop_event

async def image_sender(host, port):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"[ImageSender] Connected to {host}:{port}")

    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(image_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if item is None:
            break

        combined = np.stack([
            item['scene'], item['scene_depth'], item['wrist']
        ]).astype(np.float16)
        header = f"triple:{combined.shape}".encode('utf-8')
        header_len = struct.pack('I', len(header))
        writer.write(header_len + header + combined.tobytes())
        await writer.drain()

    writer.close()
    await writer.wait_closed()

def start_image_sender_async(host, port):
    asyncio.create_task(image_sender(host, port))

def stop_image_sender_async():
    stop_event.set()
    image_queue.put_nowait(None)