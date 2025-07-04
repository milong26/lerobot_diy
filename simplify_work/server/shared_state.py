import asyncio

image_buffer = {}
lock = asyncio.Lock()
image_queue = asyncio.Queue()
stop_event = asyncio.Event()