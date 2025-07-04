import asyncio
from image_receiver_async import start_image_server_async
from handler import start_data_server

if __name__ == '__main__':
    host = '0.0.0.0'
    asyncio.run(start_image_server_async(host, 9100))
    start_data_server(host, 9101)