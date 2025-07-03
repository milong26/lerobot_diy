from scservo_sdk import PortHandler, PacketHandler

port = PortHandler('/dev/ttyACM1')
port.openPort()
# 常见波特率列表，你可以逐个试
for baudrate in [115200, 57600, 9600]:
    port.setBaudRate(baudrate)
    print(f"Testing baudrate: {baudrate}")
    packet = PacketHandler(1.0)
    for i in range(0, 10):
        model_number, comm_result, error = packet.ping(port, i)
        if comm_result == 0:
            print(f"Motor found at ID={i}, model={model_number} at baudrate {baudrate}")
        else:
            print(f"No response from ID={i} at baudrate {baudrate}")
