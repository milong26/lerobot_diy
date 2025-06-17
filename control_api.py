from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.common.robots.so100_follower import SO100FollowerConfig, SO100Follower

# 相机
camera_config = {
    "scene": RealSenseCameraConfig(
        serial_number_or_name="806312060427", width=640, height=480, fps=30,use_depth=True
        ),
    "wrist":OpenCVCameraConfig(
        index_or_path=6, width=640, height=480, fps=30
    )

}

# follower
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0",
    id="congbi",
    cameras=camera_config
)

# leader
teleop_config = SO100LeaderConfig(
    port="/dev/ttyACM1",
    id="zhubi",
)

robot = SO100Follower(robot_config)
teleop_device = SO100Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)