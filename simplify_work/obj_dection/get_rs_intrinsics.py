import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, 30)
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()


# 获取深度传感器的深度标尺
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Width:", intr.width)
print("Height:", intr.height)
print("ppx:", intr.ppx)
print("ppy:", intr.ppy)
print("fx:", intr.fx)
print("fy:", intr.fy)
print("model:", intr.model)
print("coeffs:", intr.coeffs)

print("depth_scale:",depth_scale)

pipeline.stop()

