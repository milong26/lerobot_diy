import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random  # 实际使用中你可以替换成传感器读取代码，比如 serial 或其他接口
import numpy as np
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import sys
import pygame
from datetime import datetime
from anyskin import AnySkinProcess
import argparse








sensor_stream = AnySkinProcess(num_mags=5, port="/dev/ttyACM2")
sensor_stream.start()
time.sleep(2.0)  # 等待串口连接稳定
def get_baseline():
    """
    从实时数据流中采集一定数量的样本，取平均作为新的 baseline
    """
    baseline_data = sensor_stream.get_data(num_samples=10)
    baseline_data = np.array(baseline_data)[:, 1:]  # 跳过时间戳列
    baseline = np.mean(baseline_data, axis=0)
    return baseline

baseline=get_baseline()
# 模拟传感器读取函数（请替换）
# def read_sensor():

samples = sensor_stream.get_data(num_samples=256)  # 例如 256 个样本
data_array = np.array(samples)[:, 1:]  # 去掉时间戳

# sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]

data_to_show = data_array- baseline


channel_id = 0  # 假设分析第0个通道
signal = data_array[:, channel_id]  # 取出某一通道的值

# 执行傅里叶变换
fft_result = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), d=0.5)  # 假设采样周期是 0.5 秒，请改成你实际的采样周期

# 取绝对值表示幅度
magnitude = np.abs(fft_result)

# 只画正频率部分
pos_mask = freqs >= 0
plt.plot(freqs[pos_mask], magnitude[pos_mask])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(f"FFT of Force Sensor Channel {channel_id}")
plt.grid()
plt.show()


# 启动动画
# ani = animation.FuncAnimation(fig, update, interval=500)
# plt.tight_layout()
# plt.show()