import time
from anyskin import AnySkinProcess
from lerobot.common.forcesensors.WowSkin.lerobot_wowconfig import WowForceSensorConfig
from lerobot.common.forcesensors.forcesensor import ForceSensor
import numpy as np







class WowForceSensor(ForceSensor):

    def __init__(self, config: WowForceSensorConfig):

        super().__init__(config)

        self.config = config
        # self.index_or_path = config.index_or_path
        self.baseline = None
        self.port = config.port
        self.num_mags = config.num_mags
        self.started = False
        

    def connect(self):
        """启动 Anyskin 数据流"""
        if not self.started:
            self.sensor_stream = AnySkinProcess(num_mags=self.num_mags, port=self.port)
            self.sensor_stream.start()
            time.sleep(1.0)  # 等待设备初始化
            self.started = True
            self.get_baseline()
        else:
            print("Anyskin stream already started.")


    def get_baseline(self):
        """
        从实时数据流中采集一定数量的样本，取平均作为新的 baseline
        """
        baseline_data = self.sensor_stream.get_data(num_samples=5)
        # baseline_data = np.array(baseline_data)[:, 1:]  # 跳过时间戳列
        baseline_data = np.array(baseline_data)[:,1:]  # 跳过时间戳列
        baseline = np.mean(baseline_data, axis=0)
        self.baseline=baseline
        return baseline

    def get_data(self, num_samples=10, log_frequency=True):
        if not self.started:
            raise RuntimeError("Sensor stream not started. Call start() first.")
        


        raw_batch = np.array(self.sensor_stream.get_data(num_samples=num_samples))
        # timestamps = raw_batch[:, 0]
        data = np.mean(raw_batch[:, 1:], axis=0) - self.baseline

        # if log_frequency and len(timestamps) >= 2:
        #     duration = timestamps[-1] - timestamps[0]
        #     if duration > 0:
        #         freq = (len(timestamps) - 1) / duration
        #         self._log_f.write(f"{timestamps[0]}:{freq:.2f}\n")
        #         self._log_f.flush()
        return data


    def disconnect(self):
        """停止数据流并清理资源"""
        if self.started:
            self.sensor_stream.pause_streaming()
            self.sensor_stream.join()
            self.started = False
        else:
            print("Sensor stream already stopped.")
        self._log_f.close()