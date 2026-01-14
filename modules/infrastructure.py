import cv2
import json
import threading
import os
import time

class Config:
    """配置管理器"""
    def __init__(self, config_path="config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 未找到")
        with open(config_path, 'r') as f:
            self.data = json.load(f)

    def get(self, section: str, key: str = None, default=None):
        """安全获取配置"""
        sec_data = self.data.get(section, {})
        if key:
            return sec_data.get(key, default)
        return sec_data

class WebcamStream:
    """多线程摄像头流，避免IO阻塞主线程"""
    def __init__(self, src=0, width=1280, height=720):
        # 兼容 IP 摄像头或 USB 摄像头 ID
        if isinstance(src, str) and src.isdigit(): src = int(src)
        
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 减少延迟，只缓冲最新的一帧
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # 设置为守护线程，主程序退出时它也会自动退出
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
            else:
                self.stopped = True # 如果读取失败（如相机断开），标记停止

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
        