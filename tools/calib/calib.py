import cv2
import numpy as np
import json
import os
import time
from threading import Thread

# ==========================================
# 类：多线程摄像头流 (解决卡顿的核心)
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        # 初始化摄像头
        self.stream = cv2.VideoCapture(src)
        # 优化缓冲区
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
            else:
                self.stopped = True

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 辅助函数：读取配置
# ==========================================
def load_config():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    
    default_config = {
        "camera_source": 0,
        "board_size": [9, 6],
        "min_samples": 15
    }

    if not os.path.exists(config_path):
        print(f"警告: 未找到 {config_path}，将使用默认配置。")
        print("请在同目录下创建 config.json 以指定 IP 地址。")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"成功加载配置: {config_path}")
            
            # 处理 camera_source，如果看起来像数字（USB ID），转为 int
            src = config.get("camera_source", 0)
            if isinstance(src, str) and src.isdigit():
                config["camera_source"] = int(src)
            
            return config
    except Exception as e:
        print(f"读取配置文件出错: {e}，使用默认配置。")
        return default_config

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 加载配置
    config = load_config()
    src = config["camera_source"]
    # board_size 格式是 [列数, 行数]，例如 (9, 6)
    cols, rows = config["board_size"]
    CHECKERBOARD = (cols, rows)
    MIN_SAMPLES = config.get("min_samples", 15)

    print("="*40)
    print(f"摄像头源: {src}")
    print(f"棋盘格内角点: {CHECKERBOARD}")
    print(f"目标采集数量: {MIN_SAMPLES} 张")
    print("="*40)

    # 2. 准备 3D 世界坐标点
    # 也就是 (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = [] # 存储 3D 点
    imgpoints = [] # 存储 2D 像素点

    # 3. 启动多线程摄像头
    print("正在连接摄像头，请稍候...")
    try:
        vs = WebcamStream(src=src).start()
        time.sleep(1.0) # 给摄像头一点预热时间
    except Exception as e:
        print(f"无法打开摄像头: {e}")
        return

    if vs.read() is None:
        print("错误：无法获取图像，请检查 config.json 中的 IP 地址是否正确。")
        vs.stop()
        return

    print("\n=== 操作指南 ===")
    print(" [C] 键: 采集当前帧（必须检测到棋盘格）")
    print(" [Q] 键: 结束采集并开始计算")
    print("================\n")

    count = 0
    
    while True:
        frame = vs.read()
        if frame is None:
            continue

        # 复制一份用于显示
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 寻找角点
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        corners_refined = None
        if ret:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 绘制角点
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners_refined, ret)
            
            # 状态提示
            cv2.putText(display_frame, "Pattern Found!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Show Checkerboard...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示计数
        color = (0, 255, 255) if count < MIN_SAMPLES else (0, 255, 0)
        cv2.putText(display_frame, f"Captured: {count}/{MIN_SAMPLES}", (20, display_frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF

        # 按 'c' 采集
        if key == ord('c'):
            if ret and corners_refined is not None:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                count += 1
                print(f"采集成功 [{count}]")
                # 闪屏反馈
                cv2.imshow('Camera Calibration', np.full_like(display_frame, 255))
                cv2.waitKey(50)
            else:
                print(">> 采集失败：未检测到完整棋盘格，请调整角度。")

        # 按 'q' 退出
        elif key == ord('q'):
            break

    # 4. 清理资源并开始计算
    vs.stop()
    cv2.destroyAllWindows()

    if count > 0:
        print("\n正在计算标定参数 (CPU计算中，请稍候)...")
        h, w = gray.shape[:2]
        
        # 核心标定函数
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        print("\n========== 标定成功 ==========")
        print(f"图像尺寸: {w}x{h}")
        print(f"重投影误差: {ret:.4f}")
        print("\n内参矩阵 (mtx):")
        print(mtx)
        print("\n畸变系数 (dist):")
        print(dist)

        # 保存结果
        save_path = "camera_calib.npz"
        np.savez(save_path, mtx=mtx, dist=dist)
        print(f"\n结果已保存至: {os.path.abspath(save_path)}")
        
    else:
        print("\n未采集任何图像，取消标定。")

if __name__ == "__main__":
    main()