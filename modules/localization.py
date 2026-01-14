import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional

class ArUcoSystem:
    """负责处理坐标系转换和相机位姿估计"""
    def __init__(self, aruco_cfg: Dict, calib_file: str):
        self.cfg = aruco_cfg
        
        # 1. 加载相机内参
        if not os.path.exists(calib_file):
            print(f"[Localization] 警告: 标定文件 '{calib_file}' 未找到。将使用默认参数（可能导致定位不准）。")
            # 默认内参 (适用于一般 720p 摄像头)
            self.mtx = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
            self.dist = np.zeros(5)
        else:
            with np.load(calib_file) as X:
                self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]

        # 2. 初始化 ArUco 检测器
        dict_name = self.cfg.get('dict_type', 'DICT_4X4_50')
        # getattr 用于动态获取 cv2.aruco 下的常量
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        except AttributeError:
            print(f"[Localization] 错误: ArUco 字典类型 '{dict_name}' 无效。")
            raise

        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        
        # 3. 构建世界坐标锚点 (World Anchors)
        self.world_anchors = {}
        s = self.cfg.get('marker_size', 0.05) / 2.0
        # 预计算：单个 Marker 相对于其中心的四个角点坐标 (本地坐标系)
        self.corner_offsets = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        
        layout = self.cfg.get('world_layout', {})
        for mid, coords in layout.items():
            self.world_anchors[int(mid)] = np.array(coords, dtype=np.float32)
            
        print(f"[Localization] 系统初始化完成。加载了 {len(self.world_anchors)} 个世界锚点。")

    def get_pose(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], list]:
        """
        计算相机在世界坐标系下的位姿
        Returns: (Success, rvec, tvec, corners)
        """
        if frame is None:
            return False, None, None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None:
            obj_points = []
            img_points = []
            
            # 遍历所有检测到的 Marker
            for i, marker_id in enumerate(ids.flatten()):
                mid = int(marker_id)
                # 只有当 Marker ID 在我们的 config.json 布局里时，才用于定位
                if mid in self.world_anchors:
                    center = self.world_anchors[mid]
                    # 将 Marker 的四个角转换到世界坐标系
                    obj_points.append(center + self.corner_offsets)
                    # 收集对应的像素坐标
                    img_points.append(corners[i].reshape(-1, 2))
            
            # 只有凑够了点，才能解算 PnP
            if len(obj_points) > 0:
                obj_pts = np.vstack(obj_points).astype(np.float32)
                img_pts = np.vstack(img_points).astype(np.float32)
                
                # Solve PnP (Iterative 方法比较通用且稳健)
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
                
                # 数值稳定性检查：防止解算出 NaN 或 Inf 导致后续程序崩溃
                if success:
                    if np.any(np.isnan(rvec)) or np.any(np.isinf(tvec)):
                        return False, None, None, corners
                    
                return success, rvec, tvec, corners
                
        return False, None, None, corners