import cv2
import numpy as np
from typing import List, Tuple

class AirCanvas:
    """
    空中画布系统
    负责管理笔画(Strokes)、颜色映射和 3D 渲染
    """
    def __init__(self, max_points_per_stroke=1000):
        self.strokes: List[np.ndarray] = [] # 存储所有的笔画，每个笔画是 (N, 3) 的 numpy 数组
        self.current_stroke: List[np.ndarray] = [] # 当前正在画的笔画 (临时 list)
        self.is_drawing = False
        self.max_points = max_points_per_stroke

    def start_stroke(self):
        """开始新的一笔"""
        self.is_drawing = True
        self.current_stroke = []

    def end_stroke(self):
        """结束当前笔画，存入历史"""
        self.is_drawing = False
        if len(self.current_stroke) > 1:
            # 将 list 转为 numpy (N, 3) 并存入
            stroke_np = np.array(self.current_stroke, dtype=np.float32)
            self.strokes.append(stroke_np)
        self.current_stroke = []

    def add_point(self, point_3d: np.ndarray):
        """
        添加点到当前笔画
        :param point_3d: [x, y, z]
        """
        if not self.is_drawing: return
        
        # 性能/平滑优化：
        # 如果当前点和上一个点距离太近（比如手抖），则忽略，防止点堆积
        if len(self.current_stroke) > 0:
            last_pt = self.current_stroke[-1]
            dist = np.linalg.norm(point_3d - last_pt)
            if dist < 0.002: # 2mm 以内的移动忽略
                return
        
        self.current_stroke.append(point_3d)

    def clear(self):
        self.strokes = []
        self.current_stroke = []

    def render(self, img, rvec, tvec, mtx, dist):
        """
        将所有 3D 笔画投影并画在 img 上
        """
        # 1. 渲染历史笔画
        for stroke in self.strokes:
            self._draw_stroke_gradient(img, stroke, rvec, tvec, mtx, dist)
            
        # 2. 渲染当前正在画的笔画
        if len(self.current_stroke) > 1:
            curr_np = np.array(self.current_stroke, dtype=np.float32)
            self._draw_stroke_gradient(img, curr_np, rvec, tvec, mtx, dist)

    def _draw_stroke_gradient(self, img, points_3d, rvec, tvec, mtx, dist):
        """
        核心渲染：批量投影 + 速度映射颜色
        """
        if len(points_3d) < 2: return

        # A. 批量投影 (Batch Projection) - 性能关键！
        # 一次性把 N 个 3D 点转为 2D，比循环调用快 100 倍
        img_pts, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        
        # B. 绘制线段 (带颜色渐变)
        # 我们需要计算每两个点之间的 3D 距离来决定颜色
        # 速度 v ~ distance
        dists = np.linalg.norm(points_3d[1:] - points_3d[:-1], axis=1)
        
        # 归一化距离用于颜色映射 (假设 0.05m/帧 是很快的速度)
        max_speed_ref = 0.03 
        
        h, w = img.shape[:2]

        # 遍历绘制线段
        # 虽然这里有 Python 循环，但绘图操作是轻量级的
        for i in range(len(img_pts) - 1):
            pt1 = tuple(img_pts[i])
            pt2 = tuple(img_pts[i+1])
            
            # 简单的出界检查 (性能优化)
            if not (0 <= pt1[0] < w and 0 <= pt1[1] < h): continue
            
            # 颜色映射逻辑
            # 距离越长(快) -> H=120(绿/青) -> S=低 (淡)
            # 距离越短(慢) -> H=0 (红) -> S=高 (浓)
            
            speed = dists[i]
            ratio = min(1.0, speed / max_speed_ref) # 0.0 ~ 1.0
            
            # 使用 HSV 空间生成彩虹色
            # 慢(Red) -> 快(Blue/Cyan)
            # Hue: 0 (Red) -> 180 (Cyan in OpenCV hue scale which is 0-180)
            hue = int(ratio * 120) 
            sat = 255
            val = 255
            
            # 转 BGR
            # OpenCV 的 HSV 转换比较麻烦，手动通过 numpy 这里的单个转换不划算
            # 我们用简单的线性插值做个 "热图" (Heatmap) 效果
            # 慢 = (0, 0, 255) 红
            # 快 = (255, 255, 0) 青
            
            b = int(255 * ratio)
            g = int(255 * min(1.0, ratio * 1.5))
            r = int(255 * (1.0 - ratio))
            
            color = (b, g, r)
            
            # 慢的时候画粗一点，快的时候画细一点
            thickness = max(1, int(4 * (1.0 - ratio) + 1))
            
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)