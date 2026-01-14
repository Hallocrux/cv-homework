import cv2
import numpy as np
import scipy.ndimage as ndimage
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class GameObject:
    """[Level 3] 物理实体对象"""
    id: int
    label: str
    center: np.ndarray
    voxels: np.ndarray
    bbox: np.ndarray

    def check_collision(self, pointer_pos: np.ndarray, radius=0.03) -> Tuple[bool, Optional[np.ndarray]]:
        """
        [New] 检测指尖碰撞
        :param radius: 接触半径 (米)
        """
        # A. 快速 AABB 排除
        if (pointer_pos[0] < self.bbox[0]-radius or pointer_pos[0] > self.bbox[3]+radius or
            pointer_pos[1] < self.bbox[1]-radius or pointer_pos[1] > self.bbox[4]+radius or
            pointer_pos[2] < self.bbox[2]-radius or pointer_pos[2] > self.bbox[5]+radius):
            return False, None
            
        # B. 精确体素距离检测
        # 计算指尖到所有体素的距离 (利用广播机制)
        dists = np.linalg.norm(self.voxels - pointer_pos, axis=1)
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < radius:
            return True, self.voxels[min_idx]
        return False, None

# modules/world.py
# (保留顶部的 import 和 GameObject 类定义)

class VoxelWorld:
    def __init__(self, space_cfg: Dict):
        self.cfg = space_cfg
        
        # 1. 空间初始化 (保持不变)
        xr, yr, zr = self.cfg['x_range'], self.cfg['y_range'], self.cfg['z_range']
        self.res = self.cfg['resolution']
        x = np.arange(xr[0], xr[1], self.res)
        y = np.arange(yr[0], yr[1], self.res)
        z = np.arange(zr[0], zr[1], self.res)
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing='ij')
        self.points_3d = np.vstack((self.xx.flatten(), self.yy.flatten(), self.zz.flatten())).T
        self.count = self.points_3d.shape[0]
        
        # 2. 状态初始化 (保持不变)
        self.log_odds = np.zeros(self.count, dtype=np.float32)
        self.occupancy = np.zeros(self.count, dtype=np.float32)
        self.label_ids = np.full(self.count, -1, dtype=int)
        self.label_map = {} 
        
        # 3. 贝叶斯参数 (保持不变)
        self.lo_hit_base = self.cfg.get('prob_hit', 0.2) 
        self.lo_miss_base = self.cfg.get('prob_miss', 0.15)
        self.lo_max = self.cfg.get('lo_max', 3.5)
        self.lo_min = self.cfg.get('lo_min', -3.5)
        
        # 4. [New] 加载可视化配置
        self.vis_cfg = self.cfg.get('visualization', {})
        # 默认值回退策略
        self.thresh_scan = self.vis_cfg.get('render_threshold_scan', 0.5)
        self.thresh_lock = self.vis_cfg.get('render_threshold_lock', 0.9)
        self.enable_high_conf = self.vis_cfg.get('enable_high_conf_highlight', True)
        self.thresh_high_conf = self.vis_cfg.get('high_conf_threshold', 0.9)
        
        # 颜色转换 (List -> Numpy Array)
        self.col_scan_base = np.array(self.vis_cfg.get('color_scan_base', [0, 255, 0]), dtype=np.uint8)
        self.col_scan_high = np.array(self.vis_cfg.get('color_scan_high', [0, 0, 255]), dtype=np.uint8)
        self.col_lock = np.array(self.vis_cfg.get('color_locked', [255, 0, 0]), dtype=np.uint8)
        
        # 为了兼容 analyze_objects，还是保留一个 threshold_high 属性
        self.threshold_high = self.thresh_lock 
        
        print(f"[World] Voxel Grid Initialized: {self.count} voxels.")

    def reset(self):
        self.log_odds[:] = 0.0
        self.occupancy[:] = 0.0
        self.label_ids[:] = -1
        self.label_map.clear()
        print("[World] Reset complete.")

    def update(self, detections: List[Dict], rvec, tvec, mtx, dist, frame_shape, hand_mask: Optional[np.ndarray] = None):
        """
        核心更新逻辑 (包含动态遮挡剔除)
        :param detections: YOLO 检测结果
        :param hand_mask: MediaPipe 生成的手部遮挡掩膜 (True=被手挡住)
        """
        # --- 1. 几何投影 ---
        img_points, _ = cv2.projectPoints(self.points_3d, rvec, tvec, mtx, dist)
        img_points = img_points.reshape(-1, 2)
        
        # 数值安全检查
        if not np.all(np.isfinite(img_points)): return

        h, w = frame_shape[:2]
        # 裁剪防止溢出 (扩大一点范围容错)
        np.clip(img_points[:, 0], -w, 2*w, out=img_points[:, 0])
        np.clip(img_points[:, 1], -h, 2*h, out=img_points[:, 1])

        u = np.round(img_points[:, 0]).astype(int)
        v = np.round(img_points[:, 1]).astype(int)
        
        # 过滤视野外的点
        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0: return

        # --- 2. 观测模型构建 ---
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]

        # A. 检查手部遮挡 (Dynamic Occlusion Culling)
        # 如果提供了 hand_mask，查表看哪些点落在手的区域
        is_blocked_by_hand = np.zeros(len(valid_indices), dtype=bool)
        if hand_mask is not None:
            # hand_mask[y, x] > 0 表示该像素是手
            is_blocked_by_hand = hand_mask[v_valid, u_valid] > 0

        # B. 检查 YOLO 命中
        current_hit = np.zeros(len(valid_indices), dtype=bool)
        current_weights = np.zeros(len(valid_indices), dtype=np.float32)
        current_label = np.full(len(valid_indices), -1, dtype=int)

        for det in detections:
            mask = det['mask'] 
            label = det['label']
            conf = det.get('conf', 0.8)
            
            lbl_hash = hash(label) % 10000 
            self.label_map[lbl_hash] = label
            
            # 检查 YOLO Mask 覆盖
            hits = mask[v_valid, u_valid] == 1
            
            current_hit[hits] = True
            current_label[hits] = lbl_hash
            current_weights[hits] = np.maximum(current_weights[hits], conf)

        # --- 3. 贝叶斯更新 (关键逻辑) ---
        
        # 情况 A: Hit (命中物体)
        # 策略：只要 YOLO 说这里有东西，我们通常选择相信并加分（即使被手部分遮挡）
        hit_indices = valid_indices[current_hit]
        if len(hit_indices) > 0:
            weights = current_weights[current_hit]
            # 动态权重：基础分 * (0.5 + 0.5 * 置信度)
            dynamic_hit = self.lo_hit_base * (0.5 + 0.5 * weights)
            self.log_odds[hit_indices] += dynamic_hit
            self.label_ids[hit_indices] = current_label[current_hit]

        # 情况 B: Miss (未命中) -> 雕刻/扣分
        # 策略：只有在 【未被 YOLO 命中】 且 【未被手遮挡】 时才扣分！
        # 如果 is_blocked_by_hand 为 True，则这一帧跳过更新，保护原来的体素。
        miss_local_indices = (~current_hit) & (~is_blocked_by_hand)
        miss_indices = valid_indices[miss_local_indices]
        
        if len(miss_indices) > 0:
            self.log_odds[miss_indices] -= self.lo_miss_base
        
        # --- 4. 状态结算 ---
        # 钳制数值
        np.clip(self.log_odds, self.lo_min, self.lo_max, out=self.log_odds)
        
        # 更新 Occupancy (Sigmoid)
        np.exp(-self.log_odds, out=self.occupancy)
        self.occupancy += 1.0
        np.reciprocal(self.occupancy, out=self.occupancy)

    def analyze_objects(self, min_voxels=50) -> List[GameObject]:
        """
        [Level 3] 连通域分析 + 实体提取
        当用户锁定世界时调用，将散乱的体素打包成 GameObject
        """
        print("[World] Analyzing connected components...")
        
        # 1. 二值化：只分析确信存在的点
        # 使用 threshold_high (通常是 0.9) 来提取高质量核心
        binary_grid = (self.occupancy > self.threshold_high).reshape(self.xx.shape)
        
        # 2. Scipy 连通域标记
        # structure=3x3x3 全连接 (对角线也算)
        structure = ndimage.generate_binary_structure(3, 3)
        labeled_array, num_features = ndimage.label(binary_grid, structure=structure)
        
        objects = []
        label_grid = self.label_ids.reshape(self.xx.shape)
        
        # 3. 遍历提取
        for i in range(1, num_features + 1):
            # 获取该团块的所有索引 (tuple of arrays)
            indices = np.where(labeled_array == i)
            
            # 体积筛选
            count = len(indices[0])
            if count < min_voxels:
                continue
            
            # 提取物理坐标 (利用 meshgrid)
            ox = self.xx[indices]
            oy = self.yy[indices]
            oz = self.zz[indices]
            obj_voxels = np.vstack((ox, oy, oz)).T
            
            # 语义投票 (Majority Voting)
            obj_labels = label_grid[indices]
            valid_lbls = obj_labels[obj_labels != -1]
            
            final_label = "Unknown"
            if len(valid_lbls) > 0:
                counts = np.bincount(valid_lbls)
                most_freq_hash = np.argmax(counts)
                final_label = self.label_map.get(most_freq_hash, "Unknown")
            
            # 计算包围盒与重心
            center = np.mean(obj_voxels, axis=0)
            bbox = np.array([
                np.min(obj_voxels[:,0]), np.min(obj_voxels[:,1]), np.min(obj_voxels[:,2]),
                np.max(obj_voxels[:,0]), np.max(obj_voxels[:,1]), np.max(obj_voxels[:,2])
            ])
            
            new_obj = GameObject(i, final_label, center, obj_voxels, bbox)
            objects.append(new_obj)
            
        print(f"[World] Found {len(objects)} valid objects.")
        return objects

    def get_render_data(self, is_locked: bool, hand_mask: Optional[np.ndarray] = None, frame_shape=None, mtx=None, dist=None, rvec=None, tvec=None):
        """
        [Updated] 完全可配置的渲染逻辑
        """
        # 1. 根据模式决定渲染阈值
        threshold = self.thresh_lock if is_locked else self.thresh_scan
        visible_mask = self.occupancy > threshold
        
        if not np.any(visible_mask): return None, None
        
        pts = self.points_3d[visible_mask]
        probs = self.occupancy[visible_mask] # 只需要取出来这一部分概率用于判断颜色
        
        # 2. 视觉遮挡剔除 (保持不变)
        if hand_mask is not None and frame_shape is not None and rvec is not None:
            img_pts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            h, w = frame_shape[:2]
            valid_uv = (img_pts[:,0] >= 0) & (img_pts[:,0] < w) & (img_pts[:,1] >= 0) & (img_pts[:,1] < h)
            keep_mask = np.ones(len(pts), dtype=bool)
            
            u_in = img_pts[valid_uv, 0]
            v_in = img_pts[valid_uv, 1]
            is_blocked = hand_mask[v_in, u_in] > 0
            
            indices_in_view = np.where(valid_uv)[0]
            indices_blocked = indices_in_view[is_blocked]
            keep_mask[indices_blocked] = False
            
            # 应用剔除
            pts = pts[keep_mask]
            probs = probs[keep_mask] # 概率也要跟着切片
            if len(pts) == 0: return None, None

        # 3. 颜色着色 (根据 Config)
        colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
        
        if is_locked:
            # 锁定模式：使用单一颜色
            colors[:] = self.col_lock
        else:
            # 扫描模式
            if self.enable_high_conf:
                # 基础颜色
                colors[:] = self.col_scan_base
                # 高置信度覆盖颜色
                high_conf_mask = probs > self.thresh_high_conf
                colors[high_conf_mask] = self.col_scan_high
            else:
                # 不启用高亮，全用基础色
                colors[:] = self.col_scan_base
            
        return pts, colors