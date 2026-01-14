import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any, Optional
import os

class HandSystem:
    """
    [Updated] 基于 MediaPipe Tasks API + PnP 测距的手部系统
    """
    def __init__(self, model_path='hand_landmarker.task'):
        print("[Perception] 初始化手部追踪系统 (MediaPipe Tasks + PnP)...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件 {model_path}！请下载并放入项目根目录。")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        self.prev_world_pos = None 
        # 基础滤波系数 (越小越稳，越大越快)
        self.min_alpha = 0.2  # 静止时的平滑力度 (很稳)
        self.max_alpha = 0.8  # 运动时的响应力度 (很快)
        self.velocity_scale = 10.0 # 用于控制灵敏度

    def process(self, frame: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)
        
        hand_landmarks_list = detection_result.hand_landmarks
        mask_canvas = np.zeros((h, w), dtype=np.uint8)
        
        if hand_landmarks_list:
            for hand_lms in hand_landmarks_list:
                points = []
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                
                # 绘制粗遮挡掩膜
                for start_idx, end_idx in self.HAND_CONNECTIONS:
                    cv2.line(mask_canvas, points[start_idx], points[end_idx], (255), 20)
                for pt in points:
                    cv2.circle(mask_canvas, pt, 10, (255), -1)
                if points:
                    points_np = np.array(points)
                    hull = cv2.convexHull(points_np)
                    cv2.fillConvexPoly(mask_canvas, hull, 255)

        return hand_landmarks_list, (mask_canvas > 0)

    def estimate_depth_and_transform(self, hand_lms, frame_shape, mtx, dist, rvec_aruco, tvec_aruco, hand_config: Dict):
        """
        [Final] 5点 PnP + 全向 3D 动态滤波
        """
        h, w = frame_shape[:2]
        
        # --- A. 5点模型构建 (保持不变) ---
        if 'wrist_to_middle' not in hand_config:
            # 如果没运行标定，回退到旧逻辑或报错
            return None

        # 1. 构建 3D 模型点
        p0 = [0, 0, 0] # Wrist
        p5 = [0, hand_config['wrist_to_index'], 0] # Index
        
        # Pinky (17)
        w_to_p = hand_config['wrist_to_pinky']
        w_to_i = hand_config['wrist_to_index']
        width  = hand_config['palm_width_5_17']
        cos_17 = (w_to_i**2 + w_to_p**2 - width**2) / (2 * w_to_i * w_to_p)
        sin_17 = np.sqrt(max(0, 1 - cos_17**2))
        p17 = [w_to_p * sin_17, w_to_p * cos_17, 0]
        
        # Middle (9) - 简化插值
        theta_17 = np.arctan2(p17[0], p17[1])
        d_9 = hand_config['wrist_to_middle']
        p9 = [d_9 * np.sin(theta_17 * 0.33), d_9 * np.cos(theta_17 * 0.33), 0]
        
        # Ring (13) - 简化插值
        d_13 = hand_config['wrist_to_ring']
        p13 = [d_13 * np.sin(theta_17 * 0.66), d_13 * np.cos(theta_17 * 0.66), 0]
        
        model_points = np.array([p0, p5, p9, p13, p17], dtype=np.float64)
        
        # 2. 获取 5 个像素点
        def get_px(idx): return [hand_lms[idx].x * w, hand_lms[idx].y * h]
        image_points = np.array([
            get_px(0), get_px(5), get_px(9), get_px(13), get_px(17)
        ], dtype=np.float64)
        
        # --- B. PnP 解算 (保持不变) ---
        success, rvec_hand, tvec_hand = cv2.solvePnP(
            model_points, image_points, mtx, dist, 
            flags=cv2.SOLVEPNP_SQPNP
        )
        if not success: return None
        
        # --- C. 原始指尖坐标计算 (保持不变) ---
        uv_tip = get_px(8)
        cx, cy = mtx[0,2], mtx[1,2]
        fx, fy = mtx[0,0], mtx[1,1]
        
        vec_tip_cam = np.array([(uv_tip[0] - cx) / fx, (uv_tip[1] - cy) / fy, 1.0])
        R_hand, _ = cv2.Rodrigues(rvec_hand)
        normal_cam = R_hand @ np.array([0, 0, 1])
        wrist_cam = tvec_hand.flatten()
        
        denom = np.dot(vec_tip_cam, normal_cam)
        if abs(denom) < 1e-6: return None
        scale = np.dot(wrist_cam, normal_cam) / denom
        if scale < 0: return None
        
        p_tip_cam = vec_tip_cam * scale
        
        # 转到世界坐标系 (此时是 Raw Data，含噪声)
        R_aruco, _ = cv2.Rodrigues(rvec_aruco)
        # raw_pos: [x, y, z]
        raw_pos = (R_aruco.T @ (p_tip_cam.reshape(3,1) - tvec_aruco)).flatten()
        
        # --- D. [New] 全向量 3D 动态平滑 ---
        
        # 初始化
        if self.prev_world_pos is None:
            self.prev_world_pos = raw_pos
            return raw_pos

        # 计算移动速度 (欧氏距离)
        # dist_move 代表这帧和上帧的物理距离差异
        dist_move = np.linalg.norm(raw_pos - self.prev_world_pos)
        
        # 动态计算 alpha (One-Euro Filter 的简化版)
        # 移动越快，alpha 越大 (信赖当前值，减少延迟)
        # 移动越慢，alpha 越小 (信赖历史值，增加平滑)
        # 这里的映射函数可以根据手感微调
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(1.0, dist_move * self.velocity_scale)
        
        # 对 XYZ 同时滤波
        filtered_pos = alpha * raw_pos + (1.0 - alpha) * self.prev_world_pos
        
        # 更新历史
        self.prev_world_pos = filtered_pos
        
        return filtered_pos
    
    def _solve_bone_intersection(self, start_pos_cam, ray_dir_cam, bone_length):
        """
        [Math Core] 求解下一节骨头的位置
        已知：起点 P_start, 射线方向 D, 骨长 L
        求解：P_end = origin + t * D，使得 |P_end - P_start| = L
        
        Let P_end = t * D (因为射线从相机原点(0,0,0)出发)
        |t*D - P_start|^2 = L^2
        (t*D - P_start) . (t*D - P_start) = L^2
        t^2(D.D) - 2t(D.P_start) + (P_start.P_start) - L^2 = 0
        """
        # 射线方向必须归一化
        D = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        P = start_pos_cam
        L = bone_length
        
        # 解一元二次方程 at^2 + bt + c = 0
        a = 1.0 # 因为 D 是单位向量，D.D = 1
        b = -2 * np.dot(D, P)
        c = np.dot(P, P) - L**2
        
        delta = b**2 - 4*a*c
        
        if delta < 0:
            # 射线与球体不相交 (理论上不该发生，除非 MediaPipe 极度扭曲)
            # 退化处理：取最近点 (Projection)
            t = -b / (2*a)
        else:
            # 取两个解中较大的那个 (离相机更远的，避免反向解)
            t1 = (-b + np.sqrt(delta)) / (2*a)
            t2 = (-b - np.sqrt(delta)) / (2*a)
            t = max(t1, t2)
            
        return t * D

    def estimate_finger_pose(self, hand_lms, frame_shape, mtx, dist, rvec_aruco, tvec_aruco, full_config):
        """
        [Ultra Pro v2] 黄金三角 PnP + 几何骨骼重建
        只使用 0-5-17 三点定位手掌，极大提升鲁棒性
        """
        h, w = frame_shape[:2]
        
        # 1. 验证配置
        pnp_model = full_config.get('hand_pnp_model')
        bone_lens = full_config.get('bone_lengths')
        if not pnp_model or not bone_lens: return None
        
        # 2. PnP 解算手掌刚体 (只用 Wrist + IndexRoot + PinkyRoot)
        # === [关键修改] 从 5点 改为 3点 ===
        # 这三个点构成了手掌最稳定的平面 (Metacarpal Plane)
        
        # 3D Model Points (Local)
        p0 = np.array([0.,0.,0.])
        p5 = np.array(pnp_model['5'])
        p17 = np.array(pnp_model['17'])
        model_pts = np.vstack([p0, p5, p17])
        
        # 2D Image Points (Pixel)
        def get_px(idx): return [hand_lms[idx].x * w, hand_lms[idx].y * h]
        img_pts = np.array([get_px(0), get_px(5), get_px(17)], dtype=np.float64)
        
        # 使用 SQPNP (最适合 3-4 个点)
        success, rvec_hand, tvec_hand = cv2.solvePnP(model_pts, img_pts, mtx, dist, flags=cv2.SOLVEPNP_SQPNP)
        
        if not success: 
            # 如果 SQPNP 失败 (极少见)，尝试迭代法作为保底
            success, rvec_hand, tvec_hand = cv2.solvePnP(model_pts, img_pts, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success: return None
        
        # 3. 获取相机内参用于生成射线
        cx, cy, fx, fy = mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]
        
        def get_ray(idx):
            px = get_px(idx)
            return np.array([(px[0] - cx)/fx, (px[1] - cy)/fy, 1.0])

        # 4. 逐步重建骨骼链 (Bone Chaining) - 保持不变
        # Step A: 算出 Point 5 (Index Root) 的确切位置
        R_hand, _ = cv2.Rodrigues(rvec_hand)
        p5_local = np.array(pnp_model['5'])
        p5_cam = R_hand @ p5_local + tvec_hand.flatten()
        
        # Step B: 5 -> 6
        ray_6 = get_ray(6)
        p6_cam = self._solve_bone_intersection(p5_cam, ray_6, bone_lens['5_6'])
        
        # Step C: 6 -> 7
        ray_7 = get_ray(7)
        p7_cam = self._solve_bone_intersection(p6_cam, ray_7, bone_lens['6_7'])
        
        # Step D: 7 -> 8 (Tip)
        ray_8 = get_ray(8)
        p8_cam = self._solve_bone_intersection(p7_cam, ray_8, bone_lens['7_8'])
        
        # 5. 转回世界坐标系
        R_aruco, _ = cv2.Rodrigues(rvec_aruco)
        p8_world = R_aruco.T @ (p8_cam.reshape(3,1) - tvec_aruco)
        
        # [皮肉补偿]
        hand_cfg = full_config.get('hand_model', {})
        z_offset = hand_cfg.get('z_offset', 0.01) 
        p8_world[2] += z_offset
        
        # 6. 3D 动态平滑
        raw_pos = p8_world.flatten()
        if self.prev_world_pos is None:
            self.prev_world_pos = raw_pos
            return raw_pos

        dist_move = np.linalg.norm(raw_pos - self.prev_world_pos)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(1.0, dist_move * self.velocity_scale)
        filtered_pos = alpha * raw_pos + (1.0 - alpha) * self.prev_world_pos
        self.prev_world_pos = filtered_pos
        
        return filtered_pos

class PerceptionSystem:
    # 保持不变，照抄之前的即可
    def __init__(self, yolo_cfg: Dict):
        model_path = yolo_cfg.get('model', 'yolov8n-seg.pt')
        print(f"[Perception] 加载 YOLO 模型: {model_path}...")
        self.model = YOLO(model_path)
        self.conf = yolo_cfg.get('conf_thres', 0.5)
        self.target_classes = [i for i in range(80) if i != 0] 

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, classes=self.target_classes, stream=True, verbose=False, conf=self.conf)
        detections = []
        h, w = frame.shape[:2]
        
        for r in results:
            if r.masks is None: continue
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy()
            
            for i, cls_id in enumerate(cls_ids):
                m_resized = cv2.resize(masks[i], (w, h))
                m_binary = (m_resized > 0.5).astype(np.uint8)
                detections.append({
                    'label': self.model.names[cls_id],
                    'conf': confs[i],
                    'mask': m_binary
                })
        return detections