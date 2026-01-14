import cv2
import numpy as np
import time
from modules.infrastructure import Config, WebcamStream
from modules.localization import ArUcoSystem
from modules.perception import PerceptionSystem, HandSystem
from modules.world import VoxelWorld
from modules.drawing import AirCanvas

class DigitalTwinApp:
    def __init__(self):
        print("\n=== 初始化数字孪生系统 (Level 7: High Performance) ===")
        
        # 1. 加载配置
        self.config = Config("config.json")
        self.keyframe_cfg = self.config.get("keyframe", default={"min_dist": 0.02, "min_angle": 0.05})
        self.hand_config = self.config.get("hand_model")
        
        # 获取可视化配置
        vis_cfg = self.config.get('voxel_space').get('visualization', {})
        self.touch_color = tuple(int(c) for c in vis_cfg.get('color_touched', [255, 255, 0]))
        
        # 2. 初始化子系统
        camera_src = self.config.get('camera', 'source')
        print(f"[Init] 启动摄像头 (Source: {camera_src})...")
        self.cam = WebcamStream(camera_src).start()
        time.sleep(1.0)
        
        self.aruco_sys = ArUcoSystem(
            self.config.get('aruco'), 
            self.config.get('camera', 'calib_file')
        )
        
        self.perception = PerceptionSystem(self.config.get('yolo'))
        self.hand_sys = HandSystem(model_path='hand_landmarker.task')
        self.world = VoxelWorld(self.config.get('voxel_space'))
        self.canvas = AirCanvas()
        
        # 3. 运行时状态
        self.is_scanning = True 
        self.running = True
        self.last_pose_t = None
        self.last_pose_r = None
        self.game_objects = [] 
        self.is_drawing_mode = False

        # [优化] 帧计数器，用于逻辑降频
        self.frame_count = 0
        self.cached_detections = [] # 缓存 YOLO 结果

    def should_update_map(self, rvec, tvec) -> bool:
        if self.last_pose_t is None: return True
        dist_diff = np.linalg.norm(tvec - self.last_pose_t)
        rot_diff = np.linalg.norm(rvec - self.last_pose_r)
        return dist_diff > self.keyframe_cfg['min_dist'] or rot_diff > self.keyframe_cfg['min_angle']

    def _fast_render_points(self, img, pts_3d, colors, rvec, tvec, mtx, dist):
        """
        [性能优化核心] 使用 NumPy 矢量化操作代替 cv2.circle 循环
        """
        if pts_3d is None or len(pts_3d) == 0: return

        # 1. 批量投影
        img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, mtx, dist)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        
        h, w = img.shape[:2]
        
        # 2. 边界检查 (Vectorized)
        # u: x坐标, v: y坐标
        u = img_pts[:, 0]
        v = img_pts[:, 1]
        
        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if not np.any(valid_mask): return
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_colors = colors[valid_mask]
        
        # 3. 直接修改像素内存 (极速)
        # 注意 img 索引是 [y, x] 即 [v, u]
        img[valid_v, valid_u] = valid_colors
        
        # 可选：如果你觉得点太小，可以把周围像素也画上 (会稍微慢一点点，但比 circle 快)
        img[np.clip(valid_v+1, 0, h-1), valid_u] = valid_colors
        img[valid_v, np.clip(valid_u+1, 0, w-1)] = valid_colors

    def run(self):
        print("\n=== 系统就绪 (High FPS Mode) ===")
        print(" [L] : 锁定/解锁 | [D] : 绘画模式 | [R] : 重置 | [Q] : 退出")
        
        while self.running:
            # 1. 获取画面
            frame = self.cam.read()
            if frame is None: continue
            self.frame_count += 1
            
            display_img = frame.copy()
            h, w = frame.shape[:2]
            
            # 2. 必须每帧运行的任务：手部追踪 + ArUco定位
            hand_lms_list, hand_mask = self.hand_sys.process(frame)
            found, rvec, tvec, corners = self.aruco_sys.get_pose(frame)
            
            # 手部 PnP
            cursor_3d = None
            if found and hand_lms_list:
                cursor_3d = self.hand_sys.estimate_finger_pose(
                    hand_lms_list[0], frame.shape, 
                    self.aruco_sys.mtx, self.aruco_sys.dist,
                    rvec, tvec, self.config.data
                )

            if found:
                # 绘制原点
                cv2.drawFrameAxes(display_img, self.aruco_sys.mtx, self.aruco_sys.dist, rvec, tvec, 0.1)
                
                touched_voxels_visual = []

                # --- 绘画模式逻辑 ---
                if self.is_drawing_mode:
                    cv2.putText(display_img, "MODE: AIR DRAWING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if cursor_3d is not None:
                        self.canvas.add_point(cursor_3d)

                # --- 普通模式逻辑 ---
                else:
                    if self.is_scanning:
                        # [性能优化] 关键帧 + 降频更新
                        # 只有动了，且每 5 帧才跑一次 YOLO 和 地图更新
                        if self.should_update_map(rvec, tvec):
                            
                            # 降频检查
                            if self.frame_count % 3 == 0:
                                detections = self.perception.detect(frame)
                                self.world.update(detections, rvec, tvec, self.aruco_sys.mtx, self.aruco_sys.dist, frame.shape, hand_mask)
                                # 缓存用于显示
                                self.cached_detections = detections
                            
                            self.last_pose_t = tvec.copy()
                            self.last_pose_r = rvec.copy()
                            
                            # 显示缓存的标签
                            for det in self.cached_detections:
                                mask_center = np.mean(np.where(det['mask']), axis=1).astype(int)[::-1]
                                if not np.any(np.isnan(mask_center)):
                                    cv2.putText(display_img, det['label'], tuple(mask_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                        else:
                            cv2.putText(display_img, "Stationary", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    
                    else:
                        # 锁定模式交互
                        if cursor_3d is not None and self.game_objects:
                            for obj in self.game_objects:
                                is_hit, _ = obj.check_collision(cursor_3d, radius=0.03)
                                if is_hit:
                                    touched_voxels_visual.append(obj.voxels)
                                    cv2.putText(display_img, f"TOUCH: {obj.label}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.touch_color, 2)

                # --- 极速渲染层 ---
                
                # 1. 渲染体素 (Vectorized)
                pts_3d, colors = self.world.get_render_data(
                    is_locked=not self.is_scanning,
                    hand_mask=hand_mask,
                    frame_shape=frame.shape, mtx=self.aruco_sys.mtx, dist=self.aruco_sys.dist, rvec=rvec, tvec=tvec
                )
                self._fast_render_points(display_img, pts_3d, colors, rvec, tvec, self.aruco_sys.mtx, self.aruco_sys.dist)

                # 2. 渲染高亮 (Vectorized)
                for voxels in touched_voxels_visual:
                    # 构造纯色数组
                    h_colors = np.tile(self.touch_color, (len(voxels), 1))
                    self._fast_render_points(display_img, voxels, h_colors, rvec, tvec, self.aruco_sys.mtx, self.aruco_sys.dist)

                # 3. 渲染画布 (AirCanvas 内部已优化批量投影，直接调用)
                self.canvas.render(display_img, rvec, tvec, self.aruco_sys.mtx, self.aruco_sys.dist)

                # 4. 指尖光标
                if cursor_3d is not None:
                    c_pt_prj, _ = cv2.projectPoints(cursor_3d.reshape(1,3), rvec, tvec, self.aruco_sys.mtx, self.aruco_sys.dist)
                    c_pt = tuple(c_pt_prj.reshape(-1, 2).astype(int)[0])
                    cv2.drawMarker(display_img, c_pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            else:
                cv2.putText(display_img, "ArUco Lost", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # --- 状态文字 ---
            if not self.is_drawing_mode:
                mode_str = "SCANNING" if self.is_scanning else "LOCKED"
                col = (0, 255, 0) if self.is_scanning else (0, 0, 255)
                cv2.putText(display_img, f"Mode: {mode_str}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

            cv2.imshow("Digital Twin v7.0 (Fast)", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.running = False
            elif key == ord('r'):
                self.world.reset(); self.game_objects = []; self.is_scanning = True; self.canvas.clear()
            elif key == ord('l') and not self.is_drawing_mode:
                self.is_scanning = not self.is_scanning
                self.last_pose_t = None
                if not self.is_scanning:
                    self.game_objects = self.world.analyze_objects(min_voxels=100)
            elif key == ord('d'):
                self.is_drawing_mode = not self.is_drawing_mode
                if self.is_drawing_mode: self.canvas.start_stroke()
                else: self.canvas.end_stroke()
            elif key == ord('n') and self.is_drawing_mode:
                self.canvas.end_stroke(); self.canvas.start_stroke()
            elif key == ord('c'):
                self.canvas.clear()

        self.cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DigitalTwinApp()
    app.run()