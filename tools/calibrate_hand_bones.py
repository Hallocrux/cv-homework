import sys
import os
import time
import json
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.infrastructure import Config, WebcamStream
from modules.localization import ArUcoSystem
from modules.perception import HandSystem

def pixel_to_world_plane(px, py, mtx, rvec, tvec):
    """射线投射到 ArUco 平面"""
    cx, cy, fx, fy = mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]
    ray_cam = np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
    
    R_mat, _ = cv2.Rodrigues(rvec)
    normal_cam = R_mat[:, 2]
    origin_cam = tvec.flatten()
    
    denom = np.dot(normal_cam, ray_cam)
    if abs(denom) < 1e-6: return None
    scale = np.dot(normal_cam, origin_cam) / denom
    if scale < 0: return None
    
    pt_cam = ray_cam * scale
    pt_world = R_mat.T @ (pt_cam - tvec.flatten())
    return pt_world

def run_bone_calibration():
    cfg = Config("config.json")
    cam = WebcamStream(cfg.get('camera', 'source')).start()
    aruco_sys = ArUcoSystem(cfg.get('aruco'), cfg.get('camera', 'calib_file'))
    hand_sys = HandSystem(model_path='hand_landmarker.task')

    # 状态定义
    STATE_WAIT, STATE_COUNTDOWN, STATE_COLLECTING, STATE_PROCESSING = 0, 1, 2, 3
    state = STATE_WAIT
    start_time = 0
    
    # 我们需要收集所有用于PnP的点(0,5,9,13,17) 以及 食指的三节骨头(6,7,8)
    # 数据结构: { point_index: [list of 3d vectors relative to wrist] }
    target_indices = [0, 5, 9, 13, 17, 6, 7, 8]
    raw_data = {idx: [] for idx in target_indices}
    
    print("\n=== 骨骼级精度标定 (Bone Calibration) ===")
    
    while True:
        frame = cam.read()
        if frame is None: continue
        display = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        
        found, rvec, tvec, _ = aruco_sys.get_pose(frame)
        lms_list, _ = hand_sys.process(frame)
        
        if found: cv2.drawFrameAxes(display, aruco_sys.mtx, aruco_sys.dist, rvec, tvec, 0.05)

        # --- 状态机 ---
        if state == STATE_WAIT:
            cv2.putText(display, "Press [S] to Start Bone Calibration", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(display, "KEEP HAND FLAT & FINGERS STRAIGHT", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                state = STATE_COUNTDOWN
                start_time = time.time()
                
        elif state == STATE_COUNTDOWN:
            remain = 3.0 - (time.time() - start_time)
            if remain <= 0:
                state = STATE_COLLECTING
                start_time = time.time()
                for k in raw_data: raw_data[k] = [] # Clear
            else:
                cv2.putText(display, f"{remain:.1f}", (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,165,255), 5)
                
        elif state == STATE_COLLECTING:
            remain = 10.0 - (time.time() - start_time)
            progress = 1.0 - (remain / 10.0)
            
            # UI
            cv2.rectangle(display, (50, h-60), (w-50, h-30), (50,50,50), -1)
            cv2.rectangle(display, (50, h-60), (int(50 + (w-100)*progress), h-30), (0,255,0), -1)
            
            if found and lms_list:
                lms = lms_list[0]
                # 1. 计算手腕绝对位置
                px0 = (lms[0].x * w, lms[0].y * h)
                w0 = pixel_to_world_plane(px0[0], px0[1], aruco_sys.mtx, rvec, tvec)
                
                if w0 is not None:
                    # 2. 收集所有点相对于手腕的向量
                    valid_frame = True
                    temp_frame_data = {}
                    
                    for idx in target_indices:
                        px = (lms[idx].x * w, lms[idx].y * h)
                        w_pt = pixel_to_world_plane(px[0], px[1], aruco_sys.mtx, rvec, tvec)
                        if w_pt is None:
                            valid_frame = False; break
                        # 保存绝对坐标用于计算长度
                        temp_frame_data[idx] = w_pt
                    
                    if valid_frame:
                        for idx, pt in temp_frame_data.items():
                            raw_data[idx].append(pt) # 存绝对坐标，方便后面算距离
                        # 画点
                        for idx in target_indices:
                            px = int(lms[idx].x * w), int(lms[idx].y * h)
                            cv2.circle(display, px, 3, (0,255,0), -1)

            if remain <= 0:
                state = STATE_PROCESSING
                
        elif state == STATE_PROCESSING:
            cv2.putText(display, "Calculating Bone Lengths...", (cx-150, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Calibration", display)
            cv2.waitKey(10)
            
            if len(raw_data[0]) < 50:
                print("Data insufficient. Retry.")
                state = STATE_WAIT
                continue
                
            # --- 核心计算 ---
            # 1. 计算 PnP 刚性模型的点 (相对于 Wrist 的向量)
            # 我们取所有帧的 Wrist->Index 向量的中位数
            pnp_model = {}
            for idx in [5, 9, 13, 17]:
                vecs = []
                for i in range(len(raw_data[idx])):
                    vecs.append(raw_data[idx][i] - raw_data[0][i]) # Vector Wrist->Point
                pnp_model[str(idx)] = np.median(np.array(vecs), axis=0).tolist()
            
            # 2. 计算骨骼长度 (标量)
            # Length 1: 5->6
            # Length 2: 6->7
            # Length 3: 7->8
            bone_lengths = {}
            pairs = [(5,6), (6,7), (7,8)]
            for start, end in pairs:
                dists = []
                for i in range(len(raw_data[start])):
                    d = np.linalg.norm(raw_data[end][i] - raw_data[start][i])
                    dists.append(d)
                bone_lengths[f"{start}_{end}"] = float(np.median(dists))
            
            # 3. 保存
            with open("config.json", 'r') as f:
                cfg_data = json.load(f)
            
            cfg_data['hand_pnp_model'] = pnp_model # 只有手掌刚体
            cfg_data['bone_lengths'] = bone_lengths # 指骨长度
            
            with open("config.json", 'w') as f:
                json.dump(cfg_data, f, indent=4)
                
            print("\n[Done] 骨骼模型已保存!")
            print(f"Index Bone 1 (5-6): {bone_lengths['5_6']*100:.2f} cm")
            print(f"Index Bone 2 (6-7): {bone_lengths['6_7']*100:.2f} cm")
            print(f"Index Bone 3 (7-8): {bone_lengths['7_8']*100:.2f} cm")
            break

        cv2.imshow("Calibration", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bone_calibration()