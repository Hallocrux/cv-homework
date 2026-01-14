# 🌌 Mono-RGB Digital Twin & AR Interaction System

> **基于   *单目摄像头*   的 3D 数字孪生、手势物理交互与空中绘画系统**

这是一个纯 Python 实现的增强现实（AR）与数字孪生实验平台。

它 ***不依赖*** 昂贵的深度相机（如 RealSense），仅使用一个 ***普通的手机摄像头***，结合 **计算机视觉（CV）**、**深度学习（DL）** 和 **几何数学**，实现了对桌面环境的 3D 重建、语义理解以及亚厘米级精度的手部物理交互。





https://github.com/user-attachments/assets/a8e87233-b9dc-46f6-bf66-813a336f206e



---

## ✨ 核心特性 (Key Features)

### 1. 🧠 认知型世界模型 (Cognitive Voxel World)

- **贝叶斯占据栅格 (Bayesian Occupancy Grid)**：利用 Log-Odds 算法融合多帧观测数据，去除噪声，构建稳定的 3D 体素地图。

- **语义实体提取 (Semantic Entity Extraction)**：结合 YOLOv8-Seg 语义分割与 3D 连通域分析，将散乱的体素自动聚类为具名的物理实体（如 "Cup #1", "Mouse #2"）。

- **动态遮挡剔除 (Dynamic Occlusion Culling)**：利用手部掩膜实时“保护”被手遮挡的背景区域，防止建图被破坏。

### 2. 🖐️ 几何级手部追踪引擎 (Geometric Hand Engine)

- **黄金三角 PnP (Golden Triangle PnP)**：利用手腕与两侧指根（3个刚性点）进行姿态解算，极大提升了手部弯曲时的鲁棒性。

- **几何骨骼重建 (Geometric Bone Reconstruction)**：不依赖 AI 猜测的深度，而是基于**球体-射线求交 (Sphere-Ray Intersection)** 算法，结合用户标定的真实骨长，数学推导指尖的绝对 3D 坐标。

- **皮肉补偿 (Flesh Offset)**：引入 Z 轴物理补偿，修正骨骼中心与皮肤表面的误差，实现“触碰即触发”。

### 3. 🎨 空中画布 (Air Canvas)

- **3D 空间绘画**：将指尖轨迹保留在 3D 空间中，支持 6DOF 视角的观察。

- **速度-颜色映射**：模拟真实墨水物理特性，移动慢时线条变粗/深色，移动快时变细/浅色。

- **极速渲染**：基于 NumPy Vectorization 的批量投影渲染，支持数千个粒子的流畅显示。

### 4. ⚡ 高性能架构 (High Performance)

- **感知分离**：MediaPipe（手部）满帧运行，YOLO（物体）降频运行。

- **矢量化渲染**：移除了 Python `for` 循环绘图，直接操作图像内存，渲染性能提升 50x+。

- **关键帧机制**：仅在相机移动时更新地图，防止静止过拟合。

---

## 📂 项目结构

Plaintext

```
Project_Root/
├── main.py                   # [入口] 主程序，集成所有子系统
├── config.json               # [配置] 系统参数、标定数据、渲染配置
├── hand_landmarker.task      # [模型] MediaPipe 手部模型文件 (需下载)
├── requirements.txt          # [依赖] Python 库列表
│
├── modules/                  # [核心模块]
│   ├── infrastructure.py     # 配置读取、摄像头多线程读取
│   ├── localization.py       # ArUco 系统 (相机位姿估计)
│   ├── perception.py         # YOLO + MediaPipe + 骨骼重建算法
│   ├── world.py              # 贝叶斯体素世界、连通域分析
│   └── drawing.py            # AirCanvas 空中绘画逻辑
│
└── tools/                    # [工具箱]
    ├── calibrate_hand_bones.py # 骨骼级精度标定工具 (生成 config 数据)
    └── calib/camera_calib.py
```

---

## 🛠️ 安装与准备 (Installation)

1. **环境配置** (推荐 Python 3.10+)
   
   Bash
   
   ```
   pip install -r requirements.txt
   ```

2. **模型下载**
   
   - 下载 hand_landmarker.task 并放入根目录：
     
     Google MediaPipe Models
   
   - (首次运行会自动下载) `yolov8n-seg.pt`。

3. **打印 ArUco 码**
   
   - 打印一张 `DICT_4X4_50` (ID=0) 的 ArUco 码，尺寸建议 5cm，作为世界原点。

4. **打印棋盘格**

5. **配置摄像头**
   
   - 安装 Droidcam。
   
   - 将手机与电脑连接到同一局域网。
   
   - 配置config.json中的ip。

---

## 🚀 使用指南 (Usage)

### 第零步：相机内参标定 (`tools/calib/calibrate_camera.py`)

在运行任何 PnP 或测量算法之前，必须消除摄像头的**畸变（Distortion）并获得准确的焦距（Focal Length）**。运行`tools/calib/calib.py`

### 第一步：手部标定 (Hand Calibration)

为了获得毫米级的交互精度，每个用户需要运行一次标定。

1. 运行 `python tools/calibrate_hand_bones.py`。

2. 将 ArUco 码放在桌面上。

3. 将右手**平放**在桌面上，五指自然张开伸直。

4. 按 **`S`** 键开始，保持不动 10 秒。

5. 程序会自动计算你的指骨长度并写入 `config.json`。

### 第二步：启动主程序

运行 `python main.py`。

### 🎮 交互控制 (Controls)

| **按键** | **功能**            | **说明**                                            |
| ------ | ----------------- | ------------------------------------------------- |
| **L**  | **Lock / Unlock** | 切换 **扫描模式** (绿) 与 **锁定交互模式** (红)。锁定后会提取物体并允许物理碰撞。 |
| **D**  | **Drawing Mode**  | 切换 **绘画模式**。进入后指尖变为画笔。                            |
| **N**  | **New Stroke**    | (绘画模式下) 断开当前笔画，开始新的一笔（适合写字换行）。                    |
| **C**  | **Clear Canvas**  | 清空所有画好的空中线条。                                      |
| **R**  | **Reset**         | 重置体素世界，清空所有记忆。                                    |
| **Q**  | **Quit**          | 安全退出程序。                                           |

---



## ⚙️ Configuration Guide (`config.json`)

`config.json` 是系统的核心配置文件，它定义了硬件参数、虚拟空间范围、物理世界锚点以及用户手部的生物特征数据。

### 1. 📷 Camera (`camera`)

定义视频输入源和内参文件。

- **`source`**: 视频源。填 `0` 代表本地 USB 摄像头，填 `http://...` 代表 IP 摄像头流地址。

- **`calib_file`**: 由 `tools/camera_calib.py` 生成的内参文件路径 (`.npz`)，用于消除镜头畸变。

### 2. 🧠 Perception & Anchors (`yolo`, `aruco`)

- **`yolo`**:
  
  - `model`: 使用的 YOLO 模型（默认为 `yolov8n-seg.pt`，实例分割版）。
  
  - `conf_thres`: 置信度阈值，低于此值的物体会被忽略。

- **`aruco`**: **世界坐标系的基石**。
  
  - `marker_size`: ArUco 码的物理边长（单位：米）。
  
  - `world_layout`: 定义每个 ArUco 码（ID 0-4）在物理世界中的绝对坐标 `[x, y, z]`。系统以此为基准建立 Digital Twin 的坐标系。

### 3. 🧊 Voxel Space (`voxel_space`)

定义数字孪生世界的边界和更新逻辑。

- **`x_range`, `y_range`, `z_range`**: 扫描区域的物理边界（单位：米）。超出此范围的物体将被忽略。

- **`resolution`**: 体素精度。`0.01` 代表 1cm，`0.02` 代表 2cm。**降低此值可提高精度，但会显著增加 CPU 负荷。**

- **Bayesian Update**:
  
  - `prob_hit` / `prob_miss`: 观测到物体时增加的概率 / 未观测到时扣除的概率（Log-Odds 算法）。
  
  - `threshold_high`: 概率大于多少时认为体素是“实心”的。

- **`visualization`**: 渲染风格配置。
  
  - `render_threshold_*`: 扫描模式和锁定模式下的渲染门槛。
  
  - `color_*`: 自定义各状态下的点云颜色 (RGB)。

### 4. 📹 Anti-Jitter (`keyframe`)

防止静止时重复更新地图导致的过拟合。

- **`min_dist` / `min_angle`**: 只有当相机移动超过此距离（米）或角度（弧度）时，才触发一次地图更新。

### 5. 🖐️ Hand Physics Model (`hand_model`, `hand_pnp_model`, `bone_lengths`)

**⚠️ 注意：此部分数据通常由 `tools/calibrate_hand_bones.py` 自动生成，请勿手动修改数值，除非调整 `z_offset`。**

- **`hand_model`**:
  
  - `z_offset`: **皮肉补偿 (Flesh Offset)**。单位：米。由于 PnP 算法基于骨骼中心，而物理交互基于皮肤表面，此数值用于修正指尖高度（通常设为 `0.01` ~ `0.015`）。

- **`hand_pnp_model`**:
  
  - 定义了手掌的**刚体结构**。包含了食指根(5)、中指根(9)、无名指根(13)、小指根(17) 相对于手腕(0) 的 3D 向量。PnP 算法利用此模型解算手掌姿态。

- **`bone_lengths`**:
  
  - 定义了**食指骨骼链**的精确长度（5-6, 6-7, 7-8）。几何重建算法利用这些长度在 3D 空间中通过球体求交推导指尖位置。



## 📐 算法原理简述 (Under the Hood)

1. PnP (Perspective-n-Point):
   
   我们使用 cv2.solvePnP(SOLVEPNP_SQPNP) 解算手掌位姿。不同于传统的 21 点拟合，我们只使用 Wrist(0)-Index(5)-Pinky(17) 这三个解剖学上的刚性点，极大消除了手掌形变带来的 Z 轴抖动。

2. 骨骼链重建 (Bone Chaining):
   
   指尖位置不是由 AI 直接预测的，而是通过求解球体与射线的交点推导得出：$P_{tip} = P_{joint} + L_{bone} \cdot \vec{Ray}$。这保证了无论手指如何弯曲，在 3D 空间中的骨骼长度始终恒定。

3. **双层滤波**:
   
   - **低通滤波**: 处理微小的传感器抖动。
   
   - **动态平滑 (One-Euro Style)**: 根据手部移动速度动态调整平滑因子，静止时极稳，快速移动时无延迟。
