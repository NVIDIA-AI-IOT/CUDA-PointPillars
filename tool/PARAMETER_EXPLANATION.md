# PointPillars 参数说明

本文档解释 `main.cpp` 中 PointPillars 推理的参数配置。

## 一、VoxelizationParameter（体素化参数）

### 1. min_range / max_range（检测范围）

```cpp
vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);   // (x_min, y_min, z_min)
vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);  // (x_max, y_max, z_max)
```

**含义**：
- 定义在点云中进行目标检测的 ROI（感兴趣区域）
- **x 方向**：0.0 ~ 69.12 米（前方）
- **y 方向**：-39.68 ~ 39.68 米（左右各约 40 米）
- **z 方向**：-3.0 ~ 1.0 米（地面上下）
- 这个范围是 KITTI 数据集的标准设置

**示意图**：
```
       俯视图（Y-Z平面）
        
   ← -39.68      0         39.68 →
   ┌─────────────────────────────┐
   │                             │
   │  检测区域 (ROI)              │
   │                             │
   └─────────────────────────────┘
   
X方向（正向）：0.0 ───────> 69.12
```

---

### 2. voxel_size（体素大小）

```cpp
vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);  // (dx, dy, dz)
```

**含义**：
- 将点云划分成体素（体素化）时，每个体素的尺寸
- **x 方向**：0.16 米
- **y 方向**：0.16 米
- **z 方向**：4.0 米（高度维度使用较大的值）

**为什么 z 方向是 4.0**：
- 点云在 Z 轴高度方向变化相对平滑
- 使用较大的体素可以减少计算量
- 对于自动驾驶场景，主要关注水平面的目标

**计算网格大小**：
```cpp
vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
```
- **x 方向网格数** = (69.12 - 0.0) / 0.16 = 432 格
- **y 方向网格数** = (39.68 - (-39.68)) / 0.16 = 496 格
- **z 方向网格数** = (1.0 - (-3.0)) / 4.0 = 1 格（单层）

---

### 3. max_voxels（最大体素数）

```cpp
vp.max_voxels = 40000;
```

**含义**：
- 单次推理中允许的最大非空体素数量
- 如果点云太密集，超过此值会被截断
- 这是 TensorRT 引擎的固定输入大小

**为什么是 40000**：
- 平衡内存占用和性能
- 避免模型输入过大
- 对于 KITTI 数据集，通常不会超过这个值

---

### 4. max_points_per_voxel（每个体素的最大点数）

```cpp
vp.max_points_per_voxel = 32;
```

**含义**：
- 每个体素中最多保留的点数
- 如果体素内点数超过 32，会随机采样到 32 个
- 如果不足 32 个，用零填充

**为什么是 32**：
- PointPillars 论文的标准设置
- 平衡特征提取的完整性和计算效率

---

### 5. max_points（最大点云点数）

```cpp
vp.max_points = 300000;
```

**含义**：
- 单帧点云中参与体素化的最大点数
- 超出此范围的点会被忽略

---

### 6. num_feature（特征维度）

```cpp
vp.num_feature = 4;
```

**含义**：
- 每个点的特征维度
- **4 维**：x, y, z, intensity（坐标 + 强度）

---

## 二、PostProcessParameter（后处理参数）

### 1. feature_size（特征图尺寸）

```cpp
pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);
```

**含义**：
- 经过 backbone 下采样后的特征图尺寸
- 由于网络有下采样操作，特征图是原始网格的 1/4
- 对应 216 x 248 的特征图

---

### 2. num_classes（类别数）

```cpp
int num_classes = 3;
```

**含义**：
- 检测的类别数量
- 1 = Car（车辆）
- 2 = Pedestrian（行人）
- 3 = Cyclist（自行车）

---

### 3. num_anchors（锚框数量）

```cpp
int num_anchors = 6;
```

**含义**：
- 每个网格单元（cell）的锚框数量
- 6 个锚框：3 个类别 × 2 个方向（0° 和 90°）

---

### 4. anchors（锚框尺寸）

```cpp
float anchors[24] = {
    3.9,1.6,1.56,0.0,      // Car: w=3.9, l=1.6, h=1.56, 方向=0°
    3.9,1.6,1.56,1.57,     // Car: w=3.9, l=1.6, h=1.56, 方向=90°
    0.8,0.6,1.73,0.0,      // Pedestrian: w=0.8, l=0.6, h=1.73
    0.8,0.6,1.73,1.57,
    1.76,0.6,1.73,0.0,     // Cyclist: w=1.76, l=0.6, h=1.73
    1.76,0.6,1.73,1.57,
};
```

**含义**：
- 每个类别的典型尺寸（宽度、长度、高度）
- 方向 0.0 = 0 弧度，1.57 ≈ π/2 = 90 度
- 这些是基于 KITTI 数据集的统计值

---

### 5. anchor_bottom_heights（锚框底部高度）

```cpp
nvtype::Float3 anchor_bottom_heights{-1.78,-0.6,-0.6};
```

**含义**：
- 每个类别锚框底部离地面的高度
- Car: -1.78m（地面下）
- Pedestrian/Cyclist: -0.6m（地面以上）

---

### 6. score_thresh（分数阈值）

```cpp
float score_thresh = 0.1;
```

**含义**：
- 检测框的置信度阈值
- 低于 0.1 的检测框会被过滤掉
- 这是一个相对宽松的阈值（后续有 NMS 进一步过滤）

---

### 7. dir_offset（方向偏移）

```cpp
float dir_offset = 0.78539;  // ≈ π/4
```

**含义**：
- 用于区分朝向 0° 和 180° 的偏移量
- 方向分类器的参考角度

---

### 8. nms_thresh（NMS 阈值）

```cpp
float nms_thresh = 0.01;
```

**含义**：
- 非极大值抑制（NMS）的 IoU 阈值
- 两个框的 IoU > 0.01 时，保留置信度更高的框
- **注意**：这里的 0.01 很小，可能与实际实现不同（通常 NMS 阈值在 0.4-0.7）

---

## 三、核心流程总结

1. **体素化**：将点云 (x, y, z, i) 转换为体素特征
2. **Backbone**：使用 Pillar Feature Net 提取特征
3. **Detection Head**：生成类别、位置、方向的预测
4. **后处理**：解码锚框、NMS 过滤，输出最终检测框

---

## 四、修改建议

### 如果想检测更远的距离：
```cpp
vp.max_range = nvtype::Float3(100.0f, -50.0f, 1.0);  // 前方扩大到 100m
```

### 如果想更高精度（更大计算量）：
```cpp
vp.voxel_size = nvtype::Float3(0.08f, 0.08f, 4.0f);  // 体素减小一半
vp.max_voxels = 80000;  // 需要相应增加
```

### 调整检测阈值：
```cpp
pp.score_thresh = 0.3;  // 更严格的阈值，减少误检
```

---

## 参考资料

- [PointPillars 论文](https://arxiv.org/abs/1812.05784)
- [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/)
- [CUDA-PointPillars GitHub](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)

