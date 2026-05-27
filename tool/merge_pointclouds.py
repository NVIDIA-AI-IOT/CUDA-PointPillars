#!/usr/bin/env python3
"""
根据两帧检测结果计算车辆偏移量，还原点云并合并
用法: python merge_pointclouds.py <frame1_txt> <frame2_txt> <frame1_bin> <frame2_bin> <output_pcd>
"""

import sys
import numpy as np
import struct
from pathlib import Path

def load_detection_file(txt_file):
    """加载检测结果文件，返回检测框列表"""
    detections = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                # 格式: x y z w l h rt id score
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                w, l, h = float(parts[3]), float(parts[4]), float(parts[5])
                rt = float(parts[6])  # rotation
                det_id = int(parts[7])
                score = float(parts[8])
                detections.append({
                    'x': x, 'y': y, 'z': z,
                    'w': w, 'l': l, 'h': h,
                    'rt': rt, 'id': det_id, 'score': score
                })
    return detections

def calculate_3d_iou(box1, box2):
    """计算两个3D边界框的IOU（简化版本，使用AABB）"""
    # 计算中心点
    cx1, cy1, cz1 = box1['x'], box1['y'], box1['z']
    cx2, cy2, cz2 = box2['x'], box2['y'], box2['z']
    
    # 计算半尺寸
    hw1, hl1, hh1 = box1['w'] * 0.5, box1['l'] * 0.5, box1['h'] * 0.5
    hw2, hl2, hh2 = box2['w'] * 0.5, box2['l'] * 0.5, box2['h'] * 0.5
    
    # AABB交集
    inter_min_x = max(cx1 - hw1, cx2 - hw2)
    inter_max_x = min(cx1 + hw1, cx2 + hw2)
    inter_min_y = max(cy1 - hl1, cy2 - hl2)
    inter_max_y = min(cy1 + hl1, cy2 + hl2)
    inter_min_z = max(cz1 - hh1, cz2 - hh2)
    inter_max_z = min(cz1 + hh1, cz2 + hh2)
    
    inter_volume = max(0, inter_max_x - inter_min_x) * \
                   max(0, inter_max_y - inter_min_y) * \
                   max(0, inter_max_z - inter_min_z)
    
    volume1 = box1['w'] * box1['l'] * box1['h']
    volume2 = box2['w'] * box2['l'] * box2['h']
    union_volume = volume1 + volume2 - inter_volume
    
    if union_volume < 1e-6:
        return 0.0
    
    return inter_volume / union_volume

def match_detections(detections1, detections2, iou_threshold=0.3):
    """匹配两帧中的检测框，返回匹配对列表"""
    matches = []
    used_indices2 = set()
    
    for i, det1 in enumerate(detections1):
        best_iou = 0.0
        best_idx2 = -1
        
        for j, det2 in enumerate(detections2):
            if j in used_indices2:
                continue
            
            iou = calculate_3d_iou(det1, det2)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_idx2 = j
        
        if best_idx2 >= 0:
            matches.append((i, best_idx2, best_iou))
            used_indices2.add(best_idx2)
    
    return matches

def calculate_offset(detections1, detections2, matches):
    """根据匹配的检测框计算平均偏移量"""
    if not matches:
        print("警告: 没有找到匹配的车辆，使用第一个检测框计算偏移")
        if detections1 and detections2:
            dx = detections2[0]['x'] - detections1[0]['x']
            dy = detections2[0]['y'] - detections1[0]['y']
            dz = detections2[0]['z'] - detections1[0]['z']
            return dx, dy, dz
        return 0.0, 0.0, 0.0
    
    offsets = []
    for idx1, idx2, iou in matches:
        det1 = detections1[idx1]
        det2 = detections2[idx2]
        dx = det2['x'] - det1['x']
        dy = det2['y'] - det1['y']
        dz = det2['z'] - det1['z']
        offsets.append((dx, dy, dz))
        print(f"匹配车辆 {idx1} <-> {idx2} (IOU: {iou:.3f}), 偏移: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
    
    # 计算平均偏移量
    avg_dx = np.mean([o[0] for o in offsets])
    avg_dy = np.mean([o[1] for o in offsets])
    avg_dz = np.mean([o[2] for o in offsets])
    
    print(f"\n平均偏移量: ({avg_dx:.3f}, {avg_dy:.3f}, {avg_dz:.3f})")
    return avg_dx, avg_dy, avg_dz

def load_bin_pointcloud(bin_file):
    """加载.bin格式的点云文件（x, y, z, intensity）"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    
    # 每个点4个float（x, y, z, intensity）
    num_points = len(data) // (4 * 4)
    points = np.frombuffer(data, dtype=np.float32).reshape(num_points, 4)
    return points

def save_pcd_file(points, output_file):
    """保存点云为PCD格式"""
    with open(output_file, 'w') as f:
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        for point in points:
            f.write(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}, {point[3]:.6f}\n")
    
    print(f"已保存合并后的点云到: {output_file}")

def save_bin_file(points, output_file):
    """保存点云为BIN格式（二进制，每个点4个float：x, y, z, intensity）"""
    # 确保是float32类型
    points_float32 = points.astype(np.float32)
    # 写入二进制文件
    with open(output_file, 'wb') as f:
        points_float32.tofile(f)
    
    print(f"已保存合并后的点云到: {output_file}")

def main():
    if len(sys.argv) < 6:
        print("用法: python merge_pointclouds.py <frame1_txt> <frame2_txt> <frame1_bin> <frame2_bin> <output_pcd>")
        sys.exit(1)
    
    frame1_txt = sys.argv[1]
    frame2_txt = sys.argv[2]
    frame1_bin = sys.argv[3]
    frame2_bin = sys.argv[4]
    output_pcd = sys.argv[5]
    
    # 生成对应的.bin文件名
    output_bin = output_pcd.replace('.pcd', '.bin')
    
    print(f"加载检测结果...")
    detections1 = load_detection_file(frame1_txt)
    detections2 = load_detection_file(frame2_txt)
    
    print(f"第一帧检测到 {len(detections1)} 辆车")
    print(f"第二帧检测到 {len(detections2)} 辆车")
    
    # 匹配检测框
    print("\n匹配车辆...")
    matches = match_detections(detections1, detections2)
    print(f"找到 {len(matches)} 个匹配")
    
    # 计算偏移量
    dx, dy, dz = calculate_offset(detections1, detections2, matches)
    
    # 加载点云
    print(f"\n加载点云文件...")
    points1 = load_bin_pointcloud(frame1_bin)
    points2 = load_bin_pointcloud(frame2_bin)
    
    print(f"第一帧点云: {len(points1)} 个点")
    print(f"第二帧点云: {len(points2)} 个点")
    
    # 对第二帧点云应用偏移量（还原）
    print(f"\n应用偏移量 ({dx:.3f}, {dy:.3f}, {dz:.3f}) 还原第二帧点云...")
    points2_restored = points2.copy()
    points2_restored[:, 0] -= dx  # x坐标
    points2_restored[:, 1] -= dy  # y坐标
    points2_restored[:, 2] -= dz  # z坐标
    
    # 合并点云
    print("合并点云...")
    merged_points = np.vstack([points1, points2_restored])
    print(f"合并后点云: {len(merged_points)} 个点")
    
    # 保存结果
    save_pcd_file(merged_points, output_pcd)
    save_bin_file(merged_points, output_bin)
    
    print("\n完成!")

if __name__ == "__main__":
    main()

