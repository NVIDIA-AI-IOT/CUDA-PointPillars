#include "detect/process.hpp"
#include "detect/draw_meshlab.hpp"
#include "detect/dbscan.hpp"
#include <algorithm>

namespace detect {

// 判断点是否在长方体内部 
bool point_in_3d_box(float px, float py, float pz, const RangeConfig& cfg)
{
    float cx = cfg.center_x;
    float cy = cfg.center_y;
    float cz = cfg.center_z;
    float l = cfg.range_x+cfg.range_x;
    float w = cfg.range_y+cfg.range_y;
    float h = cfg.range_z;

    // 计算旋转矩阵 cos/sin
    float cos_r = std::cos(-cfg.ry); // 逆旋转
    float sin_r = std::sin(-cfg.ry);

    // 将点转换到 ROI 框局部坐标系（绕Z轴逆旋转）
    float dx = px - cx;
    float dy = py - cy;
    float dz = pz - cz;

    float local_x = cos_r * dx - sin_r * dy;
    float local_y = sin_r * dx + cos_r * dy;
    float local_z = dz; // Z方向不动

    // 判断是否在轴对齐框内
    if (local_x < -l*0.5f || local_x > l*0.5f)
        return false;
    if (local_y < -w*0.5f || local_y > w*0.5f)
        return false;
    if (local_z < -h*0.5f || local_z > h*0.5f)
        return false;

    return true;
}

// 求地面方程参数
// Returns plane parameters [a, b, c, d] for equation ax + by + cz + d = 0
std::vector<float> segment_plane_ransac(const std::vector<float>& points, float distance_threshold = 0.05f, int ransac_n = 3,  int num_iterations = 10000) 
{
    std::vector<float> plane_model(4, 0.0f);

    if (points.size() < 12) {  // Need at least 4 points (3 coordinates each)
    return plane_model;
    }

    int num_points = points.size() / 4;  // Assuming x, y, z, intensity format

    // Random seed
    std::srand(std::time(nullptr));

    int best_inliers = 0;
    float best_a = 0, best_b = 0, best_c = 0, best_d = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
    // Randomly select 3 points
    int idx1 = std::rand() % num_points;
    int idx2 = std::rand() % num_points;
    int idx3 = std::rand() % num_points;

    // Avoid selecting same point
    while (idx2 == idx1) idx2 = std::rand() % num_points;
    while (idx3 == idx1 || idx3 == idx2) idx3 = std::rand() % num_points;

    // Get points
    float* p1 = (float*)points.data() + idx1 * 4;
    float* p2 = (float*)points.data() + idx2 * 4;
    float* p3 = (float*)points.data() + idx3 * 4;

    // Compute plane equation ax + by + cz + d = 0
    // Vectors from p1 to p2 and p1 to p3
    float v1x = p2[0] - p1[0];
    float v1y = p2[1] - p1[1];
    float v1z = p2[2] - p1[2];

    float v2x = p3[0] - p1[0];
    float v2y = p3[1] - p1[1];
    float v2z = p3[2] - p1[2];

    // Normal vector n = v1 × v2
    float nx = v1y * v2z - v1z * v2y;
    float ny = v1z * v2x - v1x * v2z;
    float nz = v1x * v2y - v1y * v2x;

    // Normalize normal vector
    float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-6f) continue;  // Degenerate case

    nx /= norm;
    ny /= norm;
    nz /= norm;

    // Compute d = -(nx*p1x + ny*p1y + nz*p1z)
    float d = -(nx * p1[0] + ny * p1[1] + nz * p1[2]);

    // Count inliers
    int inliers = 0;
    for (int i = 0; i < num_points; ++i) {
    float* p = (float*)points.data() + i * 4;
    float dist = std::abs(nx * p[0] + ny * p[1] + nz * p[2] + d);
    if (dist <= distance_threshold) {
    inliers++;
    }
    }

        // Update best model
        if (inliers > best_inliers) {
        best_inliers = inliers;
        best_a = nx;
        best_b = ny;
        best_c = nz;
        best_d = d;
        }
    }

    // Ensure normal vector points upward (b > 0)
    if (best_b < 0) {
    best_a = -best_a;
    best_b = -best_b;
    best_c = -best_c;
    best_d = -best_d;
    }

    plane_model[0] = best_a;
    plane_model[1] = best_b;
    plane_model[2] = best_c;
    plane_model[3] = best_d;

    std::cout << "Ground plane: a=" << best_a << ", b=" << best_b 
    << ", c=" << best_c << ", d=" << best_d 
    << " (inliers: " << best_inliers << "/" << num_points << ")" << std::endl;

    return plane_model;
}

inline float intensity_to_kitti(float raw_i) {
    // 限制范围
    if (raw_i < 0) raw_i = 0.0;
    if (raw_i > 255) raw_i = 255.0;

    // KITTI 风格：log 非线性压缩（拟合 Velodyne 旧驱动）
    float kitti_i = std::log1p(raw_i) / std::log1p(255.0f);

    // 限幅到 0~1
    if (kitti_i < 0) kitti_i = 0;
    if (kitti_i > 1) kitti_i = 1;

    return kitti_i;
}

inline float kitti_to_intensity(float kitti_i) {
    // 确保输入在有效范围内
    if (kitti_i < 0) kitti_i = 0.0;
    if (kitti_i > 1) kitti_i = 1.0;

    // 反向计算：kitti_i = log1p(raw_i) / log1p(255) 
    // 所以：log1p(raw_i) = kitti_i * log1p(255)
    // 因此：raw_i = expm1(kitti_i * log1p(255))
    float raw_i = std::expm1(kitti_i * std::log1p(255.0f));
    
    return std::move(raw_i);
}

// 预处理：过滤点云（去除地面点，保留ROI区域内的点）
void pre_processing(std::vector<float> &src, std::vector<float> &points_filtered, const RangeConfig& range_config)
{
    const auto& cfg = get_config();
    double a = cfg.ground_plane_a;
    double b = cfg.ground_plane_b;
    double c = cfg.ground_plane_c;
    double d = cfg.ground_plane_d;
    
    // Normalize denominator for distance calculation
    double denom =1.0/ std::sqrt(a*a + b*b + c*c);
    
    // Filter points based on distance to ground plane
    float ground_distance_threshold = cfg.ground_distance_threshold;
    for(int i = 0; i < src.size(); i += 4)
    {

        double distance = std::fabs(a * src[i] + b * src[i+1] + c * src[i+2] + d) * denom;
        if(distance > ground_distance_threshold  && point_in_3d_box(src[i], src[i+1], src[i+2], range_config) )
        {
            points_filtered.push_back(src[i]);
            points_filtered.push_back(src[i+1]);
            points_filtered.push_back(src[i+2]);
            points_filtered.push_back(intensity_to_kitti(src[i+3]));
        }
        else 
        {   
            src[i+3]=-3.0f;  //非检测区域点  -3.0f
        }
    }
    
    // 对 points_filtered 按 z 值从大到小快速排序
    int num_points = points_filtered.size() / 4;
    if (num_points > 0) {
        std::vector<int> indices(num_points);
        for (int i = 0; i < num_points; ++i) {
            indices[i] = i;
        }
        // 按 z 值从大到小排序索引
        std::sort(indices.begin(), indices.end(), [&points_filtered](int a, int b) {
            return points_filtered[a * 4 + 2] > points_filtered[b * 4 + 2];  // 按 z 值降序
        });
        // 重新排列 points_filtered
        std::vector<float> temp = points_filtered;
        for (int i = 0; i < num_points; ++i) {
            int orig_idx = indices[i];
            points_filtered[i * 4 + 0] = temp[orig_idx * 4 + 0];
            points_filtered[i * 4 + 1] = temp[orig_idx * 4 + 1];
            points_filtered[i * 4 + 2] = temp[orig_idx * 4 + 2];
            points_filtered[i * 4 + 3] = temp[orig_idx * 4 + 3];
        }
    }
    
    // 求地
    // segment_plane_ransac(points_filtered, 0.05f, 3,100000);
}

// 计算绕Y轴旋转矩阵（相机坐标系）
void rot_y_matrix(float angle, float R[3][3])
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    R[0][0] = c;  R[0][1] = 0;  R[0][2] = s;
    R[1][0] = 0;  R[1][1] = 1;  R[1][2] = 0;
    R[2][0] = -s; R[2][1] = 0;  R[2][2] = c;
}

// 计算绕Z轴旋转矩阵（雷达坐标系，在XY平面内旋转）
void rot_z_matrix(float angle, float R[3][3])
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    R[0][0] = c;  R[0][1] = -s; R[0][2] = 0;
    R[1][0] = s;  R[1][1] = c;  R[1][2] = 0;
    R[2][0] = 0;  R[2][1] = 0;  R[2][2] = 1;
}

// 计算最小外接贴地长方体（雷达坐标系，绕Z轴旋转）
bool compute_min_box_with_rot(const std::vector<nvtype::Float3> &points_in_box, 
                               float rot_y,
                               float &height, float &width, float &length,
                               float &pos_x, float &pos_y, float &pos_z)
{
    if (points_in_box.empty()) {
        return false;
    }
    
    // 点云转局部坐标系 (对齐 rot_y，雷达坐标系绕Z轴旋转)
    float R_world2local[3][3];
    rot_z_matrix(-rot_y, R_world2local);  // 使用绕Z轴旋转，而不是绕Y轴
    
    // 计算第一个点在地面上的z坐标（全局坐标系，地面方程：ax + by + cz + d = 0）
    // 在雷达坐标系中，Z是高度方向，求解z：z = -(ax + by + d) / c
    const auto& cfg = get_config();
    double a = cfg.ground_plane_a;
    double b = cfg.ground_plane_b;
    double c = cfg.ground_plane_c;
    double d = cfg.ground_plane_d;
    double z_ground_global = -(a * points_in_box[0].x + b * points_in_box[0].y + d) / c;
    
    // 将地面点转换到局部坐标系
    float ground_pt_local_z = R_world2local[2][0] * points_in_box[0].x + 
                              R_world2local[2][1] * points_in_box[0].y + 
                              R_world2local[2][2] * z_ground_global;
    
    // 转换所有点到局部坐标系
    std::vector<nvtype::Float3> pts_local;
    for (const auto &pt : points_in_box) {
        float local_x = R_world2local[0][0] * pt.x + R_world2local[0][1] * pt.y + R_world2local[0][2] * pt.z;
        float local_y = R_world2local[1][0] * pt.x + R_world2local[1][1] * pt.y + R_world2local[1][2] * pt.z;
        float local_z = R_world2local[2][0] * pt.x + R_world2local[2][1] * pt.y + R_world2local[2][2] * pt.z;
        pts_local.push_back(nvtype::Float3(local_x, local_y, local_z));
    }
    
    // 计算局部坐标系下的最小最大值
    float x_min = pts_local[0].x, x_max = pts_local[0].x;
    float y_min = pts_local[0].y, y_max = pts_local[0].y;
    float z_min = ground_pt_local_z, z_max = pts_local[0].z;  // z_min使用地面z值（贴地，局部坐标系）
    
    for (const auto &pt : pts_local) {
        if (pt.x < x_min) x_min = pt.x;
        if (pt.x > x_max) x_max = pt.x;
        if (pt.y < y_min) y_min = pt.y;
        if (pt.y > y_max) y_max = pt.y;
        if (pt.z > z_max) z_max = pt.z;  // z_min已经设置为地面值，只需要更新z_max
    }
    
    width = x_max - x_min;   // local_x 对应宽度
    length = y_max - y_min;  // local_y 对应长度
    height = z_max - z_min;  // local_z 对应高度（贴地）
    
    // 计算底部中心
    // x, y 使用几何中心（局部坐标系）
    float cx_local = (x_min + x_max) / 2.0f;
    float cy_local = (y_min + y_max) / 2.0f;
    float cz_local = (z_min + z_max) / 2.0f;  // 几何中心的z
    
    // 先计算框的几何中心在全局坐标系中的位置
    pos_x = R_world2local[0][0] * cx_local + R_world2local[1][0] * cy_local + R_world2local[2][0] * cz_local;
    pos_y = R_world2local[0][1] * cx_local + R_world2local[1][1] * cy_local + R_world2local[2][1] * cz_local;
    
    pos_z = -(a * pos_x + b * pos_y + d) / c;
    pos_z += height / 2.0f;
    
    return true;
}

// 校准3D框：获取最小外接贴地长方体
void calib_3d_box(const std::vector<float> &points_filtered, 
    detect::ProcessingBox &box)
{
    // 从 detect::ProcessingBox 构造 RangeConfig，复用 point_in_3d_box 函数
    RangeConfig cfg;
    cfg.center_x = box.x;
    cfg.center_y = box.y;
    cfg.center_z = box.z;
    // 注意：boxCorners 中 w 对应旋转后的 x 坐标，l 对应旋转后的 y 坐标
    // point_in_3d_box 中 range_x 用于 local_x 判断，range_y 用于 local_y 判断
     // 边长*1.2  扩大一点点  防止检测框小  
    cfg.range_x = box.w * 0.5f+std::min(box.w*0.04f, 0.5f);  // w 对应 x 方向（point_in_3d_box 中会乘以2）
    cfg.range_y = box.l * 0.5f+std::min(box.l*0.04f, 0.5f);  // l 对应 y 方向（point_in_3d_box 中会乘以2）
    cfg.range_z = box.h * 1.0f+std::min(box.h*0.2f, 0.5f);  // h 对应 z 方向（高度，不是半高）
    cfg.ry = box.rt;  // 旋转角度（都是绕Z轴旋转，在雷达坐标系XY平面内）
    // 筛选点云中属于ROI框的点
    std::vector<nvtype::Float3> points_in_box;
    int num_points = points_filtered.size() / 4;
    
    for (int i = 0; i < num_points; ++i) {
        float px = points_filtered[i * 4];
        float py = points_filtered[i * 4 + 1];
        float pz = points_filtered[i * 4 + 2];
        
        if (point_in_3d_box(px, py, pz, cfg)) {
            points_in_box.push_back(nvtype::Float3(px, py, pz));
            box.points.push_back(std::array<float, 4>{px, py, pz, kitti_to_intensity(points_filtered[i * 4 + 3])});
        }
    }
    
    // DBSCAN 聚类过滤（如果启用）
    const auto& cfg_dbscan = get_config();
    if (cfg_dbscan.use_dbscan && !points_in_box.empty()) {
        // 优化策略：只对前30%的点（z值最高的点）进行DBSCAN，其余70%直接保留
        // 由于 points_filtered 已经按 z 值从大到小排序，points_in_box 也是按 z 值从大到小的顺序
        int total_points = points_in_box.size();
        int dbscan_points_count = static_cast<int>(total_points * 0.3f);  // 前30%的点
        
        // 分割点云：前30%（高z值，进行DBSCAN）和后70%（低z值，直接保留）
        std::vector<nvtype::Float3> points_lower;  // 后70%，直接保留
        std::vector<nvtype::Float3> points_upper;  // 前30%，进行DBSCAN
        std::vector<std::array<float, 4>> box_points_lower;
        std::vector<std::array<float, 4>> box_points_upper;
        
        for (size_t i = 0; i < points_in_box.size(); ++i) {
            if (i < static_cast<size_t>(dbscan_points_count)) {
                // 前30%：需要DBSCAN处理
                points_upper.push_back(points_in_box[i]);
                box_points_upper.push_back(box.points[i]);
            } else {
                // 后70%：直接保留
                points_lower.push_back(points_in_box[i]);
                box_points_lower.push_back(box.points[i]);
            }
        }
        // 只对前30%进行DBSCAN
        if (!points_upper.empty()) {
            int num_points_upper = points_upper.size();
            const int cuda_threshold = 500;  // 点数 >= 500 时使用 CUDA 版本
            
            if (num_points_upper >= cuda_threshold) {
                // 使用 CUDA 加速版本（对于大量点云）
                dbscan_filter_cuda(points_upper, box_points_upper, cfg_dbscan.dbscan_eps_xy, cfg_dbscan.dbscan_eps_z,
                    cfg_dbscan.dbscan_max_cluster_ratio, cfg_dbscan.dbscan_z_threshold, nullptr);
            } else {
                // 使用 CPU 版本（对于少量点云）
                dbscan_filter(points_upper, box_points_upper, cfg_dbscan.dbscan_eps_xy, cfg_dbscan.dbscan_eps_z,
                    cfg_dbscan.dbscan_max_cluster_ratio, cfg_dbscan.dbscan_z_threshold);
            }
        }
        
        // 合并结果：下半部分（全部保留）+ 上半部分（DBSCAN过滤后）
        points_in_box.clear();
        box.points.clear();
        points_in_box.reserve(points_lower.size() + points_upper.size());
        box.points.reserve(box_points_lower.size() + box_points_upper.size());
        
        // 先添加下半部分
        points_in_box.insert(points_in_box.end(), points_lower.begin(), points_lower.end());
        box.points.insert(box.points.end(), box_points_lower.begin(), box_points_lower.end());
        
        // 再添加上半部分（DBSCAN过滤后）
        points_in_box.insert(points_in_box.end(), points_upper.begin(), points_upper.end());
        box.points.insert(box.points.end(), box_points_upper.begin(), box_points_upper.end());
    }
    
    if (points_in_box.empty()) {
        box.score=0;
        return;  // 如果没有点在框内，保持原box不变   
    }
    
    // 计算最小外接贴地长方体
    float height, width, length, pos_x, pos_y, pos_z;
    if (compute_min_box_with_rot(points_in_box, box.rt, height, width, length, pos_x, pos_y, pos_z)) {
        // 更新box的值
        box.h = height;
        box.w = width;
        box.l = length;
        box.x = pos_x;
        box.y = pos_y;
        box.z = pos_z;
    }
}

// 后处理：过滤和校准检测框
void post_processing(const std::vector<pointpillar::lidar::BoundingBox> &bboxes,
                     std::vector<detect::ProcessingBox> &bboxes_result,
                     const std::vector<float> &points_filtered)
{
    for (const auto &box : bboxes)
    {
        if(box.score > get_config().detection_score_threshold && box.id ==0  )
        {
            //获取最小外接贴地长方体
            detect::ProcessingBox calibrated_box = detect::ProcessingBox(box);
            calibrated_box.rt = std::fmod(calibrated_box.rt+M_PI, M_PI);
            if( calibrated_box.rt < M_PI/2.0f - M_PI/3.0f || calibrated_box.rt > M_PI/2.0f + M_PI/3.0f) //与车道方向接近垂直的框不进行处理
                continue;   

            
            // 动态角度限制：k_ 的单位是度，车越长 k_ 越小，最小为 0 度
            // k_ = (l/w) * 10度，当 l ≫ w 时 k_ 接近 0 度
            float k_ = (calibrated_box.l / calibrated_box.w) * 10.0f;  // k_ 的单位是度
            if (k_ < 0.0f) k_ = 0.0f;
            if (k_ > 10.0f) k_ = 10.0f;  // 最大限制为 10 度

            // 限制角度在 [π/2 - k_*π/180 , π/2 + k_*π/180]，k_ 的单位是度
            float k_rad = k_ * M_PI / 180.0f;  // 转换为弧度
            float upper = M_PI / 2.0f + k_rad;
            float lower = M_PI / 2.0f - k_rad;
            if(calibrated_box.rt > upper)
                calibrated_box.rt = upper;
            else if(calibrated_box.rt < lower)
                calibrated_box.rt = lower;
            calib_3d_box(points_filtered, calibrated_box);
            if(calibrated_box.score ==0 ||calibrated_box.points.size() < 10 || 
            calibrated_box.h / calibrated_box.l>3 || calibrated_box.h / calibrated_box.w>3)
                continue;
            else if(calibrated_box.h <0.5 || calibrated_box.w<0.5 || calibrated_box.l<0.5)
                continue;

            // 用单个框来画图，直接将框追加到calibrated_box.points后面（高效，无需格式转换）
            detect::SaveBoxes({calibrated_box}, calibrated_box.points, get_config().point_cloud_draw_step);
            bboxes_result.push_back(calibrated_box);
        }
    }
}
} // namespace detect
