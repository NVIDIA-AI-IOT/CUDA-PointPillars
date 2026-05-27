#include "tracking.hpp"
#include "config.hpp"
#include "pointpillar.hpp"
#include "draw_meshlab.hpp"
#include <iostream>
#include <set>

namespace tracking {

// ==================== KalmanFilter3D Implementation ====================

KalmanFilter3D::KalmanFilter3D() {
    // Initialize state
    for (int i = 0; i < 6; ++i) {
        x_[i] = 0.0f;
    }
    
    // Initialize covariance P
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_[i][j] = (i == j) ? 10.0f : 0.0f;
        }
    }
    
    // Initialize state transition matrix F (identity)
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            F_[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize process noise Q
    const float q_pos = 1e-3f;
    const float q_vel = 1e-2f;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (i < 3 && j < 3) {
                Q_[i][j] = (i == j) ? q_pos : 0.0f;
            } else if (i >= 3 && j >= 3) {
                Q_[i][j] = (i == j) ? q_vel : 0.0f;
            } else {
                Q_[i][j] = 0.0f;
            }
        }
    }
    
    // Initialize measurement noise R
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_[i][j] = (i == j) ? 0.25f : 0.0f;
        }
    }
}

void KalmanFilter3D::init(const std::array<float, 3>& x0) {
    x_[0] = x0[0];
    x_[1] = x0[1];
    x_[2] = x0[2];
    x_[3] = 0.0f; // vx
    x_[4] = 0.0f; // vy
    x_[5] = 0.0f; // vz
}

std::array<float, 3> KalmanFilter3D::predict(float dt) {
    // Update F matrix with dt
    for (int i = 0; i < 3; ++i) {
        F_[i][i + 3] = dt;
    }
    
    // Predict state: x = F * x
    std::array<float, 6> x_new;
    for (int i = 0; i < 6; ++i) {
        x_new[i] = 0.0f;
        for (int j = 0; j < 6; ++j) {
            x_new[i] += F_[i][j] * x_[j];
        }
    }
    
    // Predict covariance: P = F * P * F^T + Q
    float P_temp[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_temp[i][j] = 0.0f;
            for (int k = 0; k < 6; ++k) {
                P_temp[i][j] += F_[i][k] * P_[k][j];
            }
        }
    }
    
    float P_new[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_new[i][j] = Q_[i][j];
            for (int k = 0; k < 6; ++k) {
                P_new[i][j] += P_temp[i][k] * F_[j][k];
            }
        }
    }
    
    x_ = x_new;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_[i][j] = P_new[i][j];
        }
    }
    
    return {x_[0], x_[1], x_[2]};
}

void KalmanFilter3D::update(const std::array<float, 3>& z) {
    // Observation matrix H (3x6) - only observe position
    float H[3][6] = {0};
    H[0][0] = H[1][1] = H[2][2] = 1.0f;
    
    // Innovation: y = z - H * x
    std::array<float, 3> y;
    for (int i = 0; i < 3; ++i) {
        y[i] = z[i] - x_[i];
    }
    
    // Innovation covariance: S = H * P * H^T + R
    float S[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S[i][j] = R_[i][j];
            for (int k = 0; k < 6; ++k) {
                S[i][j] += H[i][k] * P_[k][j];
            }
        }
    }
    
    // Kalman gain: K = P * H^T * S^-1
    float K[6][3];
    float S_inv[3][3];
    // Simple 3x3 inverse
    float det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
              - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
              + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
    if (std::abs(det) < 1e-6f) {
        // Singular matrix, skip update
        return;
    }
    S_inv[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) / det;
    S_inv[0][1] = -(S[0][1] * S[2][2] - S[0][2] * S[2][1]) / det;
    S_inv[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) / det;
    S_inv[1][0] = -(S[1][0] * S[2][2] - S[1][2] * S[2][0]) / det;
    S_inv[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) / det;
    S_inv[1][2] = -(S[0][0] * S[1][2] - S[0][2] * S[1][0]) / det;
    S_inv[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) / det;
    S_inv[2][1] = -(S[0][0] * S[2][1] - S[0][1] * S[2][0]) / det;
    S_inv[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) / det;
    
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            K[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                K[i][j] += P_[i][k] * H[j][k] * S_inv[k][j];
            }
        }
    }
    
    // Update state: x = x + K * y
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            x_[i] += K[i][j] * y[j];
        }
    }
    
    // Update covariance: P = (I - K * H) * P
    float I_KH[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            I_KH[i][j] = (i == j) ? 1.0f : 0.0f;
            for (int k = 0; k < 3; ++k) {
                I_KH[i][j] -= K[i][k] * H[k][j];
            }
        }
    }
    
    float P_new[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_new[i][j] = 0.0f;
            for (int k = 0; k < 6; ++k) {
                P_new[i][j] += I_KH[i][k] * P_[k][j];
            }
        }
    }
    
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            P_[i][j] = P_new[i][j];
        }
    }
}

std::array<float, 3> KalmanFilter3D::get_state() const {
    return {x_[0], x_[1], x_[2]};
}

// ==================== Tracker Implementation ====================

Tracker::Tracker(const BBox3D& bbox3d, int id, uint64_t timestamp)
    : id_(id), time_since_update_(0), hits_(1),
      score_(bbox3d.score()), last_timestamp_(timestamp),
      yaw_(bbox3d.yaw()), line_(bbox3d.line()) {
    
    observations_.push_back(bbox3d);
    
    std::array<float, 3> x0 = {bbox3d.x(), bbox3d.y(), bbox3d.z()};
    kf_.init(x0);
}

BBox3D Tracker::predict(float dt) {
    std::array<float, 3> pos = kf_.predict(dt);
    time_since_update_++;
    // Use last observation's dimensions for prediction
    float l = 0.0f, w = 0.0f, h = 0.0f;
    if (!observations_.empty()) {
        const auto& last_obs = observations_.back();
        l = last_obs.l();
        w = last_obs.w();
        h = last_obs.h();
    }
    return BBox3D(pos[0], pos[1], pos[2], l, w, h, yaw_, line_, score_);
}

void Tracker::update(const BBox3D& bbox3d, uint64_t timestamp, std::vector<std::array<float, 4>> &points_max_car) {
    BBox3D previous_observation;
    if (!observations_.empty()) {
        previous_observation = observations_.back();
    }

    // KF update position
    std::array<float, 3> z = {bbox3d.x(), bbox3d.y(), bbox3d.z()};
    kf_.update(z);
    
    // Push back new observation instead of taking max
    observations_.push_back(bbox3d);
    yaw_ = bbox3d.yaw();
    line_ = bbox3d.line();
    // Update score to maximum
    score_ = std::max(score_, bbox3d.score());
    
    // Update points_max_car_ if new one is larger
    if (points_max_car.size() > points_max_car_.size()) {
        points_max_car_.clear();
        points_max_car_ = std::move(points_max_car);
    }
    
    // Update speed
    float dt = (timestamp - last_timestamp_) / 1000.0f; // ms -> s
    if (dt > 1e-6f && !observations_.empty() && observations_.size() > 1) {
        float dx = bbox3d.x() - previous_observation.x();
        float dy = bbox3d.y() - previous_observation.y();
        float dz = bbox3d.z() - previous_observation.z();
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        speed_.push_back(dist / dt);
        // Store position for speed calculation (use current observation position)
        speed_positions_.push_back({bbox3d.x(), bbox3d.y(), bbox3d.z()});
    }
    
    last_timestamp_ = timestamp;
    time_since_update_ = 0;
    hits_++;
}

BBox3D Tracker::get_state() const {
    std::array<float, 3> pos = kf_.get_state();
    // Use last observation's dimensions
    float l = 0.0f, w = 0.0f, h = 0.0f;
    if (!observations_.empty()) {
        const auto& last_obs = observations_.back();
        l = last_obs.l();
        w = last_obs.w();
        h = last_obs.h();
    }
    return BBox3D(pos[0], pos[1], pos[2], l, w, h, yaw_, line_, score_);
}

BBox3D Tracker::get_last_observation() const {
    if (observations_.empty()) {
        return BBox3D();
    }
    return observations_.back();
}

float Tracker::get_speed(SpeedStrategy strategy) const {
    float speed_mps = get_speed_value(speed_, strategy);  // 速度单位：m/s
    // 应用车速校准值（单位：km/h），转换为 m/s 后加上
    float calibration_mps = get_config().speed_calib / 3.6f;  // km/h 转 m/s
    return speed_mps + calibration_mps;
}

float Tracker::get_length(DimensionStrategy strategy) const {
    if (observations_.empty()) {
        return 0.0f;
    }
    std::vector<float> lengths;
    lengths.reserve(observations_.size());
    for (const auto& obs : observations_) {
        lengths.push_back(obs.l());
    }
    return get_dimension_value(lengths, strategy);
}

float Tracker::get_width(DimensionStrategy strategy) const {
    if (observations_.empty()) {
        return 0.0f;
    }
    std::vector<float> widths;
    widths.reserve(observations_.size());
    for (const auto& obs : observations_) {
        widths.push_back(obs.w());
    }
    return get_dimension_value(widths, strategy);
}

float Tracker::get_height(DimensionStrategy strategy) const {
    if (observations_.empty()) {
        return 0.0f;
    }
    std::vector<float> heights;
    heights.reserve(observations_.size());
    for (const auto& obs : observations_) {
        heights.push_back(obs.h());
    }
    return get_dimension_value(heights, strategy);
}

float Tracker::get_dimension_value(const std::vector<float>& values, DimensionStrategy strategy) const {
    if (values.empty()) {
        return 0.0f;
    }
    
    // If values count is less than 5, return average
    if (values.size() < 5) {
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        return sum / static_cast<float>(values.size());
    }
    
    switch (strategy) {
        case DimensionStrategy::MAX_VALUE: {
            // 1. 取最大值
            return *std::max_element(values.begin(), values.end());
        }
        
        case DimensionStrategy::AVERAGE: {
            // 2. 取平均值
            float sum = 0.0f;
            for (float v : values) {
                sum += v;
            }
            return sum / static_cast<float>(values.size());
        }
        
        case DimensionStrategy::TRIMMED_AVERAGE: {
            // 3. 去除最大的20%和最小的20%后，再求平均值
            if (values.size() < 3) {
                float sum = 0.0f;
                for (float v : values) {
                    sum += v;
                }
                return sum / static_cast<float>(values.size());
            }
            
            std::vector<float> sorted_values = values;
            std::sort(sorted_values.begin(), sorted_values.end());
            
            size_t trim_count = static_cast<size_t>(sorted_values.size() * 0.2f);
            size_t start_idx = trim_count;  // 去除最小的20%
            size_t end_idx = sorted_values.size() - trim_count;  // 去除最大的20%
            
            if (start_idx >= end_idx) {
                start_idx = 0;
                end_idx = sorted_values.size();
            }
            
            float sum = 0.0f;
            for (size_t i = start_idx; i < end_idx; ++i) {
                sum += sorted_values[i];
            }
            return sum / static_cast<float>(end_idx - start_idx);
        }
        
        case DimensionStrategy::TRIMMED_MAX: {
            // 4. 去除最大的前5%后，再取最大值
            if (values.size() < 2) {
                return *std::max_element(values.begin(), values.end());
            }
            
            std::vector<float> sorted_values = values;
            std::sort(sorted_values.begin(), sorted_values.end());
            
            size_t trim_count = static_cast<size_t>(values.size() * 0.05f);
            size_t end_idx = sorted_values.size() - trim_count;
            
            if (end_idx == 0 || end_idx > sorted_values.size()) {
                end_idx = sorted_values.size();
            }
            
            return *std::max_element(sorted_values.begin(), sorted_values.begin() + end_idx);
        }
        
        case DimensionStrategy::FIRST_10_PERCENT_AVG: {
            // 5. 最大的前10%的平均值
            if (values.empty()) {
                return 0.0f;
            }
            
            std::vector<float> sorted_values = values;
            std::sort(sorted_values.begin(), sorted_values.end());
            
            size_t count = static_cast<size_t>(values.size() * 0.1f);
            if (count == 0) count = 1;
            if (count > sorted_values.size()) count = sorted_values.size();
            
            size_t start_idx = sorted_values.size() - count;
            float sum = 0.0f;
            for (size_t i = start_idx; i < sorted_values.size(); ++i) {
                sum += sorted_values[i];
            }
            return sum / static_cast<float>(count);
        }
        
        case DimensionStrategy::MEDIAN: {
            // 6. 取最大20%的中位数
            if (values.empty()) {
                return 0.0f;
            }
            
            std::vector<float> sorted_values = values;
            std::sort(sorted_values.begin(), sorted_values.end());
            
            // 取最大的20%的值
            size_t n = sorted_values.size();
            size_t count = static_cast<size_t>(n * 0.20f);
            if (count == 0) count = 1;
            if (count > n) count = n;
            
            size_t start_idx = n - count;  // 从后往前取20%
            
            // 对这20%的值取中位数
            size_t subset_size = count;
            if (subset_size % 2 == 0) {
                // 偶数个元素：取中间两个数的平均值
                size_t mid1 = start_idx + subset_size / 2 - 1;
                size_t mid2 = start_idx + subset_size / 2;
                return (sorted_values[mid1] + sorted_values[mid2]) / 2.0f;
            } else {
                // 奇数个元素：取中间值
                size_t mid = start_idx + subset_size / 2;
                return sorted_values[mid];
            }
        }
        
        default:
            return *std::max_element(values.begin(), values.end());
    }
}

// Forward declaration for point_to_line_distance (used in get_speed_value)
static float point_to_line_distance(float px, float py, float pz,
                                    float lx1, float ly1, float lz1,
                                    float lx2, float ly2, float lz2);

float Tracker::get_speed_value(const std::vector<float>& speeds, SpeedStrategy strategy) const {
    if (speeds.empty()) {
        return 0.0f;
    }
    
    switch (strategy) {
        case SpeedStrategy::FIRST_80_PERCENT_AVG: {
            // 1. 前80%的平均值（过滤超过200km/h的值）
            size_t count = speeds.size();
            size_t use_count = count;
            
            if (count >= 5) {  // 只用前80%求车速  数量太少就算了
                use_count = static_cast<size_t>(count * 0.8f);
            }
            
            // Calculate average speed from the first use_count values
            // Filter out speeds greater than 200km/h
            const float max_speed = 200.0f / 3.6f;
            float sum = 0.0f;
            size_t valid_count = 0;
            for (size_t i = 0; i < use_count; ++i) {
                if (speeds[i] <= max_speed) {
                    sum += speeds[i];
                    valid_count++;
                }
            }
            
            if (valid_count == 0) {
                return 0.0f;
            }
            
            return sum / static_cast<float>(valid_count);
        }
        
        case SpeedStrategy::CLOSEST_3_TO_SPEED_LINE: {
            // 2. 距离speed_line最近的前3个点的速度的最大值（过滤超过200km/h的值）
            if (speed_positions_.empty() || speeds.size() != speed_positions_.size()) {
                // 如果没有位置信息，回退到前80%平均值策略
                return get_speed_value(speeds, SpeedStrategy::FIRST_80_PERCENT_AVG);
            }
            
            // Get speed_line configuration
            const auto& speed_line_config = get_config().speed_line_config;
            float line_start_x = speed_line_config.start_x;
            float line_start_y = speed_line_config.start_y;
            float line_start_z = speed_line_config.start_z;
            float line_end_x = speed_line_config.end_x;
            float line_end_y = speed_line_config.end_y;
            float line_end_z = speed_line_config.end_z;
            
            // Calculate distance from each point to speed_line
            struct SpeedWithDistance {
                float speed;
                float distance;
            };
            
            std::vector<SpeedWithDistance> speed_distances;
            speed_distances.reserve(speeds.size());
            
            const float max_speed = 200.0f / 3.6f;
            for (size_t i = 0; i < speeds.size(); ++i) {
                // Filter out speeds greater than 200km/h
                if (speeds[i] > max_speed) {
                    continue;
                }
                
                // Calculate distance from point to speed_line
                float distance = point_to_line_distance(
                    speed_positions_[i][0], speed_positions_[i][1], speed_positions_[i][2],
                    line_start_x, line_start_y, line_start_z,
                    line_end_x, line_end_y, line_end_z
                );
                
                speed_distances.push_back({speeds[i], distance});
            }
            
            if (speed_distances.empty()) {
                return 0.0f;
            }
            
            // Sort by distance (ascending)
            std::sort(speed_distances.begin(), speed_distances.end(),
                     [](const SpeedWithDistance& a, const SpeedWithDistance& b) {
                         return a.distance < b.distance;
                     });
            
            // Take the closest 3 points (or all if less than 3) and find the maximum speed
            size_t count = std::min(static_cast<size_t>(3), speed_distances.size());
            float max_speed_value = speed_distances[0].speed;
            for (size_t i = 1; i < count; ++i) {
                if (speed_distances[i].speed > max_speed_value) {
                    max_speed_value = speed_distances[i].speed;
                }
            }
            
            return max_speed_value;
        }
        
        default:
            // 默认使用前80%平均值策略
            return get_speed_value(speeds, SpeedStrategy::FIRST_80_PERCENT_AVG);
    }
}

// ==================== MultiObjectTracker Implementation ====================

MultiObjectTracker::MultiObjectTracker(float iou_threshold, int max_age,
                                       DimensionStrategy dimension_strategy,
                                       SpeedStrategy speed_strategy)
    : next_id_(0), iou_threshold_(iou_threshold), max_age_(max_age),
      dimension_strategy_(dimension_strategy), speed_strategy_(speed_strategy) {
}

void MultiObjectTracker::update(std::vector<BBox3D>& detections, 
    std::vector<std::vector<std::array<float, 4>>> &car_points_frame, 
    uint64_t timestamp, std::vector<float> &points) {

    std::lock_guard<std::mutex> lock(trackers_mutex_);
    points_now = points;
    
    // 删除离 line2 太近的box
    filter_detections_by_line2_distance(detections, car_points_frame);
    
    Boxes_now = detections;
    // Step 1: Predict
    for (auto& trk : trackers_) {
        float dt = std::max(1e-3f, (float)(timestamp - trk.get_last_timestamp()) / 1000.0f);
        trk.predict(dt);
    }
    
    // Step 2: Build cost matrix
    int n_trk = trackers_.size();
    int n_det = detections.size();

    // 创建新追踪器
    if (n_trk == 0) {
        for (int j = 0; j < n_det; ++j) {
            trackers_.emplace_back(detections[j], next_id_++, timestamp);
            trackers_id_.emplace_back("");  // Default empty string
        }
        return;
    }
    
    std::vector<std::vector<float>> cost(n_trk, std::vector<float>(n_det));
    for (int i = 0; i < n_trk; ++i) {
        for (int j = 0; j < n_det; ++j) {
            // Convert IOU to cost: IOU ranges from 0 to 1 (higher is better)
            // Cost = 1 - IOU (lower is better)
            float iou = bbox3d_iou(trackers_[i].get_state(), detections[j]);
            cost[i][j] = 1.0f - iou;
        }
    }
    
    // Step 3: Match
    Assignment assignment = linear_sum_assignment(cost);
    
    // Step 4: 更新匹配的追踪器
    for (const auto& match : assignment.matched) {
        int i = match.first;
        int j = match.second;
        trackers_[i].update(detections[j], timestamp, car_points_frame[j]);
        // 只更新追踪器  暂时不用更新result_map_
        // if (!trackers_id_[i].empty()) {
        //     const std::string& unique_id = trackers_id_[i];
        //     const auto& trk = trackers_[i];
        //     BBox3D val = trk.get_last_observation();  // 使用观测值，因为已经匹配上了
        //     BestResult best_result;
        //     best_result.length = trk.get_length(dimension_strategy_);
        //     best_result.width = trk.get_width(dimension_strategy_);
        //     best_result.height = trk.get_height(dimension_strategy_);
        //     best_result.centre_l = val.x();
        //     best_result.centre_w = val.y();
        //     best_result.centre_h = val.z();
        //     best_result.speed = trk.get_speed();
        //     best_result.score = trk.get_score();
        //     best_result.points_max_car = trk.get_points_max_car();
        //     update_result_entry(unique_id, best_result, 0, timestamp);
        // }
    }
    
    // 没匹配的创建新追踪器
    for (int j : assignment.unmatched_det) {
        trackers_.emplace_back(detections[j], next_id_++, timestamp);
        trackers_id_.emplace_back("");
        // 为新创建的追踪器传入car_points_frame
        if (j < car_points_frame.size()) {
            trackers_.back().update(detections[j], timestamp, car_points_frame[j]);
        }
    }
    
    // Step 6: Remove old trackers
    // First, collect indices to remove
    std::vector<size_t> indices_to_remove;
    for (size_t i = 0; i < trackers_.size(); ++i) {

        if (trackers_[i].get_time_since_update() > max_age_) {
            indices_to_remove.push_back(i);
        }
        else if(!trackers_id_[i].empty())
        {
            //如果状态为2  表示等待超时了也需要删除
            // 使用 find() 避免自动创建不存在的键
            auto it = result_map_.find(trackers_id_[i]);
            if(it != result_map_.end() && it->second.status_code == 2)
            {
                indices_to_remove.push_back(i);
            }
        }
    }
    
    // Remove from both vectors (from back to front to preserve indices)
    for (auto it = indices_to_remove.rbegin(); it != indices_to_remove.rend(); ++it) {
        size_t idx = *it;
        const std::string& unique_id = trackers_id_[idx];
        if ( !unique_id.empty()) {
            // 使用 find() 避免自动创建不存在的键
            auto result_it = result_map_.find(unique_id);
            if(result_it != result_map_.end() && result_it->second.status_code == 2)  //  追踪超时的  比如车不动了 需要获取最后一次观测值
            {
                //不更新了 直接删
                result_map_.erase(unique_id);
                trackers_.erase(trackers_.begin() + idx);
                trackers_id_.erase(trackers_id_.begin() + idx);
            }
            else //追踪不到 但有ID  下次再删除
            {
                // 在删除之前，如果有unique_id，更新result_map_
                const auto& trk = trackers_[idx];
                BBox3D val = trk.get_state();  // 使用预测后的状态 
                BestResult best_result;
                best_result.length = trk.get_length(dimension_strategy_);
                best_result.width = trk.get_width(dimension_strategy_);
                best_result.height = trk.get_height(dimension_strategy_);
                best_result.centre_l = val.x();
                best_result.centre_w = val.y();
                best_result.centre_h = val.z();
                best_result.speed = trk.get_speed(speed_strategy_);
                best_result.score = trk.get_score();
                best_result.points_max_car.clear();
                best_result.points_max_car = std::move(trk.points_max_car_);
                update_result_entry(unique_id, best_result, 1, timestamp);  // 1 表示可以获取了
            }
        }
        else {  // 没有ID的直接删
            trackers_.erase(trackers_.begin() + idx);
            trackers_id_.erase(trackers_id_.begin() + idx);
        }
    }
}

// Forward declaration
static float point_to_line_distance(float px, float py, float pz,
                                    float lx1, float ly1, float lz1,
                                    float lx2, float ly2, float lz2);

// Filter out detections that are too close to line2
void MultiObjectTracker::filter_detections_by_line2_distance(
    std::vector<BBox3D>& detections, 
    std::vector<std::vector<std::array<float, 4>>>& car_points_frame) const {
    
    float min_distance = 4;
    const auto& line2_config = get_config().line2_config;
    float line2_start_x = line2_config.start_x;
    float line2_start_y = line2_config.start_y;
    float line2_start_z = line2_config.start_z;
    float line2_end_x = line2_config.end_x;
    float line2_end_y = line2_config.end_y;
    float line2_end_z = line2_config.end_z;
    
    // Calculate distances for all detections
    std::vector<float> distances;
    distances.reserve(detections.size());
    for (size_t j = 0; j < detections.size(); ++j) {
        const auto& det = detections[j];
        float distance = point_to_line_distance(
            det.y(), det.x(), det.z(),  // 转为雷达坐标系
            line2_start_x, line2_start_y, line2_start_z,
            line2_end_x, line2_end_y, line2_end_z
        );
        distances.push_back(distance);
    }
    
    // Use erase-remove idiom: collect indices to keep, then rebuild vectors
    // This is more efficient than erasing from middle of vector
    std::vector<size_t> indices_to_keep;
    indices_to_keep.reserve(detections.size());
    for (size_t j = 0; j < detections.size(); ++j) {
        if (distances[j] >= min_distance) {
            indices_to_keep.push_back(j);
        }
    }
    
    // Rebuild detections and car_points_frame with only kept indices
    if (indices_to_keep.size() < detections.size()) {
        std::vector<BBox3D> new_detections;
        std::vector<std::vector<std::array<float, 4>>> new_car_points_frame;
        new_detections.reserve(indices_to_keep.size());
        new_car_points_frame.reserve(indices_to_keep.size());
        
        for (size_t idx : indices_to_keep) {
            new_detections.push_back(detections[idx]);
            if (idx < car_points_frame.size()) {
                new_car_points_frame.push_back(car_points_frame[idx]);
            }
        }
        
        detections = std::move(new_detections);
        car_points_frame = std::move(new_car_points_frame);
    }
}

// Calculate distance from point to line (3D)
static float point_to_line_distance(float px, float py, float pz,
                                    float lx1, float ly1, float lz1,
                                    float lx2, float ly2, float lz2) {
    // Line direction vector
    float dx = lx2 - lx1;
    float dy = ly2 - ly1;
    float dz = lz2 - lz1;
    
    // Vector from line start to point
    float px_vec = px - lx1;
    float py_vec = py - ly1;
    float pz_vec = pz - lz1;
    
    // Cross product: (P - P0) × v
    float cross_x = py_vec * dz - pz_vec * dy;
    float cross_y = pz_vec * dx - px_vec * dz;
    float cross_z = px_vec * dy - py_vec * dx;
    
    // Distance = |cross| / |v|
    float cross_mag = std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    float line_mag = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    if (line_mag < 1e-6f) {
        // Line is degenerate (start == end), return distance to point
        return std::sqrt(px_vec * px_vec + py_vec * py_vec + pz_vec * pz_vec);
    }
    
    return cross_mag / line_mag;
}

bool MultiObjectTracker::set_unique_id_for_closest_vehicle(const std::string& unique_id, int road_id, std::vector<std::array<float, 4>> &rendered_points) {
    std::vector<detect::ProcessingBox> bboxes ;
    const float cx = get_config().center_x;
    const float cy = get_config().center_y;
    const float cz = get_config().min_z;
    const float w  = get_config().range_x * 2.0f;
    const float l  = get_config().range_y * 2.0f;
    const float h  = get_config().range_z;
    detect::ProcessingBox range_box(cx, cy, cz, w, l, h, 0.0f, -1.0, 1.0f);
    bboxes.push_back(range_box);
    
    // detect::SaveBoxesAsPCD({}, points_filtered.data(), points_filtered.size()/4, "", get_config().point_cloud_draw_step, rendered_points);
    int best_idx = -1;
    float min_distance = get_config().min_distance;  // 最多离多少米
    
    // Line parameters from config
    const auto& line1_config = get_config().line1_config;
    float line_start_x = line1_config.start_x;
    float line_start_y = line1_config.start_y;
    float line_start_z = line1_config.start_z;
    float line_end_x = line1_config.end_x;
    float line_end_y = line1_config.end_y;
    float line_end_z = line1_config.end_z;

    std::lock_guard<std::mutex> lock(trackers_mutex_);
    int n = trackers_.size(); 
    for (size_t idx = 0; idx < n ; ++idx) {
        // Skip if unique_id has already been set
        if (!trackers_id_[idx].empty()) {
            continue;
        }
        const auto& trk = trackers_[idx];
        if (trk.get_time_since_update() != 0  ) { //只要更新了的
            continue;
        }
        
        BBox3D val = trk.get_last_observation();
        
        if ( std::fabs(val.line() - road_id) >= 0.99f) {
            continue;
        }
        
        // Calculate distance from vehicle center to line
        float distance = point_to_line_distance(
             val.y(),val.x(), val.z(),    // 再转为雷达坐标系
            line_start_x, line_start_y, line_start_z,
            line_end_x, line_end_y, line_end_z
        );
        // Find the closest one
        if (distance < min_distance) {
            min_distance = distance;
            best_idx = idx;
        }
    }
    
    if (best_idx == -1) {
        bool draw_speed_line = (speed_strategy_ == SpeedStrategy::CLOSEST_3_TO_SPEED_LINE);
        detect::SaveBoxesAsPCD(bboxes, points_now.data(), points_now.size()/4, "", get_config().point_cloud_draw_step, rendered_points, draw_speed_line);
        return false;
    }
    
    // Set unique_id
    trackers_id_[best_idx] = unique_id;
    
    // Get vehicle information and save to result_map_
    const auto& trk = trackers_[best_idx ];
    BBox3D val; 
    val = trk.get_last_observation();
    pointpillar::lidar::BoundingBox best_box(val.y(), val.x()+0.5*val.l(), val.z(), val.l(),val.w(), val.h(), val.yaw(), -1.0, 1.0f);
    //printf("unique_id: %s yaw: %f\n", unique_id.c_str(), val.yaw());
    bboxes.push_back(best_box);
    std::swap(bboxes[0], bboxes[1]);  //让ROI 放后面
    // else
    // {
    //     val = trk.get_state();
    //     pointpillar::lidar::BoundingBox best_box(val.y(), val.x()+0.5*val.l(), val.z(), val.l(),val.w(), val.h(), val.yaw(), -1.0, 1.0f);
    //     bboxes.push_back(best_box);
    //     std::swap(bboxes[0], bboxes[1]);  //让ROI 放后面
    // }
    
    bool draw_speed_line = (speed_strategy_ == SpeedStrategy::CLOSEST_3_TO_SPEED_LINE);
    detect::SaveBoxesAsPCD(bboxes, points_now.data(), points_now.size()/4, "", get_config().point_cloud_draw_step, rendered_points, draw_speed_line);

    BestResult best_result;
    best_result.length = trk.get_length(dimension_strategy_);
    best_result.width = trk.get_width(dimension_strategy_);
    best_result.height = trk.get_height(dimension_strategy_);
    best_result.centre_l = val.x();
    best_result.centre_w = val.y();
    best_result.centre_h = val.z();
    best_result.speed = trk.get_speed(speed_strategy_);
    best_result.score = trk.get_score();
    uint64_t timestamp = trk.get_last_timestamp();
    best_result.points_max_car.clear();
    // Update result_map_ with status_code 0 (success)
    update_result_entry(unique_id, best_result, 0, timestamp);
    return true;
}

float MultiObjectTracker::euclidean_distance(const BBox3D& b1, const BBox3D& b2) const {
    if (std::fabs(b1.line() - b2.line()) >= 1.0f) {
        return 99999999.0f;
    }
    float dx = b1.x() - b2.x();
    float dy = b1.y() - b2.y();
    float dz = b1.z() - b2.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Calculate 3D bounding box IOU (Intersection over Union)
// BBox3D: [x, y, z, l, w, h, yaw, line, score]
// x和l对应车长, y和w对应车宽, z和h对应车高
// xyz表示车最前最下宽的中心，需要还原为实际中心点
// yaw是绕z轴（车高）旋转的角度，单位为弧度
float MultiObjectTracker::bbox3d_iou(const BBox3D& b1, const BBox3D& b2) const {
    // Convert from "车前下点" to actual center point
    // x是最前端，中心应该在 x + l/2
    // y是宽的中心，保持不变
    // z是最下端，中心应该在 z + h/2
    if (std::fabs(b1.line() - b2.line()) >= 1.0f) {
        return 0.0f;
    }
    float center1_x = b1.x() + b1.l() * 0.5f;
    float center1_y = b1.y();
    float center1_z = b1.z() + b1.h() * 0.5f;
    
    float center2_x = b2.x() + b2.l() * 0.5f;
    float center2_y = b2.y();
    float center2_z = b2.z() + b2.h() * 0.5f;
    
    // Calculate half dimensions for each box
    float half1_l = b1.l() * 0.5f;
    float half1_w = b1.w() * 0.5f;
    float half1_h = b1.h() * 0.5f;
    
    float half2_l = b2.l() * 0.5f;
    float half2_w = b2.w() * 0.5f;
    float half2_h = b2.h() * 0.5f;
    
    // Get yaw angles (in radians, rotation around z-axis)
    float yaw1 = b1.yaw();
    float yaw2 = b2.yaw();
    
    // For rotated boxes, compute rotated bounding box (AABB) approximation
    // Calculate the axis-aligned bounding box (AABB) of the rotated box
    auto compute_rotated_aabb = [](float cx, float cy, float cz,
                                   float half_l, float half_w, float half_h,
                                   float yaw,
                                   float& min_x, float& max_x,
                                   float& min_y, float& max_y,
                                   float& min_z, float& max_z) {
        float cosr = std::cos(yaw);
        float sinr = std::sin(yaw);
        
        // Rotate the 4 corners of the bottom face
        auto rot = [&](float x, float y) -> std::pair<float, float> {
            return {x * cosr - y * sinr, x * sinr + y * cosr};
        };
        
        // Bottom face corners (relative to center)
        auto p0 = rot(-half_w, -half_l);
        auto p1 = rot( half_w, -half_l);
        auto p2 = rot( half_w,  half_l);
        auto p3 = rot(-half_w,  half_l);
        
        // Find min/max in x and y
        min_x = std::min({cx + p0.first, cx + p1.first, cx + p2.first, cx + p3.first});
        max_x = std::max({cx + p0.first, cx + p1.first, cx + p2.first, cx + p3.first});
        min_y = std::min({cy + p0.second, cy + p1.second, cy + p2.second, cy + p3.second});
        max_y = std::max({cy + p0.second, cy + p1.second, cy + p2.second, cy + p3.second});
        
        // Z axis is not affected by yaw rotation
        min_z = cz - half_h;
        max_z = cz + half_h;
    };
    
    float min1_x, max1_x, min1_y, max1_y, min1_z, max1_z;
    float min2_x, max2_x, min2_y, max2_y, min2_z, max2_z;
    
    compute_rotated_aabb(center1_x, center1_y, center1_z, 
                        half1_l, half1_w, half1_h, yaw1,
                        min1_x, max1_x, min1_y, max1_y, min1_z, max1_z);
    
    compute_rotated_aabb(center2_x, center2_y, center2_z,
                        half2_l, half2_w, half2_h, yaw2,
                        min2_x, max2_x, min2_y, max2_y, min2_z, max2_z);
    
    // Calculate intersection of AABBs
    float inter_min_x = std::max(min1_x, min2_x);
    float inter_max_x = std::min(max1_x, max2_x);
    float inter_min_y = std::max(min1_y, min2_y);
    float inter_max_y = std::min(max1_y, max2_y);
    float inter_min_z = std::max(min1_z, min2_z);
    float inter_max_z = std::min(max1_z, max2_z);
    
    // Calculate intersection volume
    float inter_volume = 0.0f;
    if (inter_max_x > inter_min_x && inter_max_y > inter_min_y && inter_max_z > inter_min_z) {
        inter_volume = (inter_max_x - inter_min_x) * 
                      (inter_max_y - inter_min_y) * 
                      (inter_max_z - inter_min_z);
    }
    
    // Calculate union volume
    float volume1 = b1.l() * b1.w() * b1.h();
    float volume2 = b2.l() * b2.w() * b2.h();
    float union_volume = volume1 + volume2 - inter_volume;
    
    // Calculate IOU
    if (union_volume < 1e-6f) {
        return 0.0f;
    }
    
    return inter_volume / union_volume;
}

// Simplified Hungarian algorithm for assignment
MultiObjectTracker::Assignment MultiObjectTracker::linear_sum_assignment(
    const std::vector<std::vector<float>>& cost) const {
    
    Assignment assignment;
    
    int n_trk = cost.size();
    int n_det = cost.empty() ? 0 : cost[0].size();
    
    if (n_trk == 0 || n_det == 0) {
        for (int i = 0; i < n_trk; ++i) {
            assignment.unmatched_trk.push_back(i);
        }
        for (int j = 0; j < n_det; ++j) {
            assignment.unmatched_det.push_back(j);
        }
        return assignment;
    }
    
    // Simple greedy assignment
    std::set<int> assigned_dets;
    std::set<int> assigned_trks;
    
    // Find minimum cost pairs
    for (int i = 0; i < n_trk; ++i) {
        float min_cost = std::numeric_limits<float>::max();
        int best_j = -1;
        
        for (int j = 0; j < n_det; ++j) {
            if (cost[i][j] < min_cost && cost[i][j] <= iou_threshold_) {
                min_cost = cost[i][j];
                best_j = j;
            }
        }
        
        if (best_j != -1 && assigned_dets.find(best_j) == assigned_dets.end()) {
            assignment.matched.emplace_back(i, best_j);
            assigned_dets.insert(best_j);
            assigned_trks.insert(i);
        }
    }
    
    for (int i = 0; i < n_trk; ++i) {
        if (assigned_trks.find(i) == assigned_trks.end()) {
            assignment.unmatched_trk.push_back(i);
        }
    }
    
    for (int j = 0; j < n_det; ++j) {
        if (assigned_dets.find(j) == assigned_dets.end()) {
            assignment.unmatched_det.push_back(j);
        }
    }
    
    return assignment;
}

void MultiObjectTracker::update_result_entry(const std::string& unique_id, BestResult& result, int status_code, uint64_t timestamp) {
    result_map_[unique_id] = ResultEntry(result, status_code, timestamp);
}

} // namespace tracking

