#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <map>
#include <string>
#include <mutex>

namespace tracking {

// Strategy enum for getting length/width/height from observations
enum class DimensionStrategy {
    MAX_VALUE = 1,              // 取最大值
    AVERAGE = 2,                // 取平均值
    TRIMMED_AVERAGE = 3,        // 去除最大的20%和最小的20%后，再求平均值
    TRIMMED_MAX = 4,            // 去除最大的前5%后，再取最大值
    FIRST_10_PERCENT_AVG = 5,   // 最大的前10%的平均值
    MEDIAN = 6                  // 取最大20%的中位数
};

// Strategy enum for getting speed from observations
enum class SpeedStrategy {
    FIRST_80_PERCENT_AVG = 1,           // 前80%的平均值（过滤超过200km/h的值）
    CLOSEST_3_TO_SPEED_LINE = 2         // 距离speed_line最近的前3个点的速度的最大值（过滤超过200km/h的值）
};

// Line structure for 3D line segment
struct Line {
    float start_x, start_y, start_z;
    float end_x, end_y, end_z;
    
    Line() : start_x(0), start_y(0), start_z(0), end_x(0), end_y(0), end_z(0) {}
    Line(float sx, float sy, float sz, float ex, float ey, float ez)
        : start_x(sx), start_y(sy), start_z(sz), end_x(ex), end_y(ey), end_z(ez) {}
};

// Simple Kalman Filter for 3D position tracking
class KalmanFilter3D {
public:
    KalmanFilter3D();
    void init(const std::array<float, 3>& x0);
    std::array<float, 3> predict(float dt);
    void update(const std::array<float, 3>& z);
    std::array<float, 3> get_state() const;
    
private:
    std::array<float, 6> x_;  // state: [x, y, z, vx, vy, vz]
    float P_[6][6];           // covariance matrix
    float Q_[6][6];           // process noise
    float R_[3][3];           // measurement noise
    float F_[6][6];           // state transition matrix
};

// Single Object Tracker
struct BBox3D {
    std::array<float, 9> data;  // [x, y, z, l, w, h, yaw, line, score]
    
    BBox3D() : data{0} {}
    BBox3D(float x, float y, float z, float l, float w, float h, float yaw, float line, float score = 0.0f) {
        data[0] = x; data[1] = y; data[2] = z;
        data[3] = l; data[4] = w; data[5] = h;
        data[6] = yaw; data[7] = line; data[8] = score;
    }
    
    float& x() { return data[0]; }
    float& y() { return data[1]; }
    float& z() { return data[2]; }
    float& l() { return data[3]; }
    float& w() { return data[4]; }
    float& h() { return data[5]; }
    float& yaw() { return data[6]; }
    float& line() { return data[7]; }
    float& score() { return data[8]; }
    
    const float& x() const { return data[0]; }
    const float& y() const { return data[1]; }
    const float& z() const { return data[2]; }
    const float& l() const { return data[3]; }
    const float& w() const { return data[4]; }
    const float& h() const { return data[5]; }
    const float& yaw() const { return data[6]; }
    const float& line() const { return data[7]; }
    const float& score() const { return data[8]; }
};

class Tracker {
public:
    Tracker(const BBox3D& bbox3d, int id, uint64_t timestamp);
    
    BBox3D predict(float dt = 0.1f);
    void update(const BBox3D& bbox3d, uint64_t timestamp, std::vector<std::array<float, 4>> &points_max_car );
    BBox3D get_state() const;
    BBox3D get_last_observation() const;
    float get_speed(SpeedStrategy strategy) const;
    float get_length(DimensionStrategy strategy) const;
    float get_width(DimensionStrategy strategy) const;
    float get_height(DimensionStrategy strategy) const;
    float get_score() const { return score_; }
    int get_id() const { return id_; }
    uint64_t get_last_timestamp() const { return last_timestamp_; }
    int get_time_since_update() const { return time_since_update_; }
    std::vector<std::array<float, 4>> points_max_car_;  // 维护最大的点云数据

private:
    KalmanFilter3D kf_;
    std::vector<BBox3D> observations_;  // Array of observations, similar to speed_
    int id_;
    int time_since_update_;
    int hits_;
    std::vector<float> speed_;  // Speed values
    std::vector<std::array<float, 3>> speed_positions_;  // Position (dx, dy, dz) for each speed measurement
    float score_;
    uint64_t last_timestamp_;
    
    float yaw_, line_;  // yaw and line are updated directly, not from observations
    
    // Helper function to get dimension value based on strategy
    float get_dimension_value(const std::vector<float>& values, DimensionStrategy strategy) const;
    
    // Helper function to get speed value based on strategy
    float get_speed_value(const std::vector<float>& speeds, SpeedStrategy strategy) const;
};

// Multi-Object Tracker
class MultiObjectTracker {
public:
    MultiObjectTracker(float iou_threshold = 0.7f, int max_age = 5,
                      DimensionStrategy dimension_strategy = DimensionStrategy::TRIMMED_MAX,
                      SpeedStrategy speed_strategy = SpeedStrategy::CLOSEST_3_TO_SPEED_LINE);
    
    void update(std::vector<BBox3D>& detections, std::vector<std::vector<std::array<float, 4>>> &car_points_frame, uint64_t timestamp ,std::vector<float> &points);
    
    struct BestResult {
        float length, width, height;
        float centre_l, centre_w, centre_h;
        float speed;
        float score;
        std::vector<std::array<float, 4>> points_max_car;
    };
    
    std::vector<float> points_now;
    std::vector<BBox3D> Boxes_now;
    
    // Result entry for unique_id map
    struct ResultEntry {
        BestResult result;
        int status_code;   //0: 存入, 1: 可以取了, 2: 超时了
        uint64_t timestamp;
        
        ResultEntry() : status_code(0), timestamp(0) {}
        ResultEntry(const BestResult& res, int status, uint64_t ts)
            : result(res), status_code(status), timestamp(ts) {}
    };
    
    bool set_unique_id_for_closest_vehicle(const std::string& unique_id, int road_id,std::vector<std::array<float, 4>> &rendered_points);
    
    // Update result entry for a unique_id
    void update_result_entry(const std::string& unique_id, BestResult& result, int status_code, uint64_t timestamp);
    
    std::map<std::string, ResultEntry> result_map_;
private:
    std::vector<Tracker> trackers_;
    std::vector<std::string> trackers_id_;  // Track unique_id for each tracker, default is ""
    int next_id_;
    float iou_threshold_;
    int max_age_;
    DimensionStrategy dimension_strategy_;  // Strategy for getting dimensions
    SpeedStrategy speed_strategy_;          // Strategy for getting speed
    mutable std::mutex trackers_mutex_;  // Mutex to protect trackers_ and trackers_id_
    
    float euclidean_distance(const BBox3D& b1, const BBox3D& b2) const;
    
    // Calculate 3D bounding box IOU (Intersection over Union)
    // Returns IOU value between 0.0 and 1.0, higher value means more similar
    // Different lanes still get IOU score (not filtered out)
    float bbox3d_iou(const BBox3D& b1, const BBox3D& b2) const;
    
    // Linear Sum Assignment (Hungarian Algorithm)
    struct Assignment {
        std::vector<std::pair<int, int>> matched;
        std::vector<int> unmatched_trk;
        std::vector<int> unmatched_det;
    };
    
    Assignment linear_sum_assignment(const std::vector<std::vector<float>>& cost) const;
    
    // Filter out detections that are too close to line2
    void filter_detections_by_line2_distance(std::vector<BBox3D>& detections, 
                                             std::vector<std::vector<std::array<float, 4>>>& car_points_frame) const;
};

} // namespace tracking

#endif // TRACKING_HPP

