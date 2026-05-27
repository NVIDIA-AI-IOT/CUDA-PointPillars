#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <mutex>
#include <string>

//---------------------------
// 结构体定义
//---------------------------
struct LineConfig {
    float start_x = 0, start_y = 0, start_z = 0;
    float end_x = 0, end_y = 0, end_z = 0;
};

struct RangeConfig {
    float center_x = 0, center_y = 0, center_z = 0;
    float range_x = 0, range_y = 0, range_z = 0;

    float min_x = 0, min_y = 0, min_z = 0;
    float max_x = 0, max_y = 0, max_z = 0;

    float ry = 0;
    int lane_count = 0;
    
    std::string lidar_model;
    
    // Lidar SDK configuration
    float lidar_roll = 0;      // 翻滚
    float lidar_pitch = 0;      // 上下
    float lidar_yaw = 0;        // 转动
    float lidar_x = 0;
    float lidar_y = 0;
    float lidar_z = 0;
    std::string lidar_ip_address;  // lidar ip or pcap file path

    // Web service configuration
    std::string web_service_ip;    // web service IP address
    int web_service_port;           // web service port
    int web_service_timeout_sec;    // 超时时间connection/read/write timeout in seconds

    // Tracking configuration
    float min_distance;             // 最多离触发线多少米 

    // Detection configuration
    float detection_score_threshold;  // 检测框分数阈值（后处理过滤）

    // Draw configuration
    float point_cloud_draw_step;  // 点云绘图步长（用于SaveBoxesAsPCD函数）

    // Ground plane parameters (ax + by + cz + d = 0)
    double ground_plane_a;  // 地平面参数 a
    double ground_plane_b;  // 地平面参数 b
    double ground_plane_c;  // 地平面参数 c
    double ground_plane_d;  // 地平面参数 d
    float ground_distance_threshold;  // 地平面距离阈值（用于过滤地面点）

    LineConfig line1_config;
    LineConfig line2_config;
    float speed_calib = 0.0f;  // 车速校准值（单位：km/h），最终速度会加上这个值
    LineConfig speed_line_config;  // Speed line for speed calculation

    // DBSCAN clustering configuration
    bool use_dbscan = true;              // 是否启用 DBSCAN 聚类
    float dbscan_eps_xy = 1.5f;          // DBSCAN XY平面邻域半径（米）- x,y方向允许更大的距离
    float dbscan_eps_z = 0.2f;           // DBSCAN Z方向邻域半径（米）- z方向要求更严格
    float dbscan_max_cluster_ratio = 0.05f; // 最大簇比例阈值：点数小于最大簇*此比例的簇可能被删除
    float dbscan_z_threshold = 0.19f;       // Z值阈值（米）：如果簇的平均z值 > 主簇平均z值 + 此阈值，则删除（用于过滤树叶等高空噪声）

    // Data points configuration
    int data_points_type = 0;            // 数据类型：0表示不返回点云数据，只返回路径
    std::string points_file_path = "/mnt/data/sda/ocm_3/";  // 点云数据文件保存路径
    
    // Snowflake algorithm configuration
    int snowflake_datacenter_id = 1;    // 雪花算法数据中心ID，范围0-31
    int snowflake_worker_id = 1;        // 雪花算法机器ID，范围0-31
};

//---------------------------
// 单例类定义
//---------------------------
class RangeConfigSingleton {
public:
    // 线程安全、无锁的 C++11 单例
    static RangeConfig& getInstance();
    
    // 设置 JSON 配置文件路径（必须在第一次调用 getInstance() 之前调用）
    static void setConfigFilePath(const std::string& json_file_path);

    RangeConfigSingleton(const RangeConfigSingleton&) = delete;
    RangeConfigSingleton& operator=(const RangeConfigSingleton&) = delete;

private:
    RangeConfigSingleton() = default;
    ~RangeConfigSingleton() = default;

    static void initializeConfig(RangeConfig& config, const std::string& json_file_path);
    static std::string config_file_path_;
};

// 全局简单接口
inline RangeConfig& get_config() {
    return RangeConfigSingleton::getInstance();
}

#endif // CONFIG_HPP
