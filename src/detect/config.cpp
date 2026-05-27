#include "config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

//-------------------------------------------
// 静态成员变量
//-------------------------------------------
std::string RangeConfigSingleton::config_file_path_ = "";

//-------------------------------------------
// 设置配置文件路径
//-------------------------------------------
void RangeConfigSingleton::setConfigFilePath(const std::string& json_file_path) {
    config_file_path_ = json_file_path;
}

//-------------------------------------------
// 单例实例
//-------------------------------------------
RangeConfig& RangeConfigSingleton::getInstance() {
    static RangeConfig instance;

    static bool initialized = false;
    if (!initialized) {
        initializeConfig(instance, config_file_path_);
        initialized = true;
    }

    return instance;
}

//-------------------------------------------
// 初始化配置（从 JSON 文件或使用默认值）
//-------------------------------------------
void RangeConfigSingleton::initializeConfig(RangeConfig& config, const std::string& json_file_path) {
    // 设置默认值
    float center_x = 30.9;
    float center_y = 7.2;
    float center_z = -1.9;
    float range_x = 8.3;
    float range_y = 20;
    float range_z = 6;
    float ry = 0.0f;
    int lane_count = 4;
    
    // 尝试从 JSON 文件读取配置
    if (!json_file_path.empty()) {
        try {
            std::ifstream file(json_file_path);
            if (file.is_open()) {
                nlohmann::json j;
                file >> j;
                
                // 读取基本配置
                if (j.contains("center_x")) center_x = j["center_x"];
                if (j.contains("center_y")) center_y = j["center_y"];
                if (j.contains("center_z")) center_z = j["center_z"];
                if (j.contains("range_x")) range_x = j["range_x"];
                if (j.contains("range_y")) range_y = j["range_y"];
                if (j.contains("range_z")) range_z = j["range_z"];
                if (j.contains("ry")) ry = j["ry"];
                if (j.contains("lane_count")) lane_count = j["lane_count"];
                
                // 读取 lidar_model
                if (j.contains("lidar_model")) {
                    config.lidar_model = j["lidar_model"];
                } else {
                    config.lidar_model = "../model/pointpillar.plan";
                }
                
                // 读取 Lidar SDK 配置
                if (j.contains("lidar")) {
                    auto& lidar = j["lidar"];
                    if (lidar.contains("roll")) config.lidar_roll = lidar["roll"].get<float>()* 3.141592653589793 / 180;
                    if (lidar.contains("pitch")) config.lidar_pitch = lidar["pitch"].get<float>()* 3.141592653589793 / 180;
                    if (lidar.contains("yaw")) config.lidar_yaw = lidar["yaw"].get<float>()* 3.141592653589793 / 180;
                    if (lidar.contains("x")) config.lidar_x = lidar["x"];
                    if (lidar.contains("y")) config.lidar_y = lidar["y"];
                    if (lidar.contains("z")) config.lidar_z = lidar["z"];
                    if (lidar.contains("ip_address")) config.lidar_ip_address = lidar["ip_address"];
                } else {
                    // 默认值
                    config.lidar_roll = 1.6f * 3.141592653589793 / 180;
                    config.lidar_pitch = 10.8 * 3.141592653589793 / 180;
                    config.lidar_yaw = 101.4 * 3.141592653589793 / 180;
                    config.lidar_x = 37.0f;
                    config.lidar_y = -27.6f;
                    config.lidar_z = 4.3f;
                    config.lidar_ip_address = "10.168.1.62";
                }
                
                // 读取 Web service 配置
                if (j.contains("web_service")) {
                    auto& web = j["web_service"];
                    if (web.contains("ip")) config.web_service_ip = web["ip"];
                    if (web.contains("port")) config.web_service_port = web["port"];
                    if (web.contains("timeout_sec")) config.web_service_timeout_sec = web["timeout_sec"];
                } else {
                    config.web_service_ip = "10.168.1.66";
                    config.web_service_port = 18874;
                    config.web_service_timeout_sec = 15;
                }
                
                // 读取 Tracking 配置
                if (j.contains("tracking")) {
                    auto& tracking = j["tracking"];
                    if (tracking.contains("min_distance")) config.min_distance = tracking["min_distance"];
                } else {
                    config.min_distance = 8.0f;
                }
                
                // 读取 Detection 配置
                if (j.contains("detection")) {
                    auto& detection = j["detection"];
                    if (detection.contains("score_threshold")) config.detection_score_threshold = detection["score_threshold"];
                } else {
                    config.detection_score_threshold = 0.2f;
                }
                
                // 读取 Draw 配置
                if (j.contains("draw")) {
                    auto& draw = j["draw"];
                    if (draw.contains("point_cloud_draw_step")) config.point_cloud_draw_step = draw["point_cloud_draw_step"];
                } else {
                    config.point_cloud_draw_step = 0.05f;
                }
                
                // 读取 Ground plane 配置
                if (j.contains("ground_plane")) {
                    auto& gp = j["ground_plane"];
                    if (gp.contains("a")) config.ground_plane_a = gp["a"];
                    if (gp.contains("b")) config.ground_plane_b = gp["b"];
                    if (gp.contains("c")) config.ground_plane_c = gp["c"];
                    if (gp.contains("d")) config.ground_plane_d = gp["d"];
                } else {
                    config.ground_plane_a = 0.00283535;
                    config.ground_plane_b = 0.00755401;
                    config.ground_plane_c = -0.999967;
                    config.ground_plane_d = -4.08259;
                }
                
                // 读取 ground_distance_threshold 配置
                if (j.contains("ground_distance_threshold")) {
                    config.ground_distance_threshold = j["ground_distance_threshold"].get<float>();
                } else {
                    config.ground_distance_threshold = 0.2f;  // 默认值
                }
                
                // 读取 Line1 配置
                if (j.contains("line1")) {
                    auto& line1 = j["line1"];
                    if (line1.contains("start_x")) config.line1_config.start_x = line1["start_x"];
                    if (line1.contains("start_y")) config.line1_config.start_y = line1["start_y"];
                    if (line1.contains("start_z")) config.line1_config.start_z = line1["start_z"];
                    if (line1.contains("end_x")) config.line1_config.end_x = line1["end_x"];
                    if (line1.contains("end_y")) config.line1_config.end_y = line1["end_y"];
                    if (line1.contains("end_z")) config.line1_config.end_z = line1["end_z"];
                } else {
                    config.line1_config.start_x = 40.0f;
                    config.line1_config.start_y = -2.0f;
                    config.line1_config.start_z = -3.0f;
                    config.line1_config.end_x = -1.0f;
                    config.line1_config.end_y = -2.0f;
                    config.line1_config.end_z = -3.0f;
                }
                
                // 读取 Line2 配置
                if (j.contains("line2")) {
                    auto& line2 = j["line2"];
                    if (line2.contains("start_x")) config.line2_config.start_x = line2["start_x"];
                    if (line2.contains("start_y")) config.line2_config.start_y = line2["start_y"];
                    if (line2.contains("start_z")) config.line2_config.start_z = line2["start_z"];
                    if (line2.contains("end_x")) config.line2_config.end_x = line2["end_x"];
                    if (line2.contains("end_y")) config.line2_config.end_y = line2["end_y"];
                    if (line2.contains("end_z")) config.line2_config.end_z = line2["end_z"];
                } else {
                    config.line2_config.start_x = 45.0f;
                    config.line2_config.start_y = -14.0f;
                    config.line2_config.start_z = -3.5f;
                    config.line2_config.end_x = 0.0f;
                    config.line2_config.end_y = -14.0f;
                    config.line2_config.end_z = -3.5f;
                }
                
                // 读取 speed_calib 配置（单位：km/h）
                if (j.contains("speed_calib")) {
                    config.speed_calib = j["speed_calib"].get<float>();
                } else {
                    config.speed_calib = 0.0f;
                }
                
                // 读取 speed_line 配置
                if (j.contains("speed_line")) {
                    auto& speed_line = j["speed_line"];
                    if (speed_line.contains("start_x")) config.speed_line_config.start_x = speed_line["start_x"];
                    if (speed_line.contains("start_y")) config.speed_line_config.start_y = speed_line["start_y"];
                    if (speed_line.contains("start_z")) config.speed_line_config.start_z = speed_line["start_z"];
                    if (speed_line.contains("end_x")) config.speed_line_config.end_x = speed_line["end_x"];
                    if (speed_line.contains("end_y")) config.speed_line_config.end_y = speed_line["end_y"];
                    if (speed_line.contains("end_z")) config.speed_line_config.end_z = speed_line["end_z"];
                } else {
                    config.speed_line_config.start_x = 45.0f;
                    config.speed_line_config.start_y = -5.0f;
                    config.speed_line_config.start_z = -3.0f;
                    config.speed_line_config.end_x = 0.0f;
                    config.speed_line_config.end_y = -5.0f;
                    config.speed_line_config.end_z = -3.0f;
                }

                // DBSCAN configuration
                if (j.contains("dbscan") && j["dbscan"].is_object()) {
                    const auto& dbscan = j["dbscan"];
                    if (dbscan.contains("use_dbscan")) config.use_dbscan = dbscan["use_dbscan"].get<bool>();
                    if (dbscan.contains("eps_xy")) config.dbscan_eps_xy = dbscan["eps_xy"].get<float>();
                    if (dbscan.contains("eps_z")) config.dbscan_eps_z = dbscan["eps_z"].get<float>();
                    if (dbscan.contains("max_cluster_ratio")) config.dbscan_max_cluster_ratio = dbscan["max_cluster_ratio"].get<float>();
                    if (dbscan.contains("z_threshold")) config.dbscan_z_threshold = dbscan["z_threshold"].get<float>();
                } else {
                    config.use_dbscan = false;
                    config.dbscan_eps_xy = 1.5f;
                    config.dbscan_eps_z = 0.2f;
                    config.dbscan_max_cluster_ratio = 0.05f;
                    config.dbscan_z_threshold = 0.19f;
                }

                // Data points configuration
                if (j.contains("data_points_type")) config.data_points_type = j["data_points_type"].get<int>();
                if (j.contains("points_file_path")) config.points_file_path = j["points_file_path"].get<std::string>();
                
                // Snowflake algorithm configuration
                if (j.contains("snowflake_datacenter_id")) config.snowflake_datacenter_id = j["snowflake_datacenter_id"].get<int>();
                if (j.contains("snowflake_worker_id")) config.snowflake_worker_id = j["snowflake_worker_id"].get<int>();
                
                std::cout << "Configuration loaded from: " << json_file_path << std::endl;
            } else {
                std::cerr << "Warning: Cannot open config file: " << json_file_path << ", using default values." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing config file: " << e.what() << ", using default values." << std::endl;
        }
    } else {
        // 如果没有指定 JSON 文件路径，使用默认值
        config.lidar_model = "../model/pointpillar.plan";
        config.lidar_roll = 1.6f * 3.141592653589793 / 180;
        config.lidar_pitch = 10.8 * 3.141592653589793 / 180;
        config.lidar_yaw = 101.4 * 3.141592653589793 / 180;
        config.lidar_x = 37.0f;
        config.lidar_y = -27.6f;
        config.lidar_z = 4.3f;
        config.lidar_ip_address = "10.168.1.62";
        config.web_service_ip = "10.168.1.66";
        config.web_service_port = 18874;
        config.web_service_timeout_sec = 15;
        config.min_distance = 8.0f;
        config.detection_score_threshold = 0.2f;
        config.point_cloud_draw_step = 0.05f;
        config.ground_plane_a = 0.00283535;
        config.ground_plane_b = 0.00755401;
        config.ground_plane_c = -0.999967;
        config.ground_plane_d = -4.08259;
        config.ground_distance_threshold = 0.2f;  // 默认值
        config.line1_config.start_x = 40.0f;
        config.line1_config.start_y = -2.0f;
        config.line1_config.start_z = -3.0f;
        config.line1_config.end_x = -1.0f;
        config.line1_config.end_y = -2.0f;
        config.line1_config.end_z = -3.0f;
        config.line2_config.start_x = 45.0f;
        config.line2_config.start_y = -14.0f;
        config.line2_config.start_z = -3.5f;
        config.line2_config.end_x = 0.0f;
        config.line2_config.end_y = -14.0f;
        config.line2_config.end_z = -3.5f;
        config.speed_calib = 0.0f;  // 默认校准值为 0 km/h
        config.speed_line_config.start_x = 45.0f;
        config.speed_line_config.start_y = -5.0f;
        config.speed_line_config.start_z = -3.0f;
        config.speed_line_config.end_x = 0.0f;
        config.speed_line_config.end_y = -5.0f;
        config.speed_line_config.end_z = -3.0f;
        config.use_dbscan = true;
        config.dbscan_eps_xy = 1.5f;
        config.dbscan_eps_z = 0.2f;
        config.dbscan_max_cluster_ratio = 0.1f;
        config.dbscan_z_threshold = 0.2f;
        config.data_points_type = 0;
        config.points_file_path = "/mnt/data/sda/ocm_3/";
        config.snowflake_datacenter_id = 1;
        config.snowflake_worker_id = 1;
    }
    
    // 设置基本配置（从 JSON 或默认值）
    config.center_x = center_x;
    config.center_y = center_y;
    config.center_z = center_z;
    config.range_x = range_x;
    config.range_y = range_y;
    config.range_z = range_z;
    
    // 计算 min/max
    config.min_x = center_x - range_x;
    config.min_y = center_y - range_y;
    config.min_z = center_z;
    config.max_x = center_x + range_x;
    config.max_y = center_y + range_y;
    config.max_z = center_z + range_z;
    
    config.ry = ry;
    config.lane_count = lane_count;
}
