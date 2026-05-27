#include "http_server.hpp"
#include "config.hpp"
#include "snowflake.hpp"
#include <hv/HttpServer.h>
#include <hv/HttpService.h>
#include <hv/HttpClient.h>
#include <hv/hasync.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <exception>
#include <thread>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <mutex>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <regex>
#include <memory>
#include <atomic>
#include <cstring>
#include <cerrno>

using json = nlohmann::json;
using namespace hv;

namespace http_server {

// 全局雪花算法实例（延迟初始化）
static std::unique_ptr<snowflake::Snowflake> g_snowflake = nullptr;
static std::once_flag g_snowflake_init_flag;

// 初始化雪花算法实例
static void init_snowflake() {
    const auto& config = get_config();
    g_snowflake = std::make_unique<snowflake::Snowflake>(
        config.snowflake_datacenter_id,
        config.snowflake_worker_id
    );
}

// 获取雪花算法实例
static snowflake::Snowflake& get_snowflake() {
    std::call_once(g_snowflake_init_flag, init_snowflake);
    return *g_snowflake;
}

// 全局日志文件流和互斥锁
static std::ofstream g_log_file;
static std::mutex g_log_mutex;
static std::string g_current_log_filename;

// 获取当前日期和小时格式的日志文件名：YYYY-MM-DD_HH.log
static std::string get_log_filename() {
    std::time_t now = std::time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    
    std::ostringstream filename_stream;
    filename_stream << "log/" 
                    << std::setfill('0') << std::setw(4) << (1900 + timeinfo->tm_year) << "-"
                    << std::setfill('0') << std::setw(2) << (timeinfo->tm_mon + 1) << "-"
                    << std::setfill('0') << std::setw(2) << timeinfo->tm_mday << "_"
                    << std::setfill('0') << std::setw(2) << timeinfo->tm_hour << ".log";
    return filename_stream.str();
}

// 删除30天前的日志文件
static void delete_old_log_files() {
    DIR* dir = opendir("log");
    if (dir == nullptr) {
        return;  // log目录不存在，直接返回
    }
    
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);
    
    // 计算30天前的时间戳
    std::tm cutoff_tm = *now_tm;
    cutoff_tm.tm_mday -= 30;
    std::time_t cutoff_time = std::mktime(&cutoff_tm);
    
    // 正则表达式匹配日志文件名格式：yyyy-mm-dd_tt.log
    std::regex log_pattern(R"((\d{4})-(\d{2})-(\d{2})_(\d{2})\.log)");
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // 跳过 . 和 ..
        if (filename == "." || filename == "..") {
            continue;
        }
        
        // 检查文件名是否匹配日志格式
        std::smatch match;
        if (std::regex_match(filename, match, log_pattern)) {
            // 解析日期
            int year = std::stoi(match[1].str());
            int month = std::stoi(match[2].str());
            int day = std::stoi(match[3].str());
            
            // 构建时间结构
            std::tm file_tm = {};
            file_tm.tm_year = year - 1900;
            file_tm.tm_mon = month - 1;
            file_tm.tm_mday = day;
            file_tm.tm_hour = 0;
            file_tm.tm_min = 0;
            file_tm.tm_sec = 0;
            
            // 转换为时间戳
            std::time_t file_time = std::mktime(&file_tm);
            
            // 如果文件日期早于30天前，删除它
            if (file_time < cutoff_time) {
                std::string filepath = "log/" + filename;
                if (unlink(filepath.c_str()) == 0) {
                    // 删除成功（可选：记录日志）
                }
            }
        }
    }
    
    closedir(dir);
}

// 确保日志文件已打开（如果小时变化则切换文件）
static void ensure_log_file_open() {
    std::string log_filename = get_log_filename();
    
    // 如果文件流未打开或小时变化，需要打开/切换文件
    if (!g_log_file.is_open() || g_current_log_filename != log_filename) {
        // 关闭旧文件（如果打开）
        if (g_log_file.is_open()) {
            g_log_file.close();
        }
        
        // 创建log文件夹（如果不存在）
        struct stat info;
        if (stat("log", &info) != 0) {
            mkdir("log", 0755);
        }
        
        // 删除30天前的日志文件
        delete_old_log_files();
        
        // 打开新文件（追加模式）
        g_log_file.open(log_filename, std::ios::app);
        g_current_log_filename = log_filename;
    }
}

// 全局日志函数：将内容追加到按小时命名的日志文件中
static std::string write_log(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    
    // 获取当前时间（精确到毫秒）
    auto now_system = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now_system);
    std::tm* timeinfo = std::localtime(&now_time_t);
    
    // 计算毫秒部分
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now_system.time_since_epoch()) % 1000;
    
    // 确保日志文件已打开
    ensure_log_file_open();
    
    // 生成时间戳：YYYY-MM-DD HH:MM:SS.mmm
    std::ostringstream timestamp_stream;
    timestamp_stream << std::setfill('0') 
                     << std::setw(4) << (1900 + timeinfo->tm_year) << "-"
                     << std::setfill('0') << std::setw(2) << (timeinfo->tm_mon + 1) << "-"
                     << std::setfill('0') << std::setw(2) << timeinfo->tm_mday << " "
                     << std::setfill('0') << std::setw(2) << timeinfo->tm_hour << ":"
                     << std::setfill('0') << std::setw(2) << timeinfo->tm_min << ":"
                     << std::setfill('0') << std::setw(2) << timeinfo->tm_sec << "."
                     << std::setfill('0') << std::setw(3) << ms.count();
    
    // 写入日志文件
    if (g_log_file.is_open()) {
        g_log_file << "[" << timestamp_stream.str() << "] " << message << std::endl;
    }
    return timestamp_stream.str();
}

void start_pointcloud_server(const std::string& host, int port, DetectorFunc detector, bool* running) 
{
    HttpService service;
    HttpServer server(&service);
    
    // 设置线程数：支持每秒10次请求，每个请求处理3秒
    // 理论需求：10 QPS × 3秒 = 30个并发线程
    // 设置为150个线程以应对峰值和系统开销
    server.setThreadNum(50);
    
    // 设置超时时间：每个请求处理6秒，设置读写超时为8秒（留有余量）
    // libhv 默认超时较长，但我们可以通过 keepalive_timeout 设置
    service.keepalive_timeout = 8000;  // 8秒，单位：毫秒
    
    // Handle POST requests to /pointcloud/detect
    // 方案B：异步处理，使用 HttpResponseWriterPtr，耗时推理在 hv::async 完成后再 writer->End()
    service.POST("/pointcloud/detect", [detector](const HttpRequestPtr& req, const HttpResponseWriterPtr& writer) {
        std::string body = req->body;
        std::string timestamp_str = write_log("Body: " + body);
        
        try {
            // Parse JSON request
            json req_data = json::parse(body);
            
            // Validate request_type
            if (!req_data.contains("req_type") || req_data["req_type"] != "get_point_cloud_detect_request") {
                json error_resp = {
                    {"ret_type", "get_point_cloud_detect_response"},
                    {"ret_header", {
                        {"code", 1},
                        {"message", "Invalid request"}
                    }},
                    {"ret_body", json::object()}
                };
                writer->Begin();
                writer->WriteHeader("Content-Type", "application/json");
                writer->WriteBody(error_resp.dump());
                writer->End();
                return;
            }
            
            // Extract req_body
            if (!req_data.contains("req_body")) {
                json error_resp = {
                    {"ret_type", "get_point_cloud_detect_response"},
                    {"ret_header", {
                        {"code", 1},
                        {"message", "Missing req_body"}
                    }},
                    {"ret_body", json::object()}
                };
                writer->Begin();
                writer->WriteHeader("Content-Type", "application/json");
                writer->WriteBody(error_resp.dump());
                writer->End();
                return;
            }
            
            json req_body = req_data["req_body"];
            std::string unique_id = req_body.value("unique_id", "");
            int road_id = req_body.value("channel", 0);
            
            if (unique_id.empty()) {
                json error_resp = {
                    {"ret_type", "get_point_cloud_detect_response"},
                    {"ret_header", {
                        {"code", 2},
                        {"message", "Missing unique_id"}
                    }},
                    {"ret_body", json::object()}
                };
                writer->Begin();
                writer->WriteHeader("Content-Type", "application/json");
                writer->WriteBody(error_resp.dump());
                writer->End();
                return;
            }
            
            // 异步执行耗时推理，完成后再回包
            hv::async([detector, writer, unique_id, road_id, timestamp_str]() {
                DetectionResult result;
                if (detector) {
                    result = detector(unique_id, road_id);
                } else {
                    result = {};
                }
                
                json success_resp = {
                    {"ret_type", "get_point_cloud_detect_response"},
                    {"ret_header", {{"code", 0}}},
                    {"ret_body", {
                        {"PointCloudsMessage", {
                            {"unique_id", unique_id},
                            {"length", result.length*1000},
                            {"width", result.width*1000},
                            {"height", result.height*1000},
                            {"centre_length", result.centre_length*1000},
                            {"centre_width", result.centre_width*1000},
                            {"centre_height", result.centre_height*1000},
                            {"speed", result.speed * 3.6f},
                            {"score", result.score},
                            {"coordinate_system", "ECEF"},
                            {"sensor_type", "LiDAR"},
                            {"timestamp", timestamp_str}
                        }}
                    }}
                };
                write_log(success_resp.dump());
                writer->Begin();
                writer->WriteHeader("Content-Type", "application/json");
                writer->WriteBody(success_resp.dump());
                writer->End();
            });
            
        } catch (const json::parse_error& e) {
            json error_resp = {
                {"ret_type", "get_point_cloud_detect_response"},
                {"ret_header", {
                    {"code", 500},
                    {"message", "JSON parse error"}
                }},
                {"ret_body", json::object()}
            };
            writer->Begin();
            writer->WriteHeader("Content-Type", "application/json");
            writer->WriteBody(error_resp.dump());
            writer->End();
        } catch (const std::exception& e) {
            std::ostringstream forward_msg;
            forward_msg << "Error:" << body << " Error processing request: " << e.what();
            write_log(forward_msg.str());
            json error_resp = {
                {"ret_type", "get_point_cloud_detect_response"},
                {"ret_header", {
                    {"code", 500},
                    {"message", "Internal Server Error"}
                }},
                {"ret_body", json::object()}
            };
            writer->Begin();
            writer->WriteHeader("Content-Type", "application/json");
            writer->WriteBody(error_resp.dump());
            writer->End();
        }
    });
    
    // 设置服务器地址和端口
    server.setHost(host.c_str());
    server.setPort(port);
    
    std::cout << "[INFO] Starting HTTP server on " << host << ":" << port << std::endl;
    
    try {
        // 运行服务器（阻塞）
        server.run();
    } catch (const std::exception& e) {
        std::cerr << "HTTP Server error: " << e.what() << std::endl;
        if (running) {
            *running = false;
        }
    }
    
    std::cout << "HTTP Server stopped." << std::endl;
}

// 辅助函数：从 unique_id（纳秒时间戳）或雪花算法ID构建文件路径
static std::string build_file_path_from_unique_id(const std::string& unique_id, const std::string& base_path, bool is_car_file, bool use_snowflake = false) {
    try {
        std::string file_id;
        std::time_t time_t;
        
        if (use_snowflake) {
            // 使用雪花算法生成ID
            int64_t snowflake_id = get_snowflake().nextId();
            file_id = std::to_string(snowflake_id);
            
            // 从雪花算法ID中提取时间戳（雪花算法ID包含时间戳信息）
            // 雪花算法ID格式：时间戳(41位) + 数据中心ID(5位) + 机器ID(5位) + 序列号(12位)
            // 时间戳在最高位，需要右移22位（5+5+12）
            // 雪花算法的epoch是1577836800000LL (2020-01-01 00:00:00 UTC)
            int64_t timestamp_ms = (snowflake_id >> 22) + 1577836800000LL; // 加上epoch
            time_t = static_cast<std::time_t>(timestamp_ms / 1000);
        } else {
            // 使用unique_id（纳秒时间戳）
            file_id = unique_id;
            uint64_t timestamp_ns = std::stoull(unique_id);
            uint64_t timestamp_s = timestamp_ns / 1000000000ULL;  // 转换为秒
            time_t = static_cast<std::time_t>(timestamp_s);
        }
        
        // 转换为时间结构
        std::tm* timeinfo = std::localtime(&time_t);
        
        // 构建目录路径：YYYYMMDD/HH/MM
        std::ostringstream dir_path;
        dir_path << base_path;
        if (base_path.back() != '/') {
            dir_path << "/";
        }
        dir_path << std::setfill('0') << std::setw(4) << (1900 + timeinfo->tm_year)
                 << std::setfill('0') << std::setw(2) << (timeinfo->tm_mon + 1)
                 << std::setfill('0') << std::setw(2) << timeinfo->tm_mday << "/"
                 << std::setfill('0') << std::setw(2) << timeinfo->tm_hour << "/"
                 << std::setfill('0') << std::setw(2) << timeinfo->tm_min << "/";
        
        // 构建文件路径
        std::ostringstream file_path;
        file_path << dir_path.str() << file_id;
        if (is_car_file) {
            file_path << "_car.json";
        } else {
            file_path << ".json";
        }
        
        return file_path.str();
    } catch (const std::exception& e) {
        // 如果转换失败，返回空字符串
        std::cerr << "[ERROR] build_file_path_from_unique_id failed: " << e.what() << std::endl;
        return "";
    }
}

// 辅助函数：创建目录（如果不存在），支持多级目录
static bool create_directory_if_not_exists(const std::string& file_path) {
    try {
        // 提取目录路径（去掉文件名）
        size_t last_slash = file_path.find_last_of('/');
        if (last_slash == std::string::npos) {
            return true; // 没有目录部分，直接返回
        }
        
        std::string dir = file_path.substr(0, last_slash);
        
        // 如果目录已存在，直接返回
        struct stat info;
        if (stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
            return true;
        }
        
        // 递归创建目录
        // 从根目录开始逐级创建
        std::string current_path;
        size_t pos = 0;
        
        // 处理绝对路径（以/开头）
        if (dir[0] == '/') {
            current_path = "/";
            pos = 1;
        }
        
        while (pos < dir.length()) {
            size_t next_slash = dir.find('/', pos);
            if (next_slash == std::string::npos) {
                current_path += dir.substr(pos);
            } else {
                current_path += dir.substr(pos, next_slash - pos + 1);
            }
            
            // 检查当前路径是否存在
            if (stat(current_path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
                // 目录不存在，创建它
                if (mkdir(current_path.c_str(), 0755) != 0) {
                    // 如果创建失败且不是已存在的错误，返回false
                    if (errno != EEXIST) {
                        std::cerr << "[ERROR] Failed to create directory: " << current_path 
                                  << ", error: " << strerror(errno) << std::endl;
                        return false;
                    }
                }
            }
            
            if (next_slash == std::string::npos) {
                break;
            }
            pos = next_slash + 1;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to create directory: " << e.what() << std::endl;
        return false;
    }
}

// 辅助函数：保存点云数据到JSON文件，保留3位小数
static bool save_point_cloud_to_file(const std::string& file_path, 
                                     const std::vector<std::array<float, 4>>& points) {
    try {
        // 创建目录（如果不存在）
        if (!create_directory_if_not_exists(file_path)) {
            std::cerr << "[ERROR] Failed to create directory for: " << file_path << std::endl;
            return false;
        }
        
        // 写入文件
        std::ofstream ofs(file_path);
        if (!ofs.is_open()) {
            std::cerr << "[ERROR] Failed to open file for writing: " << file_path << std::endl;
            return false;
        }
        
        // 设置输出流精度为3位小数
        ofs << std::fixed << std::setprecision(3);
        
        // 手动构建紧凑格式的JSON数组，确保每个数字都是3位小数
        ofs << "[";
        for (size_t i = 0; i < points.size(); ++i) {
            const auto& p = points[i];
            // 保留3位小数：先乘以1000，四舍五入，再除以1000
            float x = std::round(p[0] * 1000.0f) / 1000.0f;
            float y = std::round(p[1] * 1000.0f) / 1000.0f;
            float z = std::round(p[2] * 1000.0f) / 1000.0f;
            float intensity = std::round(p[3] * 1000.0f) / 1000.0f;
            
            if (i > 0) {
                ofs << ",";
            }
            ofs << "[" << x << "," << y << "," << z << "," << intensity << "]";
        }
        ofs << "]" << std::endl;
        ofs.close();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to save point cloud to file: " << e.what() << std::endl;
        return false;
    }
}

void async_forward_to_other_service(const std::string& unique_id,
                                   const tracking::MultiObjectTracker::BestResult& best,
                                   std::vector<std::array<float, 4>> &point_cloud,
                                   std::vector<std::array<float, 4>> &points_max_car,
                                   int road_id, const std::string &lidar_tpye) {
    std::thread([unique_id, best, point_cloud = std::move(point_cloud), points_max_car = std::move(points_max_car), road_id, lidar_tpye]() mutable {
        const auto& config = get_config();
        int data_points_type = config.data_points_type;
        std::string points_file_path = config.points_file_path;
        
        nlohmann::json payload;
        double factor = 1000;
        payload["vehicle_width"] = std::round(best.width * factor);
        payload["vehicle_height"] = std::round(best.height * factor);
        payload["vehicle_length"] = std::round(best.length * factor);
        payload["vehicle_centre_width"] = std::round(best.centre_w * factor);
        payload["vehicle_centre_height"] = std::round(best.centre_h * factor);
        payload["vehicle_centre_length"] = std::round(best.centre_l * factor);
        payload["vehicle_score"] = std::round(best.score * factor) / factor;
        payload["vehicle_speed"] = std::round(best.speed * 3.6f * factor) / factor;
        payload["vehicle_serial_number"] = unique_id;
        payload["vehicle_lidar_type"] = lidar_tpye;
        payload["vehicle_detect_time"] = "";
        payload["data_points_type"] = data_points_type;
        
        //write_log(payload.dump());
        // 根据 data_points_type 决定返回点云数据还是路径
        if (data_points_type == 0) {
            // data_points_type 为 0：返回路径，点云数据为空数组，并保存点云数据到本地文件
            // 使用雪花算法生成ID
            std::string vehicle_car_points_path = build_file_path_from_unique_id(unique_id, points_file_path, true, true);
            std::string vehicle_radar_points_path = build_file_path_from_unique_id(unique_id, points_file_path, false, true);
            payload["vehicle_car_points_path"] = vehicle_car_points_path;
            payload["vehicle_radar_points_path"] = vehicle_radar_points_path;
            
            // 保存点云数据到本地文件（保留3位小数）
            save_point_cloud_to_file(vehicle_car_points_path, points_max_car);
            save_point_cloud_to_file(vehicle_radar_points_path, point_cloud);
            
            // 点云数据返回空数组
            payload["vehicle_radar_points"] = nlohmann::json::array();
            payload["vehicle_car_points"] = nlohmann::json::array();
        } else {
            // data_points_type 为 1：返回真实点云数据，路径为空
            payload["vehicle_car_points_path"] = "";
            payload["vehicle_radar_points_path"] = "";
            
            // 全图点云
            payload["vehicle_radar_points"] = nlohmann::json::array();
            for (const auto &p : point_cloud) {
                payload["vehicle_radar_points"].push_back(
                    {std::round(p[0] * factor) / factor, std::round(p[1] * factor) / factor, 
                    std::round(p[2] * factor) / factor, std::round(p[3] * factor) / factor}
                );
            }
            // 最大单车点云
            payload["vehicle_car_points"] = nlohmann::json::array();
            for (const auto &p : points_max_car) {
                payload["vehicle_car_points"].push_back(
                    {std::round(p[0] * factor) / factor, std::round(p[1] * factor) / factor, 
                    std::round(p[2] * factor) / factor, std::round(p[3] * factor) / factor}
                );
            }
        }

        std::string body = payload.dump();

        const std::string& web_ip = config.web_service_ip;
        const int web_port = config.web_service_port;
        const int timeout_sec = config.web_service_timeout_sec;
        const std::string web_url = "http://" + web_ip + ":" + std::to_string(web_port);

        const int max_attempts = 2;
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            const bool is_last_attempt = (attempt + 1) == max_attempts;
            try {
                HttpClient client(web_ip.c_str(), web_port);
                client.setTimeout(timeout_sec);

                HttpRequest req;
                req.method = HTTP_POST;
                req.url = web_url + "/api/ocm?type=3";
                req.SetBody(body);
                req.SetHeader("Content-Type", "application/json");

                HttpResponse resp;
                int ret = client.send(&req, &resp);
                if (ret == 0) {
                    std::ostringstream forward_msg;
                    forward_msg << "[FORWARD] Sent to " << web_url
                                << " (unique_id=" << unique_id << ") - status: " << resp.status_code
                                << " body: " << resp.body;
                    write_log(forward_msg.str());
                    break;
                } else {
                    if (!is_last_attempt) {
                        std::ostringstream timeout_msg;
                        timeout_msg << "[FORWARD] Timeout sending to " << web_url
                                     << " (unique_id=" << unique_id << ") - retrying ("
                                     << (max_attempts - attempt - 1) << " attempts left)";
                        write_log(timeout_msg.str());
                    } else {
                        std::ostringstream timeout_msg2;
                        timeout_msg2 << "[FORWARD] Timeout sending to " << web_url
                                      << " (unique_id=" << unique_id << ") - max retries reached";
                        write_log(timeout_msg2.str());
                    }
                }
            } catch (const std::exception &e) {
                if (!is_last_attempt) {
                    std::ostringstream exception_msg;
                    exception_msg << "[FORWARD] Exception while sending (retrying, "
                                  << (max_attempts - attempt - 1) << " attempts left): "
                                  << e.what();
                    write_log(exception_msg.str());
                } else {
                    std::ostringstream exception_msg2;
                    exception_msg2 << "[FORWARD] Exception while sending (max retries reached): "
                                   << e.what();
                    write_log(exception_msg2.str());
                }
            }
        }
    }).detach();
}

} // namespace http_server
