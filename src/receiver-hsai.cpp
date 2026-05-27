 #include <cuda_runtime.h>

#include <math.h>
#include <cmath>
#include <iostream>
#include <array>
#include <thread>
#include <mutex>
#include <chrono>
// #include <ctime>
// #include <iomanip>
// #include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <functional>
#include <sys/statvfs.h>
#include <dirent.h>
#include <cstring>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>

#include <nlohmann/json.hpp>
#include "common/check.hpp"
#include "hsai_sdk_wrapper.h"
#include "detect/draw_meshlab.hpp"
#include "detect/http_server.hpp"
#include "detect/process.hpp"
#include "thread_pool.hpp"
#include "detect/tracking.hpp"
#include "detect/config.hpp"
#include "pointpillar.hpp"

const double PI = 3.141592653589793;

// Thread-safe storage for latest point cloud data
std::vector<float> points_queue;
uint64_t time_queue = 0;
std::mutex queue_lock;
bool running = true;

thread_pool* pool = nullptr;
std::unordered_map<std::thread::id, std::shared_ptr<pointpillar::lidar::Core> > thread_algo_ptr;
cudaStream_t stream;
tracking::MultiObjectTracker* mot = nullptr;

void GetDeviceInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

void start_lidar()
{
    try {
        const auto& config = get_config();
        HesaiSDK::InitHesaiSDK(config.lidar_ip_address, config.lidar_roll, config.lidar_pitch, 
                               config.lidar_yaw, config.lidar_x, config.lidar_y, config.lidar_z);
        std::cout << "Hesai SDK started. " << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Hesai error: " << e.what() << std::endl;
    }
}

std::shared_ptr<pointpillar::lidar::Core> create_core() 
{
    pointpillar::lidar::VoxelizationParameter vp;
    vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
    vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
    vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
    vp.grid_size =
        vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 40000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);
    pp.nms_thresh = 0.1;

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = get_config().lidar_model;
    param.lidar_post = pp;
    auto core =  pointpillar::lidar::create_core(param);
    if (core == nullptr) {
        std::cerr << "Core creation failed in create_core." << std::endl;
        return nullptr;
    }
    core->set_timer(false);
    return core;
}

void loop_get_lidar_data() 
{
    std::cout << "Point cloud collection thread started." << std::endl;

    // Track last successful data reception time
    auto last_data_time = std::chrono::steady_clock::now();
    const auto timeout_duration = std::chrono::minutes(5); // 5 minutes timeout

    while (running) {
        try {
            // Check if 5 minutes have passed without receiving data
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_data_time);
            
            if (elapsed >= timeout_duration) {
                std::cerr << "Warning: No lidar data received for 5 minutes. Radar may be powered off. Reinitializing..." << std::endl;
                
                // Uninitialize SDK
                try {
                    HesaiSDK::UninitHesaiSDK();
                    std::cout << "HesaiSDK uninitialized." << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Error during UninitHesaiSDK: " << e.what() << std::endl;
                }
                
                // Wait a bit before reinitializing
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                // Reinitialize SDK
                try {
                    const auto& config = get_config();
                    HesaiSDK::InitHesaiSDK(config.lidar_ip_address, config.lidar_roll, config.lidar_pitch, 
                                           config.lidar_yaw, config.lidar_x, config.lidar_y, config.lidar_z);
                    std::cout << "HesaiSDK reinitialized successfully." << std::endl;
                    last_data_time = std::chrono::steady_clock::now(); // Reset timer
                } catch (const std::exception& e) {
                    std::cerr << "Error during InitHesaiSDK reinitialization: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(5)); // Wait longer before retry
                    continue;
                }
            }

            // Get point cloud from Hesai SDK
            std::vector<HesaiSDK::PointData> pc_data = HesaiSDK::GetPointCloudData();
            
            // Wait until we have enough points (similar to Python version)
            int attempts = 0; 
            while (pc_data.empty() || pc_data.size() < 20000) {
                if (attempts++ > 100) break; // Avoid infinite loop
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                pc_data = HesaiSDK::GetPointCloudData();
            }
            if (!pc_data.empty() && pc_data.size() >= 20000) {
                std::vector<float> points_xyzi;
                for (const auto& p : pc_data) 
                {
                    points_xyzi.push_back(p.x);
                    points_xyzi.push_back(p.y);
                    points_xyzi.push_back(p.z);
                    points_xyzi.push_back((float)p.intensity);
                }
                
                // Update latest point cloud data
                {
                    std::lock_guard<std::mutex> lock(queue_lock);
                    points_queue = std::move(points_xyzi);
                    time_queue = pc_data[0].timestamp;
                }
                
                last_data_time = std::chrono::steady_clock::now();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(45)); 
        } catch (const std::exception& e) {
            std::cerr << "Error in loop_get_lidar_data: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // Don't set running = false on error, let it retry
        }
    }

    std::cout << "Point cloud collection thread stopped." << std::endl;
}

// Detection function called by HTTP server
http_server::DetectionResult handle_detection_request(const std::string& unique_id, int road_id) 
{
   
    http_server::DetectionResult result;
    tracking::MultiObjectTracker::BestResult best={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    std::vector<std::array<float, 4>> rendered_points;
    bool flag =mot->set_unique_id_for_closest_vehicle(unique_id, road_id,rendered_points); //去设置unique_id
    // if( !flag )
    // {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(150));  //再给一次机会 
    //     flag = mot->set_unique_id_for_closest_vehicle(unique_id, road_id,rendered_points);
    // }
    if(flag)
    {
        int T = 10;   // 10*200ms = 2s
        while(mot->result_map_[unique_id].status_code != 1 && T)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            T--;
        }
        best = mot->result_map_[unique_id].result;
        mot->result_map_[unique_id].status_code = 2;
    }
    std::vector<std::array<float, 4>> points_max_car = std::move(best.points_max_car);
    result.length = best.length;
    result.width = best.width;
    result.height = best.height;
    result.centre_length = best.centre_l;
    result.centre_width = best.centre_w;
    result.centre_height = best.centre_h;
    result.speed = best.speed;
    result.score = best.score;
    http_server::async_forward_to_other_service(unique_id, best, rendered_points, points_max_car, road_id, "Hsai");
    return result;
}

int cnt_zser = 0 ;
// 检测 前处理->推理->后处理
void detect_task_lidar(std::vector<float> &points, std::vector<detect::ProcessingBox> &bboxes_result) 
{
    bboxes_result.clear();
    if (points.empty()) 
        return;
    std::vector<float> points_filtered;
    detect::pre_processing(points, points_filtered, get_config());

    std::thread::id id_ = std::this_thread::get_id();
    if (thread_algo_ptr[id_] == nullptr) {
        thread_algo_ptr[id_] = create_core();
        if (!thread_algo_ptr[id_]) {
            std::cerr << "Failed to create detector core" << std::endl;
            return;
        }
    }
    auto ptr = thread_algo_ptr[id_];
    
    if (points_filtered.empty()) 
        return;
    
    int points_size = points_filtered.size() / 4;
    std::vector<pointpillar::lidar::BoundingBox> bboxes = ptr->forward(points_filtered.data(), points_size, stream); 
    detect::post_processing(bboxes, bboxes_result, points_filtered);

    // 暂时不用在这儿画框
    // if(cnt_zser%200==0)
    // {
        // auto current_time = std::chrono::system_clock::now();
        // auto current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        //                         current_time.time_since_epoch())
        //                         .count();
        // std::string save_pcd_name = []{
        //     // std::chrono::system_clock::now() 返回的是 time_point，只包含从1970-01-01开始的时长
        //     // 要获取年月日，必须转换为日历时间（time_t -> tm结构体）
        //     auto now = std::chrono::system_clock::now();
        //     auto time_t = std::chrono::system_clock::to_time_t(now);
        //     std::tm tm = *std::localtime(&time_t);
            
        //     // 获取毫秒部分（time_point 可以精确到毫秒）
        //     auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        //         now.time_since_epoch()).count() % 1000;
            
        //     std::ostringstream filename_stream;
        //     filename_stream << "../train/" 
        //                     << std::setfill('0') << std::setw(4) << (1900 + tm.tm_year) << "-"
        //                     << std::setfill('0') << std::setw(2) << (tm.tm_mon + 1) << "-"
        //                     << std::setfill('0') << std::setw(2) << tm.tm_mday << "_"
        //                     << std::setfill('0') << std::setw(2) << tm.tm_hour << "_"
        //                     << std::setfill('0') << std::setw(2) << tm.tm_min << "_"
        //                     << std::setfill('0') << std::setw(2) << tm.tm_sec << "_"
        //                     << std::setfill('0') << std::setw(3) << millis << ".pcd";  
        //     return filename_stream.str();
        // }();
    // }
    // {// roi框
    //     const float cx = get_config().center_x;
    //     const float cy = get_config().center_y;
    //     const float cz = get_config().min_z;
    //     const float w  = get_config().range_x * 2.0f;
    //     const float l  = get_config().range_y * 2.0f;
    //     const float h  = get_config().range_z;
    //     detect::ProcessingBox range_box(cx, cy, cz, w, l, h, 0.0f, -1.0, 1.0f);
    //     bboxes.push_back(range_box);
    // }
    // std::vector<std::array<float, 4>> rendered_points;
    // detect::SaveBoxesAsPCD(bboxes_result, points_filtered.data(), points_filtered.size()/4, save_pcd_name, get_config().point_cloud_draw_step, rendered_points);
    // cnt_zser++;
}

// 雷达实时检测
void point_cloud_detect() {
    while (running) {
        try {
            // Get latest point cloud data
            std::vector<float> points;
            uint64_t time;
            {
                std::lock_guard<std::mutex> lock(queue_lock);
                if (!points_queue.empty()) 
                {
                    points = std::move(points_queue);   // 取最新的点云数据，move后队列自动清空
                    time = time_queue;
                }
            }
            if(points.empty()) 
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            
             //先一次跑全部车道 之后再看是不是每个车道开个线程去跑
            std::vector<detect::ProcessingBox> bboxes;
            auto result_pool = pool->enqueue(detect_task_lidar, std::ref(points), std::ref(bboxes));
            result_pool.get();
            
            // 计算车道id
            std::vector<tracking::BBox3D> detections_frame;
            std::vector<std::vector<std::array<float, 4>>> car_points_frame;
            for (auto& box : bboxes) {
                const float center_x = get_config().center_x;
                const float range_x  = get_config().range_x;
                const float max_x    = center_x + range_x;
                const int lane_count = std::max(1, get_config().lane_count);
                const float total_width = range_x * 2.0f;

                float road_id = 1.0;
                if (lane_count > 1 && total_width > 0.0f) {
                    const float lane_span = total_width / static_cast<float>(lane_count); 
                    float best_dist = std::numeric_limits<float>::max();
                    float best_lane = 1;
                    int cnt = 0;
                    for (float lane = 1; lane <= lane_count; lane+=0.5, cnt++) {
                        const float lane_center = max_x - (lane - 0.5f) * lane_span;
                        const float dist = std::fabs(box.x - lane_center);
                        if (cnt%2==0 && dist < best_dist) {
                            best_dist = dist;
                            best_lane = lane;
                        }
                        else if(cnt%2==1 && dist < 0.5 && dist < best_dist) {
                            best_dist = dist;
                            best_lane = lane;
                        }
                    }
                    road_id = best_lane;
                }

                //雷达坐标   
                // w-长 l-宽 h-高
                // y-前后 x-左右 z-上下
                //车前下点作为车中心
                detections_frame.emplace_back(
                     box.y-0.5*box.w, box.x, box.z, box.w, box.l, box.h, box.rt, road_id, box.score);
                car_points_frame.push_back(std::move(box.points));
            }

            // Run tracking on both frames
            mot->update(detections_frame,car_points_frame, time/1000, points);
            // {// roi框
            //     const float cx = get_config().center_x;
            //     const float cy = get_config().center_y;
            //     const float cz = get_config().min_z;
            //     const float w  = get_config().range_x * 2.0f;
            //     const float l  = get_config().range_y * 2.0f;
            //     const float h  = get_config().range_z;
            //     detect::ProcessingBox range_box(cx, cy, cz, w, l, h, 0.0f, -1.0, 1.0f);
            //     bboxes.push_back(range_box);
            // }
            
            // {//推理框
            //     const float cx = 34.56; 
            //     const float cy = 0;
            //     const float cz = -3;
            //     const float w  = 34.56 * 2.0f;
            //     const float l  = 39.680 * 2.0f;
            //     const float h  = 4;
            //     detect::ProcessingBox range_box(cx, cy, cz, w, l, h, 0.0f, -1.0, 1.0f);
            //     bboxes.push_back(range_box);
            // }

            // auto current_time = std::chrono::system_clock::now();
            // auto current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            //                       current_time.time_since_epoch())
            //                       .count();
            // std::string save_pcd_name =
            //     "../results/detection_" + std::to_string(time) + "_" + std::to_string(current_ms) + ".pcd";
            // std::vector<std::array<float, 4>> rendered_points;
            // detect::SaveBoxesAsPCD(bboxes, points.data(), points.size()/4, save_pcd_name, get_config().point_cloud_draw_step, rendered_points);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in handle_detection_request: " << e.what() << std::endl;
        }

    }
}

// 磁盘监控和清理函数
void disk_space_monitor_thread() {
    while (running) {
        try {
            const auto& config = get_config();
            std::string points_file_path = config.points_file_path;
            
            // 获取磁盘空间信息
            struct statvfs stat;
            if (statvfs(points_file_path.c_str(), &stat) == 0) {
                // 计算剩余空间百分比
                unsigned long long total_space = stat.f_blocks * stat.f_frsize;
                unsigned long long free_space = stat.f_bavail * stat.f_frsize;
                double free_percentage = (double)free_space / (double)total_space * 100.0;
                
                std::cout << "[DISK] Free space: " << std::fixed << std::setprecision(2) 
                          << free_percentage << "% (" << free_space / (1024ULL * 1024ULL * 1024ULL) 
                          << " GB / " << total_space / (1024ULL * 1024ULL * 1024ULL) << " GB)" << std::endl;
                
                // 如果剩余空间不足15%，开始清理
                if (free_percentage < 15.0) {
                    std::cout << "[DISK] Warning: Free space below 15%, starting cleanup..." << std::endl;
                    
                    // 获取所有 yyyyMMdd/HH/mm 格式的三级目录
                    std::vector<std::pair<std::string, std::time_t>> time_dirs;  // <路径, 时间戳>
                    
                    DIR* dir = opendir(points_file_path.c_str());
                    if (dir != nullptr) {
                        struct dirent* entry;
                        while ((entry = readdir(dir)) != nullptr) {
                            std::string date_dir_name = entry->d_name;
                            // 跳过 . 和 ..
                            if (date_dir_name == "." || date_dir_name == "..") {
                                continue;
                            }
                            
                            // 检查是否是YYYYMMDD格式的目录（8位数字）
                            if (date_dir_name.length() == 8) {
                                bool is_date_dir = true;
                                for (char c : date_dir_name) {
                                    if (!std::isdigit(c)) {
                                        is_date_dir = false;
                                        break;
                                    }
                                }
                                if (is_date_dir) {
                                    // 解析日期
                                    int year = std::stoi(date_dir_name.substr(0, 4));
                                    int month = std::stoi(date_dir_name.substr(4, 2));
                                    int day = std::stoi(date_dir_name.substr(6, 2));
                                    
                                    // 遍历该日期目录下的所有HH目录
                                    std::string date_path = points_file_path;
                                    if (points_file_path.back() != '/') {
                                        date_path += "/";
                                    }
                                    date_path += date_dir_name;
                                    
                                    DIR* date_dir = opendir(date_path.c_str());
                                    if (date_dir != nullptr) {
                                        struct dirent* hour_entry;
                                        while ((hour_entry = readdir(date_dir)) != nullptr) {
                                            std::string hour_dir_name = hour_entry->d_name;
                                            if (hour_dir_name == "." || hour_dir_name == "..") {
                                                continue;
                                            }
                                            
                                            // 检查是否是HH格式的目录（2位数字，00-23）
                                            if (hour_dir_name.length() == 2 && std::isdigit(hour_dir_name[0]) && std::isdigit(hour_dir_name[1])) {
                                                int hour = std::stoi(hour_dir_name);
                                                
                                                // 遍历该小时目录下的所有mm目录
                                                std::string hour_path = date_path + "/" + hour_dir_name;
                                                
                                                DIR* hour_dir = opendir(hour_path.c_str());
                                                if (hour_dir != nullptr) {
                                                    struct dirent* min_entry;
                                                    while ((min_entry = readdir(hour_dir)) != nullptr) {
                                                        std::string min_dir_name = min_entry->d_name;
                                                        if (min_dir_name == "." || min_dir_name == "..") {
                                                            continue;
                                                        }
                                                        
                                                        // 检查是否是mm格式的目录（2位数字，00-59）
                                                        if (min_dir_name.length() == 2 && std::isdigit(min_dir_name[0]) && std::isdigit(min_dir_name[1])) {
                                                            int minute = std::stoi(min_dir_name);
                                                            
                                                            // 构建完整路径
                                                            std::string full_path = hour_path + "/" + min_dir_name;
                                                            
                                                            // 构建时间戳用于排序
                                                            std::tm time_tm = {};
                                                            time_tm.tm_year = year - 1900;
                                                            time_tm.tm_mon = month - 1;
                                                            time_tm.tm_mday = day;
                                                            time_tm.tm_hour = hour;
                                                            time_tm.tm_min = minute;
                                                            time_tm.tm_sec = 0;
                                                            std::time_t dir_time = std::mktime(&time_tm);
                                                            
                                                            time_dirs.push_back({full_path, dir_time});
                                                        }
                                                    }
                                                    closedir(hour_dir);
                                                }
                                            }
                                        }
                                        closedir(date_dir);
                                    }
                                }
                            }
                        }
                        closedir(dir);
                        
                        // 按照时间戳排序（时间最靠前的在前面）
                        std::sort(time_dirs.begin(), time_dirs.end(), 
                                  [](const std::pair<std::string, std::time_t>& a, 
                                     const std::pair<std::string, std::time_t>& b) {
                                      return a.second < b.second;
                                  });
                        
                        // 从时间最靠前的目录开始删除
                        bool space_freed = false;
                        for (const auto& time_dir : time_dirs) {
                            // 删除整个时间目录（yyyyMMdd/HH/mm）
                            std::string cmd = "rm -rf \"" + time_dir.first + "\"";
                            int ret = system(cmd.c_str());
                            if (ret == 0) {
                                std::cout << "[DISK] Deleted directory: " << time_dir.first << std::endl;
                                space_freed = true;
                                
                                // 再次检查磁盘空间
                                struct statvfs stat_after;
                                if (statvfs(points_file_path.c_str(), &stat_after) == 0) {
                                    total_space = stat_after.f_blocks * stat_after.f_frsize;
                                    free_space = stat_after.f_bavail * stat_after.f_frsize;
                                    free_percentage = (double)free_space / (double)total_space * 100.0;
                                    stat = stat_after;  // 更新 stat 以供后续使用
                                    
                                    // 如果空间已足够，停止删除
                                    if (free_percentage >= 15.0) {
                                        std::cout << "[DISK] Free space restored to " 
                                                  << std::fixed << std::setprecision(2) 
                                                  << free_percentage << "%" << std::endl;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // 如果所有文件夹都删了还是不够，输出报警
                        if (!space_freed || free_percentage < 15.0) {
                            std::cerr << "[DISK] ALERT: All folders deleted but free space still below 15%! " 
                                      << "Current free space: " << std::fixed << std::setprecision(2) 
                                      << free_percentage << "%" << std::endl;
                        }
                    } else {
                        std::cerr << "[DISK] Error: Cannot open directory " << points_file_path << std::endl;
                    }
                }
            } else {
                std::cerr << "[DISK] Error: Cannot get disk space info for " << points_file_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[DISK] Error in disk space monitor: " << e.what() << std::endl;
        }
        
        // 每小时检测一次
        std::this_thread::sleep_for(std::chrono::hours(1));
    }
}

int main(int argc, char** argv) {
    // Set config file path before initializing config
    RangeConfigSingleton::setConfigFilePath("../config/config-hsai.json");
    
    // Construct MultiObjectTracker after config is initialized
    // Config is automatically initialized when getInstance() is first called
    mot = new tracking::MultiObjectTracker(0.75f, 5,
        tracking::DimensionStrategy::MEDIAN,
        tracking::SpeedStrategy::CLOSEST_3_TO_SPEED_LINE);
    
    pool = new thread_pool(1);
    cudaStreamCreate(&stream);
    GetDeviceInfo();
    //启动雷达
    start_lidar();
    //抓雷达数据线程
    std::thread get_lidar_thread(loop_get_lidar_data);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // 雷达检测线程
    std::thread detect_thread(point_cloud_detect);

    // HTTP服务器线程 接收AII请求并进行检测
    std::thread server_thread(http_server::start_pointcloud_server, "0.0.0.0", 8100, handle_detection_request, &running);
    
    // 磁盘监控线程
    std::thread disk_monitor_thread(disk_space_monitor_thread);
    
    // Keep main thread alive
    try {
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (...) {
        running = false;
    }
    
    // Wait for threads to finish
    get_lidar_thread.join();
    detect_thread.join();
    server_thread.join();
    disk_monitor_thread.join();
    
    // Stop lidar before exit
    HesaiSDK::UninitHesaiSDK();
    
    // Cleanup CUDA stream
    cudaStreamDestroy(stream);
    
    // Cleanup thread pool
    delete pool;
    
    // Cleanup MultiObjectTracker
    delete mot;
    
    return 0;
}
