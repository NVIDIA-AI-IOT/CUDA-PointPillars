#include "inno_sdk_wrapper.hpp"
#include "src/sdk_common/inno_lidar_api.h"
#include "src/sdk_common/inno_lidar_other_api.h"
#include "src/sdk_common/inno_lidar_packet_utils.h"
#include "src/sdk_common/inno_lidar_packet.h"
#include "src/utils/inno_lidar_log.h"

#include <mutex>
#include <thread>
#include <chrono>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>

namespace InnoSDK {

// Global variables
static std::mutex queue_mutex;
static std::vector<PointData> pointcloud_queue;
static bool sdk_running = false;
static std::thread inno_thread;
static int lidar_handle = -1;

// Transform parameters
struct TransformParam {
    bool use_flag = false;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
} transform_global;

// Transform point coordinates
void TransformPoint(float& x, float& y, float& z, const TransformParam& transform) {
    if (transform.use_flag == false) return;

    float cosa = std::cos(transform.roll);
    float sina = std::sin(transform.roll);
    float cosb = std::cos(transform.pitch);
    float sinb = std::sin(transform.pitch);
    float cosc = std::cos(transform.yaw);
    float sinc = std::sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;
    x = x_;
    y = y_;
    z = z_;
}

// Internal point data structure
struct InternalPointData {
    float x, y, z;
    uint16_t reflectivity;
    uint8_t tag;
    uint64_t timestamp;
};

static const double kUsInSecond = 1000.0;
static const double k10UsInSecond = 100.0;

// Callback processor class
class CallbackProcessor {
public:
    explicit CallbackProcessor() {
        frame_data_.reserve(500000);
        current_frame_ = -1;
    }

    void set_done() { done_ = true; }
    bool is_done() {
        if (sdk_running == false)
            done_ = false;
        return done_;
    }

    void process_message(const enum InnoMessageLevel level, const enum InnoMessageCode code, const char *error_message) {
        if ((level == INNO_MESSAGE_LEVEL_INFO && code == INNO_MESSAGE_CODE_READ_FILE_END) ||
            (level == INNO_MESSAGE_LEVEL_CRITICAL && code == INNO_MESSAGE_CODE_CANNOT_READ)) {
            this->set_done();
        }

        switch (level) {
            case INNO_MESSAGE_LEVEL_ERROR:
            case INNO_MESSAGE_LEVEL_FATAL:
            case INNO_MESSAGE_LEVEL_CRITICAL:
                printf("error_content = %s\n", error_message);
                break;
            default:
                break;
        }
    }

    int process_data(int handler, const InnoDataPacket &pkt) {
        // Iterate each point in the packet
        for (int i = 0; i < pkt.item_number; i++) {
            if (this->current_frame_ != pkt.idx) {
                // Deliver the frame data to user and then clear the buffer
                std::lock_guard<std::mutex> lock(queue_mutex);
                pointcloud_queue.clear();

                // Transform points and convert to PointData format
                for (size_t j = 0; j < this->frame_data_.size(); j++) {
                    TransformPoint(this->frame_data_[j].x, this->frame_data_[j].y, this->frame_data_[j].z, transform_global);
                    
                    pointcloud_queue.push_back(PointData{
                        this->frame_data_[j].x,
                        this->frame_data_[j].y,
                        this->frame_data_[j].z,
                        static_cast<uint8_t>(this->frame_data_[j].reflectivity),
                        this->frame_data_[j].timestamp
                    });
                }

                this->frame_data_.clear();
                this->current_frame_ = pkt.idx;
            }

            InnoEnXyzPoint *point = reinterpret_cast<InnoEnXyzPoint *>(const_cast<char *>(pkt.payload));
            
            // Skip zero points
            if (point[i].x == 0.0 && point[i].y == 0.0 && point[i].z == 0.0)
                continue;

            // Use current system time in microseconds (CST = UTC+8)
            auto now = std::chrono::system_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
            // Add 8 hours (8 * 3600 * 1000000 us) to convert UTC to CST
            uint64_t timestamp = static_cast<uint64_t>(us) + 8ULL * 3600 * 1000000;
            
            this->frame_data_.push_back(InternalPointData{
                point[i].x, point[i].y, point[i].z,
                point[i].reflectance,
                static_cast<uint8_t>(pkt.idx),
                timestamp
            });
        }

        return 0;
    }

    int process_status(const InnoStatusPacket &pkt) {
        // Sanity check
        if (!inno_lidar_check_status_packet(&pkt, 0)) {
            printf("corrupted pkt->idx = %" PRI_SIZEU, pkt.idx);
            return 0;
        }
        return 0;
    }

private:
    volatile bool done_ = false;
    int64_t current_frame_ = -1;
    std::vector<InternalPointData> frame_data_;
};

int sdk_main_loop(const std::string &lidar_ip_address, float roll, float pitch, float yaw, float x, float y, float z) {
    printf("* inno_api_version %s\n", inno_api_version());
    printf("* lidar_ip/file:%s\n", lidar_ip_address.c_str());
    printf("* roll:%.4f\n", roll);
    printf("* pitch:%.4f\n", pitch);
    printf("* yaw:%.4f\n", yaw);
    printf("* x:%.4f\n", x);
    printf("* y:%.4f\n", y);
    printf("* z:%.4f\n", z);

    transform_global.use_flag = true;
    transform_global.roll = roll;
    transform_global.pitch = pitch;
    transform_global.yaw = yaw;
    transform_global.x = x;
    transform_global.y = y;
    transform_global.z = z;

    CallbackProcessor processor;

    // Check if lidar_ip_address ends with ".inno_pc" to determine if it's a file path
    bool is_file = (lidar_ip_address.length() >= 7 && 
                    lidar_ip_address.substr(lidar_ip_address.length() - 7) == ".inno_pc");

    // Open lidar connection (file or live)
    if (is_file) {
        // Open file for playback
        lidar_handle = inno_lidar_open_file("file_playback", lidar_ip_address.c_str(), false, 1, 1, 0);
        if (lidar_handle < 0) {
            printf("Failed to open lidar file: %s\n", lidar_ip_address.c_str());
            sdk_running = false;
            return -1;
        }
        printf("* Reading from file: %s\n", lidar_ip_address.c_str());
    } else {
        // Open live lidar connection
        lidar_handle = inno_lidar_open_live("live",  // name of lidar instance
                                            lidar_ip_address.c_str(), 8010, INNO_LIDAR_PROTOCOL_PCS_UDP,
                                            8010);
        if (lidar_handle < 0) {
            printf("Failed to open lidar connection\n");
            sdk_running = false;
            return -1;
        }
    }

    // Set to XYZ point cloud output
    inno_lidar_set_attribute_string(lidar_handle, "force_xyz_pointcloud", "1");

    // Set reflectance mode to INTENSITY
    int ret_reflectance = inno_lidar_set_reflectance_mode(lidar_handle, INNO_REFLECTANCE_MODE_INTENSITY);
    if (ret_reflectance != 0) {
        printf("Warning: Failed to set reflectance mode to INTENSITY, error code: %d\n", ret_reflectance);
    } else {
        printf("* Reflectance mode set to INTENSITY\n");
    }

    // Register callback functions
    int ret = inno_lidar_set_callbacks(
        lidar_handle,
        [](const int lidar_handle, void *ctx, const uint32_t from_remote, const enum InnoMessageLevel level,
           const enum InnoMessageCode code, const char *error_message) {
            return reinterpret_cast<CallbackProcessor *>(ctx)->process_message(level, code, error_message);
        },
        [](const int lidar_handle, void *ctx, const InnoDataPacket *pkt) -> int {
            return reinterpret_cast<CallbackProcessor *>(ctx)->process_data(lidar_handle, *pkt);
        },
        [](const int lidar_handle, void *ctx, const InnoStatusPacket *pkt) -> int {
            return reinterpret_cast<CallbackProcessor *>(ctx)->process_status(*pkt);
        },
        NULL, &processor);

    if (ret != 0) {
        printf("Failed to set callbacks\n");
        inno_lidar_close(lidar_handle);
        lidar_handle = -1;
        sdk_running = false;
        return -1;
    }

    sdk_running = true;

    // Start lidar
    ret = inno_lidar_start(lidar_handle);
    if (ret != 0) {
        printf("Failed to start lidar\n");
        inno_lidar_close(lidar_handle);
        lidar_handle = -1;
        sdk_running = false;
        return -1;
    }

    // Main loop
    while (!processor.is_done() && sdk_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup
    if (lidar_handle >= 0) {
        inno_lidar_stop(lidar_handle);
        inno_lidar_close(lidar_handle);
        lidar_handle = -1;
    }

    printf("sdk end-------------\n");
    sdk_running = false;
    return 0;
}

int InitInnoSDK(const std::string& lidar_ip, float roll, float pitch, float yaw, float x, float y, float z) {
    // Start SDK thread, separate from caller
    printf("* Initializing Inno SDK\n");
    printf("* lidar_ip/file:%s\n", lidar_ip.c_str());
    printf("* roll:%.4f\n", roll);
    printf("* pitch:%.4f\n", pitch);
    printf("* yaw:%.4f\n", yaw);
    printf("* x:%.4f\n", x);
    printf("* y:%.4f\n", y);
    printf("* z:%.4f\n", z);

    inno_thread = std::thread(sdk_main_loop, lidar_ip, roll, pitch, yaw, x, y, z);
    return 0;
}

void UninitInnoSDK() {
    sdk_running = false;
    if (lidar_handle >= 0) {
        inno_lidar_stop(lidar_handle);
        inno_lidar_close(lidar_handle);
        lidar_handle = -1;
    }
    if (inno_thread.joinable()) {
        inno_thread.join();
    }
}

std::vector<PointData> GetPointCloudData() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    
    std::vector<PointData> frame = std::move(pointcloud_queue);
    pointcloud_queue.clear();
    
    return frame;
}

bool IsSDKRunning() {
    return sdk_running;
}

/**
 * @brief 计算点到原点的距离（半径）
 */
static float calculate_radius(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

/**
 * @brief 构造函数
 */
InnoPcRecorder::InnoPcRecorder(int64_t max_frame_number, const std::string& output_filename)
    : max_frame_number_(max_frame_number)
    , current_frame_count_(0)
    , output_filename_(output_filename)
    , file_initialized_(false) {
}

/**
 * @brief 保存点云数据为 .inno_pc 格式（追加模式）
 */
bool InnoPcRecorder::save_pointcloud_to_inno_pc(
    const std::vector<std::array<float, 4>>& rendered_points) {
    
    // 检查点云是否为空
    if (rendered_points.empty()) {
        inno_log_warning("Empty point cloud, nothing to save");
        return false;
    }
    
    // 检查是否已达到最大帧数
    if (current_frame_count_ >= max_frame_number_) {
        inno_log_warning("Reached maximum frame number %" PRI_SIZELD ", cannot save more frames",
                        max_frame_number_);
        return false;
    }

    // 计算数据包大小
    const uint32_t point_count = static_cast<uint32_t>(rendered_points.size());
    const size_t xyz_point_size = sizeof(InnoXyzPoint);
    const size_t packet_size = sizeof(InnoDataPacket) + point_count * xyz_point_size;
    
    // 分配内存
    char* packet_buffer = reinterpret_cast<char*>(malloc(packet_size));
    if (!packet_buffer) {
        inno_log_error("Failed to allocate memory for packet");
        return false;
    }
    memset(packet_buffer, 0, packet_size);
    
    InnoDataPacket* packet = reinterpret_cast<InnoDataPacket*>(packet_buffer);
    
    // 填充 InnoCommonHeader
    packet->common.version.magic_number = kInnoMagicNumberDataPacket;  // 必须设置 magic number
    packet->common.version.major_version = 1;
    packet->common.version.minor_version = 0;
    packet->common.version.fw_sequence = 0;
    packet->common.checksum = 0;  // 可以后续计算
    packet->common.size = static_cast<uint32_t>(packet_size);
    packet->common.source_id = 0;
    packet->common.timestamp_sync_type = 0;
    packet->common.lidar_type = INNO_LIDAR_TYPE_FALCON;  // 根据实际情况调整
    packet->common.ts_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    packet->common.lidar_mode = INNO_LIDAR_MODE_WORK_NORMAL;
    packet->common.lidar_status = INNO_LIDAR_STATUS_NORMAL;
    
    // 填充 InnoDataPacket 字段
    packet->idx = static_cast<uint64_t>(current_frame_count_);
    packet->sub_idx = 0;
    packet->sub_seq = 0;
    packet->type = INNO_ITEM_TYPE_XYZ_POINTCLOUD;  // XYZ 点云格式
    packet->item_number = point_count;
    packet->item_size = static_cast<uint16_t>(xyz_point_size);
    packet->topic = 0;
    packet->scanner_direction = 0;
    packet->use_reflectance = 0;  // 使用 intensity
    packet->multi_return_mode = INNO_MULTIPLE_RETURN_MODE_SINGLE;
    packet->confidence_level = 3;  // 最高置信度
    packet->is_last_sub_frame = 1;
    packet->is_last_sequence = 1;
    packet->has_tail = 0;
    packet->frame_sync_locked = 1;
    packet->is_first_sub_frame = 1;
    packet->last_four_channel = 0;
    packet->long_distance_mode = 0;
    packet->reserved_flag = 0;
    packet->roi_h_angle = 0;
    packet->roi_v_angle = 0;
    memset(packet->extend_reserved, 0, sizeof(packet->extend_reserved));
    
    // 填充点云数据
    InnoXyzPoint* xyz_points = reinterpret_cast<InnoXyzPoint*>(packet->payload);
    for (uint32_t i = 0; i < point_count; ++i) {
        const auto& pt = rendered_points[i];
        InnoXyzPoint& xyz_pt = xyz_points[i];
        
        xyz_pt.x = pt[0];
        xyz_pt.y = pt[1];
        xyz_pt.z = pt[2];
        xyz_pt.radius = calculate_radius(pt[0], pt[1], pt[2]);
        xyz_pt.ts_10us = 0;  // 相对时间戳，可以设置为0
        xyz_pt.scan_id = 0;
        xyz_pt.in_roi = 1;
        xyz_pt.facet = 0;
        xyz_pt.multi_return = 0;
        xyz_pt.reserved_flags = 0;
        xyz_pt.is_2nd_return = 0;
        xyz_pt.scan_idx = i % 16384;  // 限制在14位范围内
        // intensity 是归一化的 (0.0-1.0)，需要转换为 1-254 范围（255 表示反射器）
        // 将归一化的 intensity 映射到 1-254: refl = 1 + intensity * 253
        float normalized_intensity = std::max(0.0f, std::min(1.0f, pt[3]));
        xyz_pt.refl = static_cast<uint32_t>(1 + normalized_intensity * 253.0f);
        xyz_pt.type = 0;  // normal point
        xyz_pt.elongation = 0;
        xyz_pt.channel = 0;
        xyz_pt.ring_id = 0;
    }
    
    // 验证数据包
    if (!InnoDataPacketUtils::check_data_packet(*packet, packet_size)) {
        inno_log_error("Invalid data packet constructed");
        free(packet_buffer);
        return false;
    }
    
    // 打开文件（追加模式，如果是第一帧则创建新文件）
    int fd = -1;
    bool is_first_frame = !file_initialized_;
    
    if (is_first_frame) {
        // 删除已存在的文件
        if (access(output_filename_.c_str(), F_OK) == 0) {
            unlink(output_filename_.c_str());
        }
        fd = open(output_filename_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        file_initialized_ = true;
    } else {
        fd = open(output_filename_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
    
    if (fd < 0) {
        inno_log_error("Failed to open file: %s", output_filename_.c_str());
        free(packet_buffer);
        return false;
    }
    
    // 写入数据包
    size_t written = 0;
    while (written < packet_size) {
        ssize_t ret = write(fd, packet_buffer + written, packet_size - written);
        if (ret < 0) {
            inno_log_error("Failed to write to file: %s", output_filename_.c_str());
            close(fd);
            free(packet_buffer);
            return false;
        }
        written += ret;
    }
    
    close(fd);
    free(packet_buffer);
    
    // 增加已保存的帧数
    current_frame_count_++;
    
    inno_log_info("Saved %u points to %s (frame %" PRI_SIZELD "/%" PRI_SIZELD ")", 
                  point_count, output_filename_.c_str(), 
                  current_frame_count_, max_frame_number_);
    
    return true;
}

} // namespace InnoSDK

