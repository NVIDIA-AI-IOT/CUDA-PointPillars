#ifndef INNO_SDK_WRAPPER_H
#define INNO_SDK_WRAPPER_H

#include <string>
#include <vector>
#include <array>
#include <cstdint>

namespace InnoSDK {

// Point data structure
struct PointData {
    float x, y, z;
    uint8_t intensity;
    uint64_t timestamp;
};

// Initialize the Inno SDK in a background thread
// @param lidar_ip: IP address of the lidar or file path (if ends with ".inno_pc", treated as file path)
// @param roll: Roll angle in radians
// @param pitch: Pitch angle in radians
// @param yaw: Yaw angle in radians
// @param x: X translation (default: 0)
// @param y: Y translation (default: 0)
// @param z: Z translation (default: 0)
// @return: 0 on success, -1 on failure
int InitInnoSDK(const std::string& lidar_ip, float roll, float pitch, float yaw, float x = 0, float y = 0, float z = 0);

// Stop the SDK cleanly
void UninitInnoSDK();

// Get point cloud data from the SDK
// @return: vector of PointData, empty if no data available
std::vector<PointData> GetPointCloudData();

// Check if SDK is running
bool IsSDKRunning();

/**
 * @brief 点云数据保存器，用于将点云数据保存为 .inno_pc 格式
 */
class InnoPcRecorder {
public:
    /**
     * @brief 构造函数
     * @param max_frame_number 最大保存帧数
     * @param output_filename 输出文件名（.inno_pc 格式）
     */
    InnoPcRecorder(int64_t max_frame_number, const std::string& output_filename);
    
    /**
     * @brief 析构函数
     */
    ~InnoPcRecorder() = default;
    
    /**
     * @brief 保存点云数据到文件（追加模式）
     * @param rendered_points 点云数据，每个点为 [x, y, z, intensity]
     *                        注意：intensity 应该是归一化的值（0.0-1.0），会自动转换为 1-254 范围
     * @return true 保存成功，false 保存失败（点云为空或已达到最大帧数）
     */
    bool save_pointcloud_to_inno_pc(
        const std::vector<std::array<float, 4>>& rendered_points
    );

private:
    int64_t max_frame_number_;      // 最大保存帧数
    int64_t current_frame_count_;   // 当前已保存的帧数
    std::string output_filename_;   // 输出文件名
    bool file_initialized_;         // 文件是否已初始化（第一帧是否已写入）
};

} // namespace InnoSDK

#endif // INNO_SDK_WRAPPER_H

