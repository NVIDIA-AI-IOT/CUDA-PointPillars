#ifndef HSAI_SDK_WRAPPER_H
#define HSAI_SDK_WRAPPER_H

#include <string>
#include <vector>
#include <cstdint>

namespace HesaiSDK {

// Point data structure
struct PointData {
    float x, y, z;
    uint8_t intensity;
    uint64_t timestamp;
};

// Initialize the Hesai SDK in a background thread
// @param lidar_ip: IP address of the lidar
// @param roll: Roll angle in radians
// @param pitch: Pitch angle in radians
// @param yaw: Yaw angle in radians
// @return: 0 on success, -1 on failure
int InitHesaiSDK(const std::string& lidar_ip, float roll, float pitch, float yaw,float x=0,float y=0,float z=0);

// Stop the SDK cleanly
void UninitHesaiSDK();

// Get point cloud data from the SDK
// @return: vector of PointData, empty if no data available
std::vector<PointData> GetPointCloudData();

// Check if SDK is running
bool IsSDKRunning();

} // namespace HesaiSDK

#endif // HSAI_SDK_WRAPPER_H

