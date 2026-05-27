#include "hsai_sdk_wrapper.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>

using namespace HesaiSDK;

int main() {
    // Initialize SDK with lidar IP and transformation angles
    std::string lidar_ip = "10.31.10.28";
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    
    std::cout << "Initializing Hesai SDK..." << std::endl;
    int ret = InitHesaiSDK(lidar_ip, roll, pitch, yaw, x, y, z);
    
    if (ret != 0) {
        std::cerr << "Failed to initialize SDK" << std::endl;
        return -1;
    }
    
    // Wait for SDK to initialize
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "SDK initialized. Starting to receive point clouds..." << std::endl;
    
    // Main loop to get point clouds
    for (int i = 0; i < 100; ++i) {
        std::vector<PointData> points = GetPointCloudData();
        
        if (!points.empty()) {
            std::cout << "Frame " << i << ": Got " << points.size() << " points" << std::endl;
            
            // Access point data
            // Each PointData contains: x, y, z, intensity, timestamp
            
            // Example: print first point
            if (!points.empty()) {
                std::cout << "  First point: (" 
                          << points[0].x << ", " 
                          << points[0].y << ", " 
                          << points[0].z << ")"
                          << " intensity: " << (int)points[0].intensity
                          << " timestamp: " << points[0].timestamp
                          << std::endl;
            }
        } else {
            std::cout << "Frame " << i << ": No data available" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Clean up
    std::cout << "Stopping SDK..." << std::endl;
    UninitHesaiSDK();
    
    std::cout << "Done." << std::endl;
    return 0;
}

// Example 2: Accessing points individually
void process_points(const std::vector<PointData>& points) {
    // Iterate through all points
    for (const auto& point : points) {
        float x = point.x;
        float y = point.y;
        float z = point.z;
        uint8_t intensity = point.intensity;
        uint64_t timestamp = point.timestamp;
        
        // Process the point...
        // Do something with x, y, z, intensity, timestamp
        (void)x; (void)y; (void)z; (void)intensity; (void)timestamp; // Suppress unused warnings
    }
}

// Example 3: Convert to structured format
struct XYZPoint {
    float x, y, z;
    uint8_t intensity;
};

std::vector<XYZPoint> convert_to_xyz(const std::vector<PointData>& point_data) {
    std::vector<XYZPoint> points;
    points.reserve(point_data.size());
    
    for (const auto& p : point_data) {
        points.push_back({
            p.x,
            p.y,
            p.z,
            p.intensity
        });
    }
    
    return points;
}

