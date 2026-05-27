#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#include <string>
#include <functional>
#include <vector>
#include <array>
#include "tracking.hpp"

namespace http_server {

// Detection result structure
struct DetectionResult {
    float length, width, height ;
    float centre_length, centre_width, centre_height;
    float speed;
    float score;
};

// Detector function type: (unique_id, road_id) -> DetectionResult
using DetectorFunc = std::function<DetectionResult(const std::string&, int)>;

// Start the HTTP server
// @param host: Host address to bind to (default: "0.0.0.0")
// @param port: Port number to listen on (default: 8100)
// @param detector: Function to perform detection
// @param running: Reference to running flag to control server lifecycle
void start_pointcloud_server(const std::string& host = "0.0.0.0", 
                             int port = 8100,
                             DetectorFunc detector = nullptr,
                             bool* running = nullptr);

// Forward detection result to other service asynchronously
void async_forward_to_other_service(const std::string& unique_id,
                                   const tracking::MultiObjectTracker::BestResult& best,
                                   std::vector<std::array<float, 4>> &point_cloud,
                                   std::vector<std::array<float, 4>> &points_max_car,
                                   int road_id ,const std::string &lidar_tpye);

} // namespace http_server

#endif // HTTP_SERVER_HPP

