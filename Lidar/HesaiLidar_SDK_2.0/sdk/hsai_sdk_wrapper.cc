#include "hsai_sdk_wrapper.h"
#include "hesai_lidar_sdk.hpp"
#define LIDAR_PARSER_TEST

#include <mutex>
#include <thread>
#include <chrono>
#include <string>

namespace HesaiSDK {

// Global variables
static uint32_t last_frame_time = 0;
static uint32_t cur_frame_time = 0;
static std::mutex queue_mutex;
static std::vector<PointData> pointcloud_queue;
static bool sdk_running = false;
static std::thread hesai_thread;
static HesaiLidarSdk<LidarPointXYZIRT>* g_sample_ptr = nullptr;

//log info, display frame message    
void lidarCallback(const LidarDecodedFrame<LidarPointXYZIRT> &frame) { 
    cur_frame_time = GetMicroTickCount();
    last_frame_time = cur_frame_time;
    std::vector<PointData> frame_data;
    uint64_t timestamp_ = GetMicroTickCountU64();
    
    // printf("frame:%d points:%u ",frame.frame_index, frame.points_num); 
    // std::cout <<" timestamp_: "<< timestamp_<<"\n";
    
    for (uint32_t i = 0; i < frame.points_num; ++i) {
        if(frame.points[i].x == 0.0 && frame.points[i].y == 0.0 && frame.points[i].z == 0.0)
            continue;
            
        frame_data.push_back(PointData{
            frame.points[i].x, 
            frame.points[i].y, 
            frame.points[i].z,
            frame.points[i].intensity,
            timestamp_
        });
    }
    
    //printf("======  frame_data.size = %llu\n", frame_data.size());

    std::lock_guard<std::mutex> lock(queue_mutex);
    pointcloud_queue.clear();
    pointcloud_queue = std::move(frame_data);
}

void faultMessageCallback(const FaultMessageInfo& fault_message_info) {
    // Use fault message to make some judgments
    // fault_message_info.Print();
    return;
}

// Determines whether the PCAP is finished playing
bool IsPlayEnded(HesaiLidarSdk<LidarPointXYZIRT>& sdk) {
    return sdk.lidar_ptr_->IsPlayEnded();
}

int sdk_main_loop(const std::string &lidar_ip_address, float roll, float pitch, float yaw,float x=0,float y=0,float z=0 ) {
    // #ifndef _MSC_VER
    //   if (system("sudo sh -c \"echo 562144000 > /proc/sys/net/core/rmem_max\"") == -1) {
    //     printf("Command execution failed!\n");
    //   }
    // #endif
    
    HesaiLidarSdk<LidarPointXYZIRT> sample;
    g_sample_ptr = &sample;
    DriverParam param;
    
    param.decoder_param.transform_param.roll = roll;
    param.decoder_param.transform_param.pitch = pitch;
    param.decoder_param.transform_param.yaw = yaw;
    param.decoder_param.transform_param.x = x;
    param.decoder_param.transform_param.y = y;
    param.decoder_param.transform_param.z = z;
    
    // Check if lidar_ip_address ends with .pcap to determine if it's a PCAP file
    bool is_pcap_file = false;
    if (lidar_ip_address.length() >= 5) {
        std::string suffix = lidar_ip_address.substr(lidar_ip_address.length() - 5);
        is_pcap_file = (suffix == ".pcap");
    }
    
    // assign param
    if (is_pcap_file) {
        // Read from PCAP file
        param.input_param.source_type = DATA_FROM_PCAP;
        param.input_param.pcap_path = lidar_ip_address;
        param.input_param.correction_file_path = "/home/zser/HesaiLidar_SDK_2.0/correction/angle_correction/AT34C05D993CCC51.dat";
        param.input_param.firetimes_path = "/home/zser/HesaiLidar_SDK_2.0/correction/firetime_correction/AT128E2X_Firetime_Correction_File.csv";
        param.decoder_param.pcap_play_synchronization = true;
        param.decoder_param.pcap_play_in_loop = false; // pcap playback
        printf("* Reading from PCAP file: %s\n", lidar_ip_address.c_str());
    } else {
        // Read from live lidar
        param.input_param.source_type = DATA_FROM_LIDAR;
        param.input_param.device_ip_address = lidar_ip_address;  // lidar ip
        param.input_param.ptc_port = 9347; // lidar ptc port
        param.input_param.udp_port = 2368; // point cloud destination port
        param.input_param.multicast_ip_address = "";
        param.input_param.ptc_mode = PtcMode::tcp;
        param.input_param.use_ptc_connected = true;  // true: use PTC connected, false: recv correction from local file
        //param.input_param.correction_file_path = "/home/zser/HesaiLidar_SDK_2.0/correction/angle_correction/AT34C05D993CCC51.dat";
        //param.input_param.firetimes_path = "/home/zser/HesaiLidar_SDK_2.0/correction/firetime_correction/AT128E2X_Firetime_Correction_File.csv";
        param.input_param.use_someip = false;  // someip subscribe point cloud and fault message
        param.input_param.host_ip_address = ""; // point cloud destination ip, local ip
        param.input_param.fault_message_port = 0; // fault message destination port, 0: not use
        // PtcMode::tcp_ssl use
        param.input_param.certFile = "";
        param.input_param.privateKeyFile = "";
        param.input_param.caFile = "";
    }

    param.decoder_param.enable_packet_loss_tool = false;
    param.decoder_param.socket_buffer_size = 262144000;
    param.decoder_param.use_timestamp_type = UseTimestampType::sdk_recv_timestamp;
    param.decoder_param.distance_correction_flag = true;
    
    // init lidar with param
    sample.Init(param);

    // assign callback function
    sample.RegRecvCallback(lidarCallback);    
    sample.RegRecvCallback(faultMessageCallback);
    sample.Start();
    sdk_running = true;
    
    if (sample.lidar_ptr_->GetInitFinish(FailInit)) {
        sdk_running = false;
        sample.Stop();
        g_sample_ptr = nullptr;
        return -1;
    }

    while (!IsPlayEnded(sample) || GetMicroTickCount() - last_frame_time < 1000000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if(sdk_running == false)
            break;
    }

    sample.Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("sdk end-------------\n");
    g_sample_ptr = nullptr;
    return 0;
}

int InitHesaiSDK(const std::string& lidar_ip, float roll, float pitch, float yaw,float x,float y,float z) {
    // Start SDK thread, separate from caller
    printf("* lidar_ip:%s\n", lidar_ip.c_str());
    printf("* roll:%.4f\n", roll);
    printf("* pitch:%.4f\n", pitch);
    printf("* yaw:%.4f\n", yaw);
    printf("* x:%.4f\n", x);
    printf("* y:%.4f\n", y);
    printf("* z:%.4f\n", z);
    
    hesai_thread = std::thread(sdk_main_loop, lidar_ip, roll, pitch, yaw, x, y, z);
    return 0;
}

void UninitHesaiSDK() {
    sdk_running = false;
    if (hesai_thread.joinable()) {
        hesai_thread.join();
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

} // namespace HesaiSDK

