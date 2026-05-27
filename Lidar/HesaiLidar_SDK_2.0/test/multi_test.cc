#include "hesai_lidar_sdk.hpp"
#include <thread>
#include <cstdio>
#include <functional>

#define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST

// Configuration struct to hold all parameters
struct LidarConfig {
    DriverParam driver_param;
    std::function<void(const LidarDecodedFrame<LidarPointXYZICRT>&)> point_cloud_callback;
    std::function<void(const FaultMessageInfo&)> fault_callback;
    uint32_t* last_frame_time;
};

uint32_t last_frame_time_1 = 0;
uint32_t cur_frame_time_1 = 0;
uint32_t last_frame_time_2 = 0;
uint32_t cur_frame_time_2 = 0;

//log info, display frame message for lidar 1
void lidarCallback1(const LidarDecodedFrame<LidarPointXYZICRT> &frame) {  
    cur_frame_time_1 = GetMicroTickCount();
    if (last_frame_time_1 == 0) last_frame_time_1 = GetMicroTickCount();
    if (cur_frame_time_1 - last_frame_time_1 > kMaxTimeInterval) {
        printf("Lidar1 - Time between last frame and cur frame is: %u us\n", 
               (cur_frame_time_1 - last_frame_time_1));
    }
    last_frame_time_1 = cur_frame_time_1;
    printf("Lidar1 - frame:%d points:%u packet:%u start time:%lf end time:%lf\n",
           frame.frame_index, frame.points_num, frame.packet_num, 
           frame.frame_start_timestamp, frame.frame_end_timestamp);
}

//log info, display frame message for lidar 2
void lidarCallback2(const LidarDecodedFrame<LidarPointXYZICRT> &frame) {  
    cur_frame_time_2 = GetMicroTickCount();
    if (last_frame_time_2 == 0) last_frame_time_2 = GetMicroTickCount();
    if (cur_frame_time_2 - last_frame_time_2 > kMaxTimeInterval) {
        printf("Lidar2 - Time between last frame and cur frame is: %u us\n", 
               (cur_frame_time_2 - last_frame_time_2));
    }
    last_frame_time_2 = cur_frame_time_2;
    printf("Lidar2 - frame:%d points:%u packet:%u start time:%lf end time:%lf\n",
           frame.frame_index, frame.points_num, frame.packet_num, 
           frame.frame_start_timestamp, frame.frame_end_timestamp);
}

void faultMessageCallback1(const FaultMessageInfo& fault_message_info) {
    // fault_message_info.Print();
}

void faultMessageCallback2(const FaultMessageInfo& fault_message_info) {
    // fault_message_info.Print();
}

// Determines whether the PCAP is finished playing
bool IsPlayEnded(HesaiLidarSdk<LidarPointXYZICRT>& sdk)
{
    return sdk.lidar_ptr_->IsPlayEnded();
}

void lidarThread(const LidarConfig& config) {
#ifndef _MSC_VER
    if (system("sudo sh -c \"echo 562144000 > /proc/sys/net/core/rmem_max\"") == -1) {
        printf("Command execution failed!\n");
    }
#endif
    HesaiLidarSdk<LidarPointXYZICRT> sample;

    //init lidar with param
    sample.Init(config.driver_param);

    //assign callback functions
    sample.RegRecvCallback(config.point_cloud_callback);
    sample.RegRecvCallback(config.fault_callback);

    sample.Start();
    if (sample.lidar_ptr_->GetInitFinish(FailInit)) {
        sample.Stop();
        return;
    }

    while (!IsPlayEnded(sample) || GetMicroTickCount() - *config.last_frame_time < 1000000)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("The PCAP file has been parsed and we will exit the program.\n");
}

int main(int argc, char *argv[]) {
    // Create and configure parameters for first lidar
    LidarConfig config1;
    config1.driver_param.use_gpu = (argc > 1);
    // Configure driver parameters for lidar 1
#ifdef LIDAR_PARSER_TEST
    config1.driver_param.input_param.source_type = DATA_FROM_LIDAR;
    config1.driver_param.input_param.device_ip_address = "192.168.1.201";  // lidar ip
    config1.driver_param.input_param.ptc_port = 9347; // lidar ptc port
    config1.driver_param.input_param.udp_port = 2368; // point cloud destination port
    config1.driver_param.input_param.multicast_ip_address = "";

    config1.driver_param.input_param.use_ptc_connected = true;  // true: use PTC connected, false: recv correction from local file
    config1.driver_param.input_param.correction_file_path = "Your correction file path";
    config1.driver_param.input_param.firetimes_path = "Your firetime file path";

    config1.driver_param.input_param.use_someip = false;  // someip subscribe point cloud and fault message
    config1.driver_param.input_param.host_ip_address = ""; // point cloud destination ip, local ip
    config1.driver_param.input_param.fault_message_port = 9348; // fault message destination port
#endif

#ifdef SERIAL_PARSER_TEST
    config1.driver_param.input_param.source_type = DATA_FROM_SERIAL;
    config1.driver_param.input_param.rs485_com = "Your serial port name for receiving point cloud";
    config1.driver_param.input_param.rs232_com = "Your serial port name for sending cmd";
    config1.driver_param.input_param.point_cloud_baudrate = 3125000;
    config1.driver_param.input_param.correction_file_path = "Your correction file path";
#endif
    config1.driver_param.decoder_param.enable_packet_loss_tool = false;
    config1.driver_param.decoder_param.socket_buffer_size = 262144000;
    // Set the last frame time for lidar 1
    config1.last_frame_time = &last_frame_time_1;
    // Set callback functions for lidar 1
    config1.point_cloud_callback = lidarCallback1;
    config1.fault_callback = faultMessageCallback1;

    // Create and configure parameters for second lidar
    LidarConfig config2;
    config2.driver_param.use_gpu = (argc > 1);
    // Configure driver parameters for lidar 2
#ifdef LIDAR_PARSER_TEST
    config2.driver_param.input_param.source_type = DATA_FROM_LIDAR;
    config2.driver_param.input_param.device_ip_address = "192.168.1.202";  // lidar ip
    config2.driver_param.input_param.ptc_port = 9347; // lidar ptc port
    config2.driver_param.input_param.udp_port = 2369; // point cloud destination port
    config2.driver_param.input_param.multicast_ip_address = "";

    config2.driver_param.input_param.use_ptc_connected = true;  // true: use PTC connected, false: recv correction from local file
    config2.driver_param.input_param.correction_file_path = "Your correction file path";
    config2.driver_param.input_param.firetimes_path = "Your firetime file path";

    config2.driver_param.input_param.use_someip = false;  // someip subscribe point cloud and fault message
    config2.driver_param.input_param.host_ip_address = ""; // point cloud destination ip, local ip
    config2.driver_param.input_param.fault_message_port = 9348; // fault message destination port
#endif

#ifdef SERIAL_PARSER_TEST
    config2.driver_param.input_param.source_type = DATA_FROM_SERIAL;
    config2.driver_param.input_param.rs485_com = "Your serial port name for receiving point cloud";
    config2.driver_param.input_param.rs232_com = "Your serial port name for sending cmd";
    config2.driver_param.input_param.point_cloud_baudrate = 3125000;
    config2.driver_param.input_param.correction_file_path = "Your correction file path";
#endif
    config2.driver_param.decoder_param.enable_packet_loss_tool = false;
    config2.driver_param.decoder_param.socket_buffer_size = 262144000;
    // Set the last frame time for lidar 2
    config2.last_frame_time = &last_frame_time_2;
    // Set callback functions for lidar 2
    config2.point_cloud_callback = lidarCallback2;
    config2.fault_callback = faultMessageCallback2;

    // Create and start both lidar threads
    std::thread lidar_thread1(lidarThread, std::ref(config1));
    std::thread lidar_thread2(lidarThread, std::ref(config2));
    // Wait for both threads to complete
    lidar_thread1.join();
    lidar_thread2.join();
    
    return 0;
}