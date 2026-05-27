#include "hesai_lidar_sdk.hpp"
 #define LIDAR_PARSER_TEST
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

uint32_t last_frame_time = 0;
uint32_t cur_frame_time = 0;
struct PointData {
    float x, y, z;
    uint8_t reflectivity;
    uint8_t tag;
    uint64_t timestamp;
};
std::mutex queue_mutex;
std::vector<PointData> pointcloud_queue;
bool sdk_running;
std::thread hsai_thread;

//log info, display frame message    
void lidarCallback(const LidarDecodedFrame<LidarPointXYZIRT>  &frame) { 
    cur_frame_time = GetMicroTickCount();
    last_frame_time = cur_frame_time;
    std::vector<PointData> frame_data;
    uint64_t timestamp_ = GetMicroTickCountU64();
    // printf("frame:%d points:%u ",frame.frame_index, frame.points_num); 
    // std::cout <<" timestamp_: "<< timestamp_<<"\n";
    for (uint32_t i = 0; i < frame.points_num; ++i) {
      if(frame.points[i].x==0.0 && frame.points[i].y==0.0 && frame.points[i].z == 0.0)
        continue;
      if( fabs(frame.points[i].x) > 80 )
        continue;
      frame_data.push_back(PointData{
          frame.points[i].x, frame.points[i].y, frame.points[i].z ,
          frame.points[i].intensity,
          (uint8_t)frame.packet_num,
          timestamp_
      });
    }
    //printf("======  frame_data.size = %llu\n" , frame_data.size() );


    std::lock_guard<std::mutex> lock(queue_mutex);
    pointcloud_queue.clear();
    pointcloud_queue=std::move(frame_data);
}

void faultMessageCallback(const FaultMessageInfo& fault_message_info) {
  // Use fault message messages to make some judgments
  // fault_message_info.Print();
  return;
}

// Determines whether the PCAP is finished playing
bool IsPlayEnded(HesaiLidarSdk<LidarPointXYZIRT>& sdk)
{
  return sdk.lidar_ptr_->IsPlayEnded();
}

int sdk_main_loop(const std::string &lidar_ip_addrwss ,const float roll ,const float pitch,const float yaw)
{
  
// #ifndef _MSC_VER
//   if (system("sudo sh -c \"echo 562144000 > /proc/sys/net/core/rmem_max\"") == -1) {
//     printf("Command execution failed!\n");
//   }
// #endif
  HesaiLidarSdk<LidarPointXYZIRT> sample;
  DriverParam param;
  
  param.decoder_param.transform_param.roll =roll ;   // 0;
  param.decoder_param.transform_param.pitch = pitch; // 24 * 3.141592653589793 / 180;  //上下
  param.decoder_param.transform_param.yaw = yaw ;    // 10.5 * 3.141592653589793 / 180;  //转动

  // assign param
  param.input_param.source_type = DATA_FROM_LIDAR;
  param.input_param.device_ip_address = lidar_ip_addrwss;  // lidar ip
  param.input_param.ptc_port = 9347; // lidar ptc port
  param.input_param.udp_port = 2368; // point cloud destination port
  param.input_param.multicast_ip_address = "";

  param.input_param.ptc_mode = PtcMode::tcp;
  param.input_param.use_ptc_connected = false;  // true: use PTC connected, false: recv correction from local file
  param.input_param.correction_file_path = "/home/zser/HesaiLidar_SDK_2.0/correction/angle_correction/AT34C05D993CCC51.dat";
  param.input_param.firetimes_path = "/home/zser/HesaiLidar_SDK_2.0/correction/firetime_correction/AT128E2X_Firetime_Correction_File.csv";

  param.input_param.use_someip = false;  // someip subscribe point cloud and fault message
  param.input_param.host_ip_address = ""; // point cloud destination ip, local ip
  param.input_param.fault_message_port = 0; // fault message destination port, 0: not use

  // PtcMode::tcp_ssl use
  param.input_param.certFile = "";
  param.input_param.privateKeyFile = "";
  param.input_param.caFile = "";

  // param.input_param.source_type = DATA_FROM_PCAP;
  // param.input_param.pcap_path = "Your pcap file path";
  // param.input_param.correction_file_path = "Your correction file path";
  // param.input_param.firetimes_path = "Your firetime file path";


  param.decoder_param.pcap_play_synchronization = true;
  param.decoder_param.pcap_play_in_loop = false; // pcap palyback


  param.decoder_param.enable_packet_loss_tool = false;
  param.decoder_param.socket_buffer_size = 262144000;
  param.decoder_param.use_timestamp_type = UseTimestampType::sdk_recv_timestamp;
  param.decoder_param.distance_correction_flag = true;
   //init lidar with param
   sample.Init(param);

  //assign callback fuction
  sample.RegRecvCallback(lidarCallback);    
  sample.RegRecvCallback(faultMessageCallback);
  sample.Start();
  sdk_running = true;
  if (sample.lidar_ptr_->GetInitFinish(FailInit)) {
    sdk_running = false;
    sample.Stop();
    return -1;
  }

  while (!IsPlayEnded(sample) || GetMicroTickCount() - last_frame_time < 1000000)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(sdk_running == false)
       break ;
  }

  sample.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  printf("sdk end-------------\n" );
  return 0;
}


void init_hsai_lidar_sdk_cplusplus_interface(const std::string &lidar_ip_addrwss ,const float roll ,const float pitch,const float yaw) { //const std::string &config_path
  // 启动 SDK 线程，不阻塞 Python
  printf(" * lidar_ip_addrwss:%s\n",lidar_ip_addrwss.c_str() );
  printf(" * roll:%.4f\n",roll );
  printf(" * pitch:%.4f\n",pitch );
  printf(" * yaw:%.4f\n",yaw );
  hsai_thread = std::thread(sdk_main_loop, lidar_ip_addrwss,roll,pitch,yaw); //config_path
}

void uninit_hsai_lidar_sdk_cplusplus_interface() {
  sdk_running = false;
  if (hsai_thread.joinable())
    hsai_thread.join();
}

py::dict get_hsai_lidar_pointcloud_data_interface(int64_t time_interval_ms) { 
    std::vector<PointData> frame;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        std::vector<PointData> frame_temp = pointcloud_queue;
        pointcloud_queue.clear();
        int64_t begin_timestamp  = frame_temp.size()>0?frame_temp[0].timestamp:GetMicroTickCountU64();
        frame.insert(frame.end(), 
           std::make_move_iterator(frame_temp.begin()),
           std::make_move_iterator(frame_temp.end()));
    }

    size_t n = frame.size();
    //printf("n= %d\n", n );
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n), 3};
    py::array_t<float> points(shape);
    py::array_t<uint8_t> reflectivity(n);
    //py::array_t<uint8_t> tag(n);
    //uint64_t timestamp = frame.empty() ? 0 : frame[0].timestamp;

    auto buf_p = points.mutable_unchecked<2>();
    auto buf_r = reflectivity.mutable_unchecked<1>();
    //auto buf_t = tag.mutable_unchecked<1>();

    for (size_t i = 0; i < n; ++i) {
        buf_p(i, 0) = frame[i].x;
        buf_p(i, 1) = frame[i].y;
        buf_p(i, 2) = frame[i].z;
        buf_r(i) = frame[i].reflectivity;
        //buf_t(i) = frame[i].tag;
    }

    py::dict result;
    result["points"] = points;
    result["reflectivity"] = reflectivity;
    // result["tag"] = tag;
    // result["timestamp"] = timestamp;
    return result;
}



PYBIND11_MODULE(hsai_sdk_wrapper_python, m) {
    m.def("init_hsai_lidar_sdk_cplusplus_interface", &init_hsai_lidar_sdk_cplusplus_interface, "Start the hsai SDK in a background thread");
    m.def("uninit_hsai_lidar_sdk_cplusplus_interface", &uninit_hsai_lidar_sdk_cplusplus_interface, "Stop the hsai SDK cleanly");
    m.def("get_hsai_lidar_pointcloud_data_interface", &get_hsai_lidar_pointcloud_data_interface, "Poll get_hsai_lidar_pointcloud_data_interface");
}
