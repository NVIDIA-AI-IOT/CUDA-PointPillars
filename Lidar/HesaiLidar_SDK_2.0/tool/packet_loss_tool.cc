#include "hesai_lidar_sdk.hpp"

uint32_t last_frame_time;
uint32_t cur_frame_time;

#define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST
// #define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST

//log info, display frame message
void lidarCallback(const LidarDecodedFrame<LidarPointXYZIRT>  &frame) {  
  cur_frame_time = GetMicroTickCount();
  if (cur_frame_time - last_frame_time > kMaxTimeInterval) {
    printf("Time between last frame and cur frame is: %u us\n", (cur_frame_time - last_frame_time));
  }
  last_frame_time = cur_frame_time;
  printf("frame:%d points:%u packet:%u start time:%lf end time:%lf\n",frame.frame_index, frame.points_num, frame.packet_num, frame.frame_start_timestamp, frame.frame_end_timestamp);
}

void faultMessageCallback(const FaultMessageInfo& fault_message_info) {
  // Use fault message messages to make some judgments
  // fault_message_info.Print();
  return;
}

int main(int argc, char *argv[])
{
#ifndef _MSC_VER
  if (system("sudo sh -c \"echo 562144000 > /proc/sys/net/core/rmem_max\"") == -1) {
     printf("Command execution failed!\n");
  }
#endif
  HesaiLidarSdk<LidarPointXYZIRT> sample;
  DriverParam param;

  // assign param
  param.use_gpu = (argc > 1);
  // assign param
#ifdef LIDAR_PARSER_TEST
  param.input_param.source_type = DATA_FROM_LIDAR;
  param.input_param.device_ip_address = "192.168.1.201";  // lidar ip
  param.input_param.ptc_port = 9347; // lidar ptc port
  param.input_param.udp_port = 2368; // point cloud destination port
  param.input_param.multicast_ip_address = "";

  param.input_param.use_ptc_connected = true;  // true: use PTC connected, false: recv correction from local file
  param.input_param.correction_file_path = "Your correction file path";
  param.input_param.firetimes_path = "Your firetime file path";

  param.input_param.use_someip = false;  // someip subscribe point cloud and fault message
  param.input_param.host_ip_address = ""; // point cloud destination ip, local ip
  param.input_param.fault_message_port = 9348; // fault message destination port
#endif

#ifdef SERIAL_PARSER_TEST
  param.input_param.source_type = DATA_FROM_SERIAL;
  param.input_param.rs485_com = "Your serial port name for receiving point cloud";
  param.input_param.rs232_com = "Your serial port name for sending cmd";
  param.input_param.point_cloud_baudrate = 3125000;
  param.input_param.correction_file_path = "Your correction file path";
#endif

#ifdef PCAP_PARSER_TEST
  param.input_param.source_type = DATA_FROM_PCAP;
  param.input_param.pcap_path = "Your pcap file path";
  param.input_param.correction_file_path = "Your correction file path";
  param.input_param.firetimes_path = "Your firetime file path";


  param.decoder_param.pcap_play_synchronization = true;
  param.decoder_param.pcap_play_in_loop = false; // pcap palyback
#endif

#ifdef EXTERNAL_INPUT_PARSER_TEST
  param.input_param.source_type = DATA_FROM_EXTERNAL_INPUT;
  param.input_param.correction_file_path = "Your correction file path";
  param.input_param.firetimes_path = "Your firetime file path";
#endif

  param.decoder_param.socket_buffer_size = 262144000;

  param.decoder_param.enable_packet_loss_tool = true;
  param.decoder_param.enable_packet_timeloss_tool = true;
  param.decoder_param.packet_timeloss_tool_continue = true;

  //init lidar with param
  sample.Init(param);

  //assign callback fuction
  sample.RegRecvCallback(lidarCallback);
  sample.RegRecvCallback(faultMessageCallback);

  //star process thread
  last_frame_time = GetMicroTickCount();
  float run_time = 15;
  if (argc > 1 ) run_time = atof(argv[1]);

  sample.Start();
  if (sample.lidar_ptr_->GetInitFinish(FailInit)) {
    sample.Stop();
    return -1;
  }

  uint64_t start = GetMicroTickCountU64();
  while (1)
  {
    uint64_t now = GetMicroTickCountU64();

    if (now <= start) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    //packet loss tool run time
    if (int(now -start) < int(run_time * kMicrosecondToSecondInt)) std::this_thread::sleep_for(std::chrono::microseconds(1000));
    else {
      break;
    }
  }

  sample.onRelease();
  
  uint64_t end = GetMicroTickCountU64();
  printf("total recevice packet time: %lums\n", (end -start) / 1000);
  uint32_t total_packet_count = sample.lidar_ptr_->GetGeneralParser()->seqnum_loss_message_.total_packet_count;
  uint32_t total_packet_loss_count = sample.lidar_ptr_->GetGeneralParser()->seqnum_loss_message_.total_loss_count;
  printf("total receviced packet count: %u\n", total_packet_count);
  printf("package loss: \n");
  printf("total loss packet count: %u\n", total_packet_loss_count);
  if (float(total_packet_loss_count) / float(total_packet_count) > 0.01) {
    printf("Error: Packet seqnum loss rate exceeds 1%%\n");
  }
  uint32_t total_packet_timeloss_count = sample.lidar_ptr_->GetGeneralParser()->time_loss_message_.total_timeloss_count;
  printf("timestamp loss: \n");
  printf("total loss packet count: %u\n", total_packet_timeloss_count);
  if (float(total_packet_timeloss_count) / float(total_packet_count) > 0.01) {
    printf("Error: Packet timestamp loss rate exceeds 1%%\n");
  }

  sample.Stop();
}