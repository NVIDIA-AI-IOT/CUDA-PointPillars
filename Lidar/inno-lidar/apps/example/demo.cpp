/**
 * Copyright (C) 2024 - Innovusion Inc.
 * @file demo.cpp
 * @brief Example code for using the Inno LIDAR SDK.
 *
 * This code demonstrates how to use the Inno LIDAR SDK to open a LIDAR device, set the SDK callback to receive XYZ
 * point cloud data, and start and stop receiving data from the LIDAR device.
 *
 * Usage:
 *   - Compile and run the code with the appropriate command line options.
 *   - The code opens a live LIDAR device via UDP and receives XYZ point cloud data.
 *   - The received data is processed by the CallbackProcessor class.
 *   - The CallbackProcessor class prints the received package information and stores the XYZ point cloud data in a
 * vector.
 *   - The code also provides command line options to specify the LIDAR IP address and UDP port.
 *
 * Dependencies:
 *   - The code depends on the Inno LIDAR SDK and related header files.
 *
 * @note This code is provided as an example and may need to be modified to work with your specific LIDAR device and
 * application.
 * @note Make sure to include the necessary header files and link against the Inno LIDAR SDK library when compiling this
 * code.
 */

#if defined(__MINGW64__) || !defined(_WIN32)
#include <getopt.h>
#include <unistd.h>
#else
#include "src/utils/getopt_windows.h"
#endif

#include <fcntl.h>

#include <string>
#include <thread>
#include <vector>

#include "src/sdk_common/inno_lidar_api.h"
#include "src/sdk_common/inno_lidar_other_api.h"
#include "src/sdk_common/inno_lidar_packet_utils.h"
#include "src/utils/inno_lidar_log.h"
#if !defined(__MINGW64__) && defined(_WIN32)
#include <io.h>
#include <stdarg.h>
#include <stdio.h>
#endif
#include <iostream>
#include <fstream>
#include <iomanip>  // for std::setprecision
#include <sstream>

static size_t kmaxPointNumberOneFrame = 500000;
static const double kUsInSecond = 1000000.0;
static const double k10UsInSecond = 100000.0;
struct LidarOption {
  std::string lidar_ip = "172.168.1.10";
  uint16_t lidar_port = 8010;
  uint16_t lidar_udp_port = 8010;
};

struct PcdPoint {
  float x;
  float y;
  float z;
  uint16_t reflectivity;
  uint16_t facet;
  uint16_t is_2nd_return;
  uint16_t multi_return;
  uint16_t confid_level;
  double timestamp;
  uint16_t scanline;
  uint16_t scan_idx;
  uint32_t frame_id;
};

class CallbackProcessor {
  ///
  /// Print the received package information
  ///
 public:
  explicit CallbackProcessor();

  void set_done() { done_ = true; }
  bool is_done() { return done_; }

  /*@param context Callback context passed in inno_lidar_set_callbacks().
   * @param from_remote > 0 means the message is from remote source
   * @param level Severity of message.
   * @param code Error code.
   * @param error_message Error message.
   * @return void.
   */
  void process_message(const enum InnoMessageLevel level, const enum InnoMessageCode code, const char *error_message);

  /*
   * @param context Callback context passed in inno_lidar_set_callbacks().
   * @param data Pointer to InnoDataPacket
   * @return 0
   */
  int process_data(int handler, const InnoDataPacket &pkt);

  /*
   * @param context Callback context passed in inno_lidar_set_callbacks().
   * @param status Pointer to InnoDataPacket
   * @return 0
   */
  int process_status(const InnoStatusPacket &pkt);

   /*
    recorder lidar send inno_pc data 
   */
  int recorder_data(int lidar_handle, void *context, enum InnoRecorderCallbackType type,
                                      const char *buffer, int len);

 private:
  CallbackProcessor(const CallbackProcessor &) = delete;
  CallbackProcessor &operator=(const CallbackProcessor &) = delete;

 private:
  // Current frame idx
  int64_t current_frame_ = -1;
  // The number of frames received so far
  int64_t frame_so_far_ = -1;
  // Application stop
  volatile bool done_ = false;
  // Store one frame of pcd point
  std::vector<PcdPoint> frame_data_;
};

CallbackProcessor::CallbackProcessor() {
  // reserve memory for store one frame data
  frame_data_.reserve(kmaxPointNumberOneFrame);
}

void save_frame_to_pcd(const std::vector<PcdPoint> &points, int64_t frame_id) {
  std::cout <<"save_frame_to_pcd \n";
  if (points.empty()) {
    std::cout << "Frame " << frame_id << " is empty, skip saving.\n";
    return;
  }

  // 文件名：frame_00001.pcd
  std::ostringstream oss;
  oss << "frame_" << std::setw(5) << std::setfill('0') << frame_id << ".pcd";
  std::string filename = oss.str();

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  // 写入 PCD 头
  ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z intensity\n";
  ofs << "SIZE 4 4 4 4\n";
  ofs << "TYPE F F F F\n";
  ofs << "COUNT 1 1 1 1\n";
  ofs << "WIDTH " << points.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << points.size() << "\n";
  ofs << "DATA ascii\n";

  // 写入每个点的数据
  ofs << std::fixed << std::setprecision(4);
  for (const auto &p : points) {
    ofs << p.x << " " << p.y << " " << p.z << " " << static_cast<float>(p.reflectivity) << "\n";
  }

  ofs.close();
  std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
}



void CallbackProcessor::process_message(const enum InnoMessageLevel level, const enum InnoMessageCode code,
                                        const char *error_message) {
  inno_log_info("level = %d code = %d\n", level, code);

  // process exception
  if ((level == INNO_MESSAGE_LEVEL_INFO && code == INNO_MESSAGE_CODE_READ_FILE_END) ||
      (level == INNO_MESSAGE_LEVEL_CRITICAL && code == INNO_MESSAGE_CODE_CANNOT_READ)) {
    this->set_done();
  }

  switch (level) {
    case INNO_MESSAGE_LEVEL_INFO:
      inno_log_info("content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_WARNING:
      inno_log_warning("content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_ERROR:
      inno_log_error("content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_FATAL:
      inno_log_fatal("content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_CRITICAL:
      inno_log_fatal("content = %s\n", error_message);
      break;
    default:
      inno_log_info("content = %s\n", error_message);
      break;
  }
}

int CallbackProcessor::process_status(const InnoStatusPacket &pkt) {
  // sanity check
  if (!inno_lidar_check_status_packet(&pkt, 0)) {
    inno_log_error("corrupted pkt->idx = %" PRI_SIZEU, pkt.idx);
    return 0;
  }

  static size_t cnt = 0;
  if (cnt++ % 100 == 1) {
    constexpr size_t buf_size = 2048;
    char buf[buf_size]{0};

    int ret = inno_lidar_printf_status_packet(&pkt, buf, buf_size);
    if (ret > 0) {
      inno_log_info("Received status packet #%" PRI_SIZELU ": %s", cnt, buf);
    } else {
      inno_log_warning("Received status packet #%" PRI_SIZELU ": errorno: %d", cnt, ret);
    }
  }

  return 0;
}

int CallbackProcessor::process_data(int handler, const InnoDataPacket &pkt) {
  // iterate each point in the pkt
  printf("item_number = %d\n" ,pkt.item_number );
  for (int i = 0; i < pkt.item_number; i++) {
    if (pkt.type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
      /*
        If new frame comes, output statistical information.
      */
      std::cout << "111current_frame_ :"<< current_frame_<<" pkt.idx= "<<pkt.idx<<"\n";
      if (current_frame_ != pkt.idx) {
        inno_log_info("frame idx: %" PRId64 ", total point: %" PRI_SIZEU, current_frame_, frame_data_.size());
        // deliver the frame data to user and then clear the buffer
        if (current_frame_ >= 0) {  // 第一帧时跳过
          save_frame_to_pcd(frame_data_, current_frame_);
        }
        frame_data_.clear();
        frame_so_far_++;
        current_frame_ = pkt.idx;
      }
      InnoXyzPoint *point = reinterpret_cast<InnoXyzPoint *>(const_cast<char *>(pkt.payload));
      // if want to get more information, please define your pcd_point struct and copy InnoXyzPoint data to it
      PcdPoint pcd_point;
      pcd_point.x = point[i].x;
      pcd_point.y = point[i].y;
      pcd_point.z = point[i].z;
      pcd_point.reflectivity = point[i].refl;
      double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond + point[i].ts_10us / k10UsInSecond;
      pcd_point.timestamp = frame_timestamp_sec;
      pcd_point.facet = point[i].facet;
      pcd_point.is_2nd_return = point[i].is_2nd_return;
      pcd_point.multi_return = point[i].multi_return;
      frame_data_.emplace_back(pcd_point);

    } 
    else if (CHECK_EN_XYZ_POINTCLOUD_DATA(pkt.type)) {
      /*
        If new frame comes, output statistical information.
     */
      //std::cout << "22current_frame_ :"<< current_frame_<<" pkt.idx= "<<pkt.idx<<"\n";
      if (current_frame_ != pkt.idx) {
        inno_log_info("frame idx: %" PRId64 ", total point: %" PRI_SIZEU, current_frame_, frame_data_.size());
        // deliver the frame data to user and then clear the buffer
        if (current_frame_ >= 0) {  // 第一帧时跳过
          save_frame_to_pcd(frame_data_, current_frame_);
        }
        frame_data_.clear();
        frame_so_far_++;
        current_frame_ = pkt.idx;
      }
      InnoEnXyzPoint *point = reinterpret_cast<InnoEnXyzPoint *>(const_cast<char *>(pkt.payload));
      // if want to get more information, please define your PcdPoint struct and copy InnoEnXyzPoint data to it
      PcdPoint pcd_point;
      pcd_point.x = point[i].x;
      pcd_point.y = point[i].y;
      pcd_point.z = point[i].z;
      pcd_point.reflectivity = point[i].reflectance;
      double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond + point[i].ts_10us / k10UsInSecond;
      pcd_point.timestamp = frame_timestamp_sec;
      pcd_point.facet = point[i].facet;
      frame_data_.emplace_back(pcd_point);
    }
  }

  return 0;
}

int CallbackProcessor::recorder_data(int lidar_handle, void *context, enum InnoRecorderCallbackType type,
                                    const char *buffer, int len) {
  InnoDataPacket *pkt = reinterpret_cast<InnoDataPacket *>(const_cast<char *>(buffer));
  // inno_log_info("recorder_data type %d len %d sub_seq %u", type, len, pkt->sub_seq);
  if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt->type)) {
    // CO_SPHERE_POINTCLOUD_DATA Need write anghv table to the file header
  }
  return 0;
}

void usage(const char *arg0) {
  inno_fprintf(2,
               "\n"
               "Examples:\n"
               " record frames from live LIDAR via UDP.\n"
               "   %s --lidar-ip 172.168.1.10 --lidar-udp-port 8010 \n\n",
               arg0);
  return;
}

//
// parse command
//
void parse_command(int argc, char **argv, LidarOption *lidar_option) {
  // getopt_long stores the option index here.
  int c;
  struct option long_options[] = {// These options set a flag.
                                  {"lidar-ip", required_argument, 0, 'n'},
                                  {"lidar-udp-port", required_argument, 0, 'u'},
                                  {0, 0, 0, 0}};

  const char *optstring = "n:u:i:h";
  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, optstring, long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        // If this option set a flag, do nothing else now.
        if (long_options[option_index].flag != 0) {
          break;
        }
        inno_log_verify(optarg == NULL, "option %s with arg %s", long_options[option_index].name, optarg);
        break;

      // lidar live open
      case 'n':
        lidar_option->lidar_ip = optarg;
        break;

      case 'u':
        lidar_option->lidar_udp_port = strtoul(optarg, NULL, 0);
        break;

      // other
      case 'h':
        usage(argv[0]);
        exit(0);
        break;

      case '?':
        abort();

      default:
        inno_log_error("unknown options %c\n", c);
        usage(argv[0]);
        exit(1);
    }
  }
}

int main(int argc, char **argv) {
  if (argc == 1) {
    usage(argv[0]);
    exit(0);
  }
  inno_log_info("inno_api_version %s", inno_api_version());
  inno_log_info("inno_api_build_tag %s", inno_api_build_tag());
  inno_log_info("inno_api_build_time %s", inno_api_build_time());
  // Set log printing level
  inno_lidar_set_log_level(INNO_LOG_LEVEL_INFO);

  // Parse command parameters
  LidarOption lidar_option;
  parse_command(argc, argv, &lidar_option);

  // Create processing class
  CallbackProcessor processor;

  int handle;
  std::cout << "lidar_option.lidar_ip "<<lidar_option.lidar_ip<<"\n";
  std::cout << "lidar_option.lidar_port "<<lidar_option.lidar_port<<"\n";
  std::cout << "INNO_LIDAR_PROTOCOL_PCS_UDP "<<INNO_LIDAR_PROTOCOL_PCS_UDP<<"\n";
  std::cout << "lidar_option.lidar_udp_port "<<lidar_option.lidar_udp_port<<"\n";
  std::getchar();
  handle = inno_lidar_open_live("live",  // name of lidar instance
                                lidar_option.lidar_ip.c_str(), lidar_option.lidar_port, INNO_LIDAR_PROTOCOL_PCS_UDP,
                                lidar_option.lidar_udp_port);

  // set SDK to callback with XYZ POINTCLOUD and
  // then enumerate each xyz points
  inno_lidar_set_attribute_string(handle, "force_xyz_pointcloud", "1");
  // if need vehicle coordinate, force call the follow function
  // inno_lidar_set_attribute_string(handle, "force_vehicle_coordinate", "1");
  inno_log_verify(handle > 0, "cannot open lidar");

  int ret = inno_lidar_set_callbacks(
    handle,

    // message callback, receive lidar runtime debugging / error messages
    [](const int lidar_handle, void *ctx, const uint32_t from_remote, const enum InnoMessageLevel level,
       const enum InnoMessageCode code, const char *error_message) {
      return reinterpret_cast<CallbackProcessor *>(ctx)->process_message(level, code, error_message);
    },

    // data callback, receiving point cloud data
    [](const int lidar_handle, void *ctx, const InnoDataPacket *pkt) -> int {
      inno_log_verify(pkt, "pkt");
      return reinterpret_cast<CallbackProcessor *>(ctx)->process_data(lidar_handle, *pkt);
    },

    // status callback, receive radar operation status information
    [](const int lidar_handle, void *ctx, const InnoStatusPacket *pkt) -> int {
      inno_log_verify(pkt, "pkt");
      return reinterpret_cast<CallbackProcessor *>(ctx)->process_status(*pkt);
    },

    // use default get_host_time()
    NULL, &processor);

  inno_log_verify(ret == 0, "set_callbacks failed %d", ret);

 // if need  async reocde inno_pc, please reister the follow callback otherwise unnecessary to set recorder callback
  inno_lidar_set_recorder_callback(
      handle, InnoRecorderCallbackType::INNO_RECORDER_CALLBACK_TYPE_INNO_PC,
      [](int lidar_handle, void *context, enum InnoRecorderCallbackType type, const char *buffer, int len) {
        return reinterpret_cast<CallbackProcessor *>(context)->recorder_data(lidar_handle, context, type, buffer, len);
      },
      &processor);

  ret = inno_lidar_start(handle);
  inno_log_verify(ret == 0, "start failed %d", ret);

  // Blocking waiting for operation completion
  while (!processor.is_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ret = inno_lidar_stop(handle);
  inno_log_verify(ret == 0, "stop failed %d", ret);

  ret = inno_lidar_close(handle);
  inno_log_verify(ret == 0, "close failed %d", ret);

  return 0;
}
