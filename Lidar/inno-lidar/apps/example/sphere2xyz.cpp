/**
 * Copyright (C) 2024 - Innovusion Inc.
 * @file sphere2xyz.cpp
 * @brief This file contains the implementation of a demo that converts a sphere point cloud to an xyz point cloud.
 *
 * The demo demonstrates how to convert a sphere point cloud to an xyz point cloud. It includes functions for processing
 * the received package information, handling status packets, and converting block data to xyz points.
 *
 * The main class in this file is `CallbackProcessor`, which is responsible for processing the callback functions and
 * converting the point cloud data. It includes functions for converting block data to xyz points, handling co-blocks,
 * and delivering frame data.
 *
 * The code also includes various constants and data structures used in the conversion process.
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

  ~CallbackProcessor();

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

 private:
  CallbackProcessor(const CallbackProcessor &) = delete;
  CallbackProcessor &operator=(const CallbackProcessor &) = delete;
  // convert block to xyz and store to frame_data
  // Falcon point cloud data process
  void block_to_xyz_point(const InnoDataPacket &pkt, std::vector<PcdPoint> &frame_data, int unit_size,
                          int return_number);
  // convert coblock to xyz and store to frame_data
  // RWG compact point cloud data process
  void co_block_to_xyz_point(const InnoDataPacket &pkt, std::vector<PcdPoint> &frame_data, int unit_size,
                             int return_number, const InnoDataPacket *anglehv_table);

 private:
  // Current frame idx
  int64_t current_frame_ = -1;
  // Current frame sub idx
  int64_t current_frame_sub_ = -1;
  // The number of frames received so far
  int64_t frame_so_far_ = -1;
  // Application stop
  volatile bool done_ = false;
  // The number of packets in one frame
  uint64_t sum_packet_ = 0;
  // The number of points in one frame
  uint64_t sum_point_ = 0;
  // Store one frame of point cloud data
  std::vector<PcdPoint> frame_data_;
  InnoDataPacket *anglehv_table_ = nullptr;
};

CallbackProcessor::CallbackProcessor() { frame_data_.reserve(kmaxPointNumberOneFrame); }

CallbackProcessor::~CallbackProcessor() {
  if (anglehv_table_ != nullptr) {
    delete[] anglehv_table_;
    anglehv_table_ = nullptr;
  }
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

// convert innocoblock to xyz and store to frame_data
void CallbackProcessor::co_block_to_xyz_point(const InnoDataPacket &pkt, std::vector<PcdPoint> &frame_data,
                                              int unit_size, int return_number, const InnoDataPacket *anglehv_table) {
  double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
  const InnoCoBlock *block = reinterpret_cast<const InnoCoBlock *>(pkt.payload);

  // the new frame is coming, deliver last frame and store the current frame to new frame_data_
  if (pkt.idx != current_frame_) {
    // deliver frame data
    // ..........

    inno_log_info("frame idx: %" PRId64 ", total point: %" PRI_SIZEU, current_frame_, frame_data_.size());
    frame_data_.clear();
    current_frame_ = pkt.idx;
  }
  for (uint32_t i = 0; i < pkt.item_number;
       i++, block = reinterpret_cast<const InnoCoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) {
    // calculate (x,y,z) cartesian coordinate from spherical coordinate
    // for each point in the block
    // 1. use get_full_angles() to restore angle for each channel
    // 2. use get_xyzr_meter() to calculate (x,y,z)
    InnoCoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(
      &full_angles, block->header, static_cast<InnoItemType>(pkt.type),
      reinterpret_cast<const char *>(reinterpret_cast<const InnoAngleHVTable *>(anglehv_table->payload)->table));

    for (uint32_t channel = 0; channel < kInnoCompactChannelNumber; channel++) {
      for (uint32_t m = 0; m < return_number; m++) {
        const InnoCoChannelPoint &pt = block->points[innocoblock_get_idx(channel, m)];
        InnoXyzrD xyzr;
        uint32_t scan_id = 0;
        if (pt.radius > 0 && InnoDataPacketUtils::is_robinw_inside_fov_point(full_angles.angles[channel])) {
          DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(pkt.type);
          int index = block->header.scan_id * kMaxReceiverInSet + channel;
          scan_id = channel_mapping[index] + block->header.facet * tdc_channel_number;
          InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, scan_id, &xyzr,
                                              static_cast<InnoItemType>(pkt.type), pt.firing);

          PcdPoint pcd_point;
          pcd_point.x = xyzr.x;
          pcd_point.y = xyzr.y;
          pcd_point.z = xyzr.z;
          pcd_point.reflectivity = pt.refl;
          pcd_point.facet = block->header.facet;
          pcd_point.confid_level = pkt.confidence_level;
          pcd_point.timestamp = frame_timestamp_sec + block->header.ts_10us / k10UsInSecond;
          pcd_point.scanline = scan_id;
          pcd_point.scan_idx = block->header.scan_idx;

          frame_data.emplace_back(pcd_point);
        }
      }
    }
  }
}

// convert innoblock to xyz and store to frame_data
void CallbackProcessor::block_to_xyz_point(const InnoDataPacket &pkt, std::vector<PcdPoint> &frame_data, int unit_size,
                                           int return_number) {
  const InnoBlock *blocks = reinterpret_cast<const InnoBlock *>(pkt.payload);
  double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
  const InnoBlock *block = reinterpret_cast<const InnoBlock *>(&blocks[0]);

  // the new frame is coming, deliver last frame and store the current frame to frame_data_
  if (pkt.idx != current_frame_) {
    // deliver frame data
    // ..........

    inno_log_info("frame idx: %" PRId64 ", total point: %" PRI_SIZEU, current_frame_, frame_data_.size());
    frame_data_.clear();
    current_frame_ = pkt.idx;
  }
  for (uint32_t i = 0; i < pkt.item_number;
       i++, block = reinterpret_cast<const InnoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) {
    // calculate (x,y,z) cartesian coordinate from spherical coordinate
    // for each point in the block
    // 1. use get_full_angles() to restore angle for each channel
    // 2. use get_xyzr_meter() to calculate (x,y,z)
    InnoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header, static_cast<InnoItemType>(pkt.type));

    for (uint32_t channel = 0; channel < kInnoChannelNumber; channel++) {
      for (uint32_t m = 0; m < return_number; m++) {
        const InnoChannelPoint &pt = block->points[innoblock_get_idx(channel, m)];
        InnoXyzrD xyzr;

        if (pt.radius > 0) {
          InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, channel, &xyzr,
                                              static_cast<InnoItemType>(pkt.type), 1, pkt.long_distance_mode);
          PcdPoint pcd_point;
          pcd_point.x = xyzr.x;
          pcd_point.y = xyzr.y;
          pcd_point.z = xyzr.z;
          pcd_point.reflectivity = pt.refl;
          pcd_point.facet = block->header.facet;
          pcd_point.is_2nd_return = pt.is_2nd_return;
          pcd_point.multi_return = m;
          pcd_point.confid_level = pkt.confidence_level;
          pcd_point.timestamp = frame_timestamp_sec + block->header.ts_10us / k10UsInSecond;
          pcd_point.scanline = block->header.scan_id;
          pcd_point.scan_idx = block->header.scan_idx;

          frame_data.emplace_back(pcd_point);
        }
      }
    }
  }
}

/*
  this function shows how to enumerate blocks and points in the packet
  directly, and use InnoDataPacketUtils::get_xyzr_meter to get each
  point's x,y,z coodindate, then add it to frame_data
*/
int CallbackProcessor::process_data(int handler, const InnoDataPacket &pkt) {
  uint32_t return_number;
  uint32_t unit_size;
  // get block size and number of returns
  InnoDataPacketUtils::get_block_size_and_number_return(pkt, &unit_size, &return_number);

  if (pkt.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    block_to_xyz_point(pkt, frame_data_, unit_size, return_number);
  } else if (pkt.type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
             pkt.type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
    if (anglehv_table_ == nullptr) {
      // get anglehv table, which is used to calculate the angle of each point
      // different lidar have different anglehv table
      anglehv_table_ = reinterpret_cast<InnoDataPacket *>(new char[kInnoAngleHVTableMaxSize]);
      inno_lidar_get_anglehv_table(handler, anglehv_table_);
    }
    co_block_to_xyz_point(pkt, frame_data_, unit_size, return_number, anglehv_table_);
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

  handle = inno_lidar_open_live("live",  // name of lidar instance
                                lidar_option.lidar_ip.c_str(), lidar_option.lidar_port, INNO_LIDAR_PROTOCOL_PCS_UDP,
                                lidar_option.lidar_udp_port);

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
