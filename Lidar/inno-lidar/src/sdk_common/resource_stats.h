/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_COMMON_RESOURCE_STATS_H_
#define SDK_COMMON_RESOURCE_STATS_H_

#include <mutex>

namespace innovusion {
class InnoLidarBase;

/**
 * @brief ResourceStats
 */
class ResourceStats {
 public:
  enum PacketType {
    PACKET_TYPE_SRC = 0,
    PACKET_TYPE_DATA = 1,
    PACKET_TYPE_POINT = 2,
    PACKET_TYPE_MESSAGE = 3,
    PACKET_TYPE_STATUS = 4,
    PACKET_TYPE_MAX = 5,
  };

 public:
  /**
   * @brief ResourceStats constructor
   * @param l Address of InnoLidarBase
   */
  explicit ResourceStats(InnoLidarBase *l);
  virtual ~ResourceStats() {
  }

  /**
   * @brief Summary packets and bytes count for each packet type
   * @param type    PacketType
   * @param packet  Packet count
   * @param byte    Bytes count
   * @param ref_count_sum Refference count sum
   * @param intensity_sum Intensity sum
   * @return void
   */
  void update_packet_bytes(enum PacketType type,
                           size_t packet, size_t byte,
                           size_t ref_count_sum = 0,
                           uint64_t intensity_sum = 0);
  /**
   * @brief Show resources statistics periodically
   */
  void periodically_show();

  /**
   * @brief Show resource statistics
   */
  void show();

  /**
   * @brief Get last resource statistics info buffer
   * @param buf       Buffer to store info
   * @param buf_size  Buffer size
   */
  void get_last_output_info_buffer(char *buf, size_t buf_size);

 protected:
  /**
   * @brief Get extra statistics info from lidar
   * @param buffer    Buffer to store extra info
   * @param size      Buffer size
   * @param time_diff Time diff from current to last
   */
  virtual void get_extra_info_(char *buffer, size_t size, double time_diff) = 0;

 public:
  static const char *packet_type_names[PACKET_TYPE_MAX];
  constexpr static const double kStageReadBandWidthThresholdMBps = 3.0;

 protected:
  InnoLidarBase *lidar_;

 private:
  uint64_t show_cpu_count_;
  uint64_t print_stats_count_;
  uint64_t start_time_ms_;
  uint64_t lastp_time_ms_;
  uint64_t last_update_time_ms_;
  uint64_t last_print_stats_time_ms_;
  uint64_t total_ref_count_;
  uint64_t total_ref_intensity_;

  uint64_t lastp_byte_[PACKET_TYPE_MAX] = {0};
  uint64_t total_byte_[PACKET_TYPE_MAX] = {0};
  uint64_t lastp_packet_[PACKET_TYPE_MAX] = {0};
  uint64_t total_packet_[PACKET_TYPE_MAX] = {0};

  char last_output_stats_buffer_[512] = {0};

  std::mutex mutex_;
};

}  // namespace innovusion

#endif  // SDK_COMMON_RESOURCE_STATS_H_
