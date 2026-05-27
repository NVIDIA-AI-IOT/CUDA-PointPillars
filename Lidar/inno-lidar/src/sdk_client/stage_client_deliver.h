/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_STAGE_CLIENT_DELIVER_H_
#define SDK_CLIENT_STAGE_CLIENT_DELIVER_H_

#include <mutex>
#include <string>

#include "sdk_client/lidar_fault_check.h"
#include "sdk_common/inno_lidar_packet.h"
#include "utils/config.h"
#include "utils/utils.h"

struct InnoCommonHeader;

namespace innovusion {
class InnoLidarClient;

/**
 * @brief StageClientDeliverConfig
 */
class StageClientDeliverConfig : public Config {
 public:
  /**
   * @brief Initialize configuration
   */
  StageClientDeliverConfig() : Config() {
    enable_stage_client_deliver2 = true;
  }

  /**
   * @brief   Get config name
   * @return  Return config name
   */
  const char *get_type() const override {
    return "LidarClient_StageClientDeliver";
  }

  /**
   * @brief Update configurations
   * @param key   Config key
   * @param value Config value
   * @return Ignored
   */
  int set_key_value_(const std::string &key, double value) override {
    SET_CFG(enable_stage_client_deliver2);
    return -1;
  }

  /**
   * @brief Update configurations
   * @param key   Config key
   * @param value Config value
   * @return Ignored
   */
  int set_key_value_(const std::string &key, const std::string value) override {
    // no string attribute
    return -1;
  }

  BEGIN_CFG_MEMBER()
  bool enable_stage_client_deliver2;
  END_CFG_MEMBER()
};

/**
 * @brief StageClientDeliver
 */
class StageClientDeliver {
  friend InnoLidarClient;

 public:
  /**
   * @brief StageClientDeliver constructor
   * @param l InnoLidarClient pointer
   */
  explicit StageClientDeliver(InnoLidarClient *l);
  ~StageClientDeliver(void);

 public:
  /**
   * @brief Deliver stage main process
   * @param job     Inno packet
   * @param ctx     StageClientDeliver
   * @param prefer  True for normal usage, false for test usage
   * @return Return 0 for success, others for error
   */
  static int process(void *job, void *ctx, bool prefer);

 public:
  /**
   * @brief Print stage stats
   */
  void print_stats(void) const;

 private:
  /**
   * @brief Get lidar ID
   * @return Return lidar ID
   */
  const char *get_name_() const;

  /**
   * @brief Deliver stage main process
   * @param pkt     InnoCommonHeader
   * @param prefer  True for normal usage, false for test usage
   * @return Return 0 for success, others for error
   */
  int process_job_(InnoCommonHeader *pkt, bool prefer);

  /**
   * @brief Check fault
   * @param pkt Input data packet
   * @return Return 0 for success, others for error
   */
  int check_lidar_fault_in_client_(const InnoDataPacket &pkt);

  /**
   * @brief Update the result of galvo checking
   * @param check_result Golvo check result
   * @return Return 0 for success, others for error
   */
  int update_galvo_check_result_(const InnoGalvoCheckResult &check_result);

  /**
   * @brief Update the result of max distance checking
   * @param check_result Max distance check result
   * @return Return 0 for success, others for error
   */
  int update_max_distance_check_result(const InnoMaxDistanceResult &check_result);

 public:
  static const size_t kMaxXyzDataPacketBufSize = 1024 * 1024;
  static const size_t kMaxInnoDataBufferSize = 65536;

 private:
  InnoLidarClient *lidar_;
  std::mutex mutex_;

  uint64_t stats_total_jobs_;
  uint64_t stats_dropped_jobs_;
  uint64_t stats_data_jobs_;
  uint64_t stats_message_jobs_;
  uint64_t stats_status_jobs_;
  uint64_t stats_points_;
  uint64_t stats_2nd_return_points_;
  uint64_t stats_frames_;
  InnoMean convert_xyz_mean_ms_;
  InnoMean callback_mean_ms_;

  StageClientDeliverConfig config_base_;
  StageClientDeliverConfig config_;
  char xyz_data_packet_buf_[kMaxXyzDataPacketBufSize];
  InnoDataPacket *xyz_data_packet_;  // point to xyz_data_packet_buf_
  bool has_set_galvo_mirror_fault_;
  bool has_set_galvo_sensor_fault_;
  uint64_t check_galvo_times_;
  // 4bit + 4bit + 8bit
  // status + check code + speed (int km/h)
  uint16_t last_galvo_check_status_;
  // angle * 100
  uint16_t last_galvo_check_angle_;

  RingIdConverterInterface *ring_id_converter_{NULL};
  InnoSummaryPackage summary_package_;

  int cur_lidar_mode_{0};
};

/**
 * @brief StageClientDeliver2
 */
class StageClientDeliver2 {
  friend InnoLidarClient;

 public:
  /**
   * @brief StageClientDeliver2 constructor
   * @param l InnoLidarClient pointer
   */
  explicit StageClientDeliver2(InnoLidarClient *l);
  ~StageClientDeliver2(void);

 public:
  /**
   * @brief Deliver2 stage main process
   * @param job     Inno packet
   * @param ctx     StageClientDeliver2
   * @param prefer  True for normal usage, false for test usage
   * @return Return 0 for success, others for error
   */
  static int process(void *job, void *ctx, bool prefer);

 public:
  /**
   * @brief Print stage stats
   */
  void print_stats(void) const;

 private:
  /**
   * @brief Get lidar ID
   * @return Return lidar ID
   */
  const char *get_name_() const;

  /**
   * @brief Deliver2 stage main process
   * @param pkt     InnoCommonHeader
   * @param prefer  True for normal usage, false for test usage
   * @return Return 0 for success, others for error
   */
  int process_job_(InnoCommonHeader *pkt, bool prefer);

 private:
  InnoLidarClient *lidar_;
  uint64_t stats_total_jobs_;
  uint64_t stats_dropped_jobs_;
  uint64_t stats_data_jobs_;
};

}  // namespace innovusion

#endif  // SDK_CLIENT_STAGE_CLIENT_DELIVER_H_
