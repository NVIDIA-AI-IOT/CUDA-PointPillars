/**
 *  Copyright (C) 2024 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_STAGE_CLIENT_RECORDER_H_
#define SDK_CLIENT_STAGE_CLIENT_RECORDER_H_

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
 * @brief StageClientDeliver2
 */
class StageClientRecorder {
  friend InnoLidarClient;

 public:
  /**
   * @brief StageClientDeliver2 constructor
   * @param l InnoLidarClient pointer
   */
  explicit StageClientRecorder(InnoLidarClient *l);
  ~StageClientRecorder(void);

 public:
  /**
   * @brief recorder stage main process
   * @param job     Inno packet
   * @param ctx     StageClientRecorder
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
   * @brief recorder stage main process
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
