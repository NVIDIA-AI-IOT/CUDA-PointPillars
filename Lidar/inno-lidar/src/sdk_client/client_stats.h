/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_CLIENT_STATS_H_
#define SDK_CLIENT_CLIENT_STATS_H_

#include <limits.h>

#include "sdk_common/resource_stats.h"

namespace innovusion {
class InnoLidarClient;

/**
 * @brief ClientStats
 */
class ClientStats : public ResourceStats {
 public:
  explicit ClientStats(InnoLidarClient *l);
  ~ClientStats() {
  }

 public:
  /**
   * @brief Set extra lidar info to buffer.
   *        This methold will do nothing,
   *        except set the first char to 0
   * @param buf       Buffer to store extra info
   * @param buf_size  Buffer length
   * @param time_diff Time diff from current to last
   */
  void get_extra_info_(char *buf, size_t buf_size, double time_diff);

 private:
  InnoLidarClient *lidar_client_;
};

}  // namespace innovusion

#endif  // SDK_CLIENT_CLIENT_STATS_H_
