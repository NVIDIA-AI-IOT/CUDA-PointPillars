/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/client_stats.h"

#include "sdk_client/lidar_client.h"
#include "utils/consumer_producer.h"
#include "utils/log.h"

namespace innovusion {

ClientStats::ClientStats(InnoLidarClient *l) : ResourceStats(l) {
  lidar_client_ = l;
}

void ClientStats::get_extra_info_(char *buf, size_t buf_size, double time_diff) {
  buf[0] = 0;
}

}  // namespace innovusion
