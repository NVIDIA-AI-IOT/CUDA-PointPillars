/**
 *  Copyright (C) 2024 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/stage_client_recorder.h"
#include <utility>

#include "sdk_client/lidar_client.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/consumer_producer.h"

namespace innovusion {

StageClientRecorder::StageClientRecorder(InnoLidarClient *l) {
  lidar_ = l;
  stats_total_jobs_ = 0;
  stats_dropped_jobs_ = 0;
  stats_data_jobs_ = 0;
}

StageClientRecorder::~StageClientRecorder(void) {
}

int StageClientRecorder::process(void *in_job, void *ctx, bool prefer) {
  StageClientRecorder *s = reinterpret_cast<StageClientRecorder *>(ctx);
  return s->process_job_(reinterpret_cast<InnoCommonHeader *>(in_job), prefer);
}

int StageClientRecorder::process_job_(InnoCommonHeader *pkt, bool prefer) {
  stats_total_jobs_++;
  if (!prefer) {
    stats_dropped_jobs_++;
    if (stats_dropped_jobs_ % 10 == 1) {
      inno_log_warning("drop data in recorder stage.");
      print_stats();
    }
  } else {
    lidar_->do_recorder_callback(INNO_RECORDER_CALLBACK_TYPE_INNO_PC, reinterpret_cast<char *>(pkt), pkt->size);
  }
  lidar_->free_buffer_(pkt);
  return 0;
}

void StageClientRecorder::print_stats() const {
  inno_log_info(
      "StageClientRecorder: "
      "total=%" PRI_SIZEU " total_dropped=%" PRI_SIZEU "data=%" PRI_SIZEU,
      stats_total_jobs_, stats_dropped_jobs_, stats_data_jobs_);
}

}  // namespace innovusion
