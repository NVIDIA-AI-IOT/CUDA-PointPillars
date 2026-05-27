/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_common/resource_stats.h"

#if !(defined(__MINGW64__) || defined(_WIN32))
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string>
#include <sstream>

#include "sdk_common/lidar_base.h"
#include "utils/consumer_producer.h"
#include "utils/log.h"

namespace innovusion {
const char *ResourceStats::packet_type_names[PACKET_TYPE_MAX] = {
  "READ", "DATA", "POINT", "MESSAGE", "STATUS"
};

ResourceStats::ResourceStats(InnoLidarBase *l) {
  inno_log_verify(l, "invalid lidar");
  lidar_ = l;
  lastp_time_ms_ = start_time_ms_ = lidar_->get_monotonic_raw_time_ms();
  last_print_stats_time_ms_ = last_update_time_ms_ = lastp_time_ms_;
  show_cpu_count_ = 0;
  print_stats_count_ = 0;
  last_output_stats_buffer_[0] = 0;
  total_ref_count_ = 0;
  total_ref_intensity_ = 0;

  for (int i = 0; i < PACKET_TYPE_MAX; i++) {
    lastp_byte_[i] = total_byte_[i] = 0;
    lastp_packet_[i] = total_packet_[i] = 0;
  }
}

void ResourceStats::update_packet_bytes(enum PacketType type,
                                        size_t packet, size_t byte,
                                        size_t ref_count_sum,
                                        uint64_t intensity_sum) {
  inno_log_verify(type < PACKET_TYPE_MAX, "type = %d", type);
  total_byte_[type] += byte;
  total_packet_[type] += packet;
  total_ref_count_ += ref_count_sum;
  total_ref_intensity_ += intensity_sum;
  periodically_show();
}

void ResourceStats::periodically_show() {
  static const int kPrintIntervalMs = 50000;   // 50 seconds
  static const int kShowLidarStatsIntervalMs = 2 * 60 * 1000;  // 2 mins
  int64_t now_ms = lidar_->get_monotonic_raw_time_ms();
  bool do_show = false;
  bool do_print_stats = false;
  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (now_ms - last_update_time_ms_ > kPrintIntervalMs ||
        (show_cpu_count_ == 0 && now_ms - last_update_time_ms_ > 2500)) {
      last_update_time_ms_ = now_ms;
      show_cpu_count_++;
      do_show = true;
    }
    if (now_ms - last_print_stats_time_ms_ > kShowLidarStatsIntervalMs ||
         (print_stats_count_ == 0 &&
          now_ms - last_print_stats_time_ms_ >10 * 1000)) {
      last_print_stats_time_ms_ = now_ms;
      do_print_stats = true;
      print_stats_count_++;
    }
  }
  if (do_show) {
    show();
  }
  if (do_print_stats) {
    lidar_->print_stats();
    // add the debuginfo
    innovusion::InnoLog::get_instance().asynclog_info();
  }
}

void ResourceStats::show() {
  char buf[512];
  int64_t now_ms = lidar_->get_monotonic_raw_time_ms();
  double diff_s = (now_ms - lastp_time_ms_) / 1000.0;
  if (diff_s <= 0) {
    inno_log_warning("%s strange time diff %f, set to 1.0",
                     lidar_->get_name(), diff_s);
    diff_s = 1.0;
  }
  int print_off = 0;
  for (int i = 0; i < PACKET_TYPE_MAX; i++) {
    int pr = snprintf(buf + print_off, sizeof(buf) - print_off,
                      " <%s> %s=%" PRI_SIZEU "/%" PRI_SIZEU
                      ", %s=%" PRI_SIZEU "K/%" PRI_SIZEU "K, "
                      "%s=%.2fM/s"
                      "%s%s%s%s;",
                      packet_type_names[i],
                      i == PACKET_TYPE_POINT ? "frames": "packets",
                      total_packet_[i] - lastp_packet_[i],
                      total_packet_[i],
                      i == PACKET_TYPE_POINT ? "points": "bytes",
                      (total_byte_[i] - lastp_byte_[i]) >> 10,
                      (total_byte_[i]) >> 10,
                      i == PACKET_TYPE_POINT ? "point-rate": "bandwidth",
                      (total_byte_[i] - lastp_byte_[i]) / 1000000.0 / diff_s,
                      i == PACKET_TYPE_POINT ? ", ref_intensity_sum = " : "",
                      i == PACKET_TYPE_POINT ?
                           std::to_string(total_ref_intensity_).c_str() : "",
                      i == PACKET_TYPE_POINT ? ", ref_count_total = " : "",
                      i == PACKET_TYPE_POINT ?
                           std::to_string(total_ref_count_).c_str() : "");
    if (print_off + pr >= ssize_t(sizeof(buf))) {
      inno_log_error("stats buffer to small %d", print_off + pr);
      return;
    }
    print_off += pr;
  }
  double read_band_width = (total_byte_[PACKET_TYPE_SRC] -
                            lastp_byte_[PACKET_TYPE_SRC]) /
                            1000000.0 / diff_s;
  static uint32_t bandwidth_low_counter = 0;
  enum InnoLidarMode mode = INNO_LIDAR_MODE_WORK_NORMAL;
  enum InnoLidarMode pre_mode = INNO_LIDAR_MODE_WORK_NORMAL;
  enum InnoLidarStatus status = INNO_LIDAR_STATUS_NORMAL;
  uint64_t transition_mode_ms = 0;
  int ret = lidar_->\
            get_mode_status(&mode, &pre_mode, &status, &transition_mode_ms);
  if (ret == 0 && status == INNO_LIDAR_STATUS_NORMAL &&
     (mode == INNO_LIDAR_MODE_WORK_NORMAL ||
      mode == INNO_LIDAR_MODE_WORK_CALIBRATION ||
      mode == INNO_LIDAR_MODE_WORK_EXHIBITION)) {
    if (read_band_width < kStageReadBandWidthThresholdMBps &&
      (bandwidth_low_counter++ < 30 || bandwidth_low_counter % 16)) {
      inno_log_warning("<READ> bandwidth too low: %0.5f, counter: %u",
                        read_band_width, bandwidth_low_counter);
    }
  }

  char buf_extra[512];
  buf_extra[0] = 0;
  // RW-MARK ?
  get_extra_info_(buf_extra, sizeof(buf_extra), diff_s);
  // inno_log_info("%s pid=%d uptime=%.2fs, #RESOURCE_STATS#%s %s", lidar_->get_name(), getpid(),
  //               (now_ms - start_time_ms_) / 1000.0, buf, buf_extra);
  {
    std::unique_lock<std::mutex> lk(mutex_);
    strncpy(last_output_stats_buffer_, buf, sizeof(last_output_stats_buffer_));
    last_output_stats_buffer_[sizeof(last_output_stats_buffer_) - 1] = 0;
  }
  for (int i = 0; i < PACKET_TYPE_MAX; i++) {
    lastp_byte_[i] = total_byte_[i];
    lastp_packet_[i] = total_packet_[i];
  }

  lastp_time_ms_ = now_ms;
  return;
}

void ResourceStats::get_last_output_info_buffer(char *buf,
                                                size_t buf_size) {
  std::unique_lock<std::mutex> lk(mutex_);
  inno_log_verify(buf && buf_size > 0,
                  "buf_size=%" PRI_SIZELU, buf_size);
  strncpy(buf, last_output_stats_buffer_, buf_size);
  buf[buf_size - 1] = 0;
  return;
}

}  // namespace innovusion
