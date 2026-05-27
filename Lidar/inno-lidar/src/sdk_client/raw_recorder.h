/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_RAW_RECORDER_H_
#define SDK_CLIENT_RAW_RECORDER_H_

#include <string>
#include <unordered_map>

#include "sdk_common/inno_lidar_other_packet.h"
#include "sdk_client/lidar_client.h"
#include "utils/inno_thread.h"
/**
 * class RawReceiver
 * 1. Receive raw4 data packet with udp
 * 2. Parse received packets to raw data
 * 3. Save parsed raw data to temp file in specified folder
 * 4. Rename saved temp files to a REGULAR name
 *
 * REGULAR: sn12345-cause.inno_raw
 *          cause is the cause_ of sending raw data;
 */
namespace innovusion {
class InnoLidarClient;
class RawReceiver;

class RawDataBuf {
 public:
  RawDataBuf(char *buf, uint32_t len) {
    start = reinterpret_cast<char *>(malloc(len + 10));
    if (!start) {
      inno_log_error("malloc failed! retry");
      start = reinterpret_cast<char *>(malloc(len + 10));
      inno_log_verify(start, "malloc failed");
    }
    memcpy(start, buf, len);
    this->len = len;
  }
  ~RawDataBuf() {
    if (start != nullptr) {
      ::free(start);
      start = nullptr;
      len = 0;
    }
  }

  bool is_valid() const {
    return start && len > Raw4UdpHeader::kHeaderSize;
  }

  char *start;
  uint32_t len;
};

class RawSaver {
  /**
   * WORKING: process data in working mode
   * STOPPED: stop process data, if data added, drop it.
   * DESTROYED: free object
   */
  enum class Status {
    WORKING = 0,
    STOPPED = 1,
    DESTROYED = 2,
    MAX = 3,
  };

 public:
  static const uint64_t kExpectIdTimeOutDefaultS = 1;
  // data stream keep stopping for 30s then stop working
  static const uint64_t kDataStreamingStopTimeOutDefaultS = 30;
  // for same msg idx, if a packet come after 2min, we consider it is
  // as another data streaming
  static const uint64_t kDestroyTimeOutDefaultS = 60 * 2;
  static const char *invalid_filename_chars;
  static const uint64_t kMaxTotalRawFileSize = 100 * 1024 * 1024;

 public:
  explicit RawSaver(InnoLidarClient *lidar_client, RawReceiver *receiver, uint32_t msg_idx);
  ~RawSaver();
  // put data in cache
  void add_data(char *buf, uint32_t len);
  void shutdown();
  bool destroied() {
    // no lock here because status change from DESTROYED to others
    // is impossible
    return status_ == Status::DESTROYED;
  }

 private:
  static void *flush_thread_func(void *);
  /**
   * flush cached data into file.
   * If a expected data in sequence are later for kExpectIdTimeOutDefaultS
   * then flush anyway.
   * Update the max flashed sid to tell data receiver whether to cache a
   * data packet. --- Only cache packet that have bigger sid than flushed.
   */
  void flush_loop_();

  static void *timer_start_(void *);
  void timer_loop_();
  /**
   * write data to file
   * @param sid
   * @return true: write tmp file complete--reach the end or writing file error
   */
  bool save_to_tmp_file_(RawDataBuf *data_buf);
  void clear_cache_();
  void clear_cache_by_sid_(uint32_t sid);
  void finish_();
  void total_size_control_();

 private:
  // start thread when worker be created.
  std::thread thread_flush_;
  // info to construct final save path
  std::string sn_;
  std::string cause_;
  std::string tmp_path_;
  std::string save_path_;
  uint32_t msg_idx_;
  int tmp_fd_;

  RawReceiver *receiver_;
  /**
   * current expected_id
   * Every time flush thread processed a packet, it will update the value.
   * If add_data get the expected id, then signal flush thread.
   * If add_data get a smaller id, then drop it.
   * If add_data get a bigger id, then push it to cache without signal.
   */

  uint32_t expect_id_ = 0;

  /**
   * cache raw data, the memory is allocated by receiver.
   * free after data is writen to file.
   */
  std::unordered_map<uint32_t, RawDataBuf *> cache_;
  std::mutex mutex_cache_;

  /**
   * If data flush thread can't get the expected id, then wait
   * If wake up, then flush the smallest id in cache
   */
  std::condition_variable cond_cache_;

  // timer thread
  std::thread thread_timer_;
  Status status_;
  std::mutex mutex_status_;
  /**
   * In WORKING status:
   * Timer thread will wait for kDataStreamingStopTimeOutDefaultS seconds,
   * if time out then switch to stopped status.
   * Signal when receive a data packet to wake timer in time
   * to keep WORKING status.
   *
   * In STOPPED status:
   * Timer thread will wait for kDestroyTimeOutDefaultS seconds
   * then switch to DESTROYED status
   */
  std::condition_variable cond_status_;

  // statistic
  int32_t last_flush_sid_ = -1;
  bool incomplete_ = false;

  InnoLidarClient *lidar_client_;
};

/**
 * Receiver listen udp port and dispatch received data to
 * saver corresponding to msg idx. Create a saver if it not exist.
 * Destroy a saver if the saver's status is DESTROYED.
 * The saver know when to destroy itself.
 * There are one data receiver and several savers.
 */
class RawReceiver {
 public:
  static const uint32_t kMaxMapSize = 100;

 public:
  RawReceiver(InnoLidarClient *l, int udp_port, std::string raw_recoder_save_path_);
  ~RawReceiver() = default;
  static void *start(void *ctx);

 private:
  void *receive_loop_();

 public:
  int udp_port;
  std::string save_path;

 private:
  InnoLidarClient *lidar_client_;
  std::unordered_map<uint32_t, RawSaver *> msgid_saver_map_;
};
}  // namespace innovusion
#endif  //  SDK_CLIENT_RAW_RECORDER_H_
