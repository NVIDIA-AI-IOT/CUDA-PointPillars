/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_STAGE_CLIENT_READ_H_
#define SDK_CLIENT_STAGE_CLIENT_READ_H_

#if !(defined(__MINGW64__) || defined(_WIN32))
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/lidar_base.h"
#include "utils/config.h"
// #include "sdk/types_consts.h"

namespace innovusion {
class InnoLidarClient;
class BaseInput;
class LidarClientCommunication;

/**
 * @brief StageClientReadConfig
 */
class StageClientReadConfig : public Config {
 public:
  StageClientReadConfig() : Config() {
    misorder_correct_enable = 0;
  }

  /**
   * @brief   Get config name
   * @return  Return config name
   */
  const char *get_type() const override {
    return "LidarClient_StageClientRead";
  }

  /**
   * @brief Update configurations
   * @param key   Config key
   * @param value Config value
   * @return Ignored
   */
  int set_key_value_(const std::string &key, double value) override {
    SET_CFG(misorder_correct_enable);
    return 0;
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
  int misorder_correct_enable;
  END_CFG_MEMBER()
};

using deliver_packet_callback_t = std::function<void(InnoCommonHeader *)>;
using deliver_packet_after_correct_callback_t = std::function<void(InnoDataPacket *)>;
using fatal_message_callback_t = std::function<void()>;
class BaseInput {
 public:
  explicit BaseInput(InnoLidarClient *lidar, const void *param) {
    input_param_ = *reinterpret_cast<const InputParam *>(param);
    lidar_ = lidar;
    source_ = input_param_.base_param.source_type;
  }

  /**
   * @brief  Start data input process
   */
  virtual bool start() {
    start_flag_ = true;
    return true;
  }
  /**
   * @brief  stop data input process
   */
  virtual void stop() {
    if (start_flag_) {
      start_flag_ = false;
    }
  }
  virtual int read_data() = 0;

  /**
   * @brief  enable misorder data packet correct flag
   * @param enable 1 enable, 0 disable
   */
  void set_misorder_correct_enable(int enable) {
    misorder_correct_enable_ = enable;
  }
  /**
   * @brief  set data packet deliver function handler
   * @param callback data packet deliver function handler
   */
  void set_deliver_packet_callback(deliver_packet_callback_t callback) {
    deliver_packet_callback_ = callback;
  }
  /**
   * @brief  set data packet deliver with correct function handler
   * @param callback data packet deliver with correct function handler
   */
  void set_deliver_packet_after_correct_callback(deliver_packet_after_correct_callback_t callback) {
    deliver_packet_after_correct_callback_ = callback;
  }
  /**
   * @brief  set send fatal message function handler
   * @param callback send fatal message function handler
   */
  void set_fatal_message_callback(fatal_message_callback_t callback) {
    send_fatal_message_callback_ = callback;
  }
  virtual ~BaseInput() = default;

  void set_first_step(bool first_step) {
    first_step_ = first_step;
  }
  bool get_first_step() {
    return first_step_;
  }

 protected:
  InnoLidarClient *lidar_ = NULL;
  int misorder_correct_enable_ = 0;
  enum InputSource source_;
  InputParam input_param_;
  deliver_packet_callback_t deliver_packet_callback_ = nullptr;
  deliver_packet_after_correct_callback_t deliver_packet_after_correct_callback_ = nullptr;
  fatal_message_callback_t send_fatal_message_callback_ = nullptr;
  bool start_flag_ = false;
  static const size_t kMaxReadSize = 65536;
  bool first_step_ = true;
};

class FileInput : public BaseInput {
 public:
  /**
   * @brief  FileInput constructor
   * @param lidar lidar handler
   * @param param InputParam
   */
  explicit FileInput(InnoLidarClient *lidar, const void *param) : BaseInput(lidar, param) {
    filename_ = input_param_.file_param.filename;
    if (input_param_.file_param.source_type != SOURCE_TCP) {
      set_file_play_rate();
      inno_log_info("filename: %s, play_round: %d", filename_, play_round_);
    }
  }

  virtual ~FileInput() = default;
  /**
   * @brief  Read data with loop
   * @return Return 0 for success, others for error
   */
  int read_data();

  /**
   * @brief  Set playback file play rate
   */
  void set_file_play_rate();

 protected:
  virtual int read_fd_(char *buf, size_t len);
  /*
   * @brief Read one data packet or message packet or status packet
   *        from fd
   * @param data_packet Pointer to the buffer that the received data
   *                    packet will be written to
   * @param data_len The maximum len the data_packet can be, if
   *                 the size is not enough, the function will return -3.
   *                 If one data_packet is received, data_len will be
   *                 set to the size of the packet, otherwise set to 0.
   * @param message_packet Pointer to the buffer that the received
   *                       message packet will be written to
   * @param message_len The maximum len the message_packet can be, if
   *                 the size is not enough, the function will return -3.
   *                 If one message_packet is received, message_len will be
   *                 set to the size of the packet, otherwise set to 0.
   * @param status_packet Pointer to the buffer that the received status
   *                      packet will be written to.
   * @param status_len The maximum len the status_packet can be, if
   *                   the size is not enough, the function will return -3
   *                   If one status_packet is received, status_len will
   *                   ve set to the size of the packet, otherwise set to 0.
   * @param is_file The fd is for a file, not a network socket.
   *
   * @return 0 means success
   *         -1 means file ends or connection disconnected before
   *            enough data was read
   *         -2 recieved data is not valid
   *         -3 the buffers are too small
   */
  int read_packet_(InnoDataPacket *data_packet, size_t *data_len, InnoDataPacket *message_packet, size_t *message_len,
                   InnoStatusPacket *status_packet, size_t *status_len, int &major_version, bool &anglehv_table);

  /**
   * @brief Control reading rate
   * @param last_data_us Recived data timestamp
   * @param r            Recived data count
   */
  void read_file_rate_control_(InnoTimestampUs last_data_us, int r);
  /**
   * @brief Read from fd and send tp deliver stage
   * @param fd      Reading fd
   * @param is_file Is file or not
   * @return Return 0 for success, others for error
   */
  int keep_reading_();
  /**
   * @brief Set the timestamp of the last data_packet, Delay settings for file playback
   * @param latest_data_us      he timestamp of the last data_packet
   */
  void set_latest_data_us(InnoTimestampUs latest_data_us) {
    latest_data_us_ = latest_data_us;
  }

 protected:
  int fd_ = -1;
  bool cannot_open_file_ = false;
  bool reach_file_end_ = false;
  int32_t play_round_ = 0;
  const char *filename_ = NULL;
  int32_t play_rate_ = 0;
  double play_rate_x_ = 0.0;
  InnoTimestampUs start_time_us_ = 0;
  InnoTimestampUs first_data_us_ = 0;
  InnoTimestampUs latest_data_us_ = 0;
  size_t total_byte_received_ = 0;

 private:
  FileInput() = delete;
  FileInput(const FileInput &) = delete;
  FileInput operator=(const FileInput &) = delete;
};

class UdpInput : public BaseInput {
 public:
  uint32_t kPrintPacketDiffIntervalMs = 5 * 1000;
  /**
   * @brief UdpInput constructor
   * @param lidar lidar handler
   * @param param InputParam
   */
  explicit UdpInput(InnoLidarClient *lidar, const void *param) : BaseInput(lidar, param) {
    input_param_ = *reinterpret_cast<const InputParam *>(param);
    udp_port_ = input_param_.udp_param.udp_port;
  }

  virtual ~UdpInput();
  /**
   * @brief  Read data with loop
   * @return Return 0 for success, others for error
   */
  int read_data() override;

 private:
  /**
   * @brief Bind UDP port
   * @param port UDP port id
   * @return Return UDP socket fd, positive integer for success, others for error
   */
  int bind_udp_port_(uint16_t port);
  /**
   * @brief Wait until current state is STOPPING or STOPPED
   */
  void wait_until_stopping_();
  /**
   * @brief Read data/message/satus UDP and send to deliver stage
   * @param port UDP port id
   * @return Return 0 for success, others for error
   */
  int read_udp_(int32_t port);

  bool verify_inno_data_packet_counter(const InnoDataPacket &pkt, bool* same_data_sub_seq);
  bool verify_inno_status_packet_counter(const InnoStatusPacket &pkt, bool* same_status_sub_seq);
  bool verify_counter_process(int current, int last);

 private:
  uint16_t udp_port_ = 0;
  struct ip_mreq mreq_;
  bool use_mreq_ = false;
  int max_delta_counter_init_ = 3;
  int max_no_new_repeate_data_ = 0;
  int inno_data_last_sub_seq_ = 0;
  uint16_t inno_status_last_idx_ = 0;
  uint16_t max_value_ = UINT16_MAX;
  uint16_t inno_data_first_step_ = 0;
  bool inno_status_first_step_ = true;
  uint16_t sub_seq_zero_counter_ = 0;
  uint64_t inno_data_diff3_error_counter_ = 0;
  uint64_t last_print_inno_data_diff3_error_counter_ = 0;
  uint64_t last_print_data_diff_time_ = 0;
  uint64_t inno_status_diff3_error_counter_ = 0;
  uint64_t last_print_inno_status_diff3_error_counter_ = 0;
  uint64_t last_print_status_diff_time_ = 0;
  uint64_t inno_data_repeat_error_counter_ = 0;
  uint64_t inno_status_repeat_error_counter_ = 0;

 private:
  UdpInput() = delete;
  UdpInput(const UdpInput &) = delete;
  UdpInput operator=(const UdpInput &) = delete;
};

class TcpInput : public FileInput {
 public:
  /**
   * @brief UdpInput constructor
   * @param lidar lidar handler
   * @param lm LidarClientCommunication handler
   * @param param InputParam
   */
  explicit TcpInput(InnoLidarClient *lidar, LidarClientCommunication *lm, const void *param) : FileInput(lidar, param) {
    lidar_comm_ = lm;
  }
  /**
   * @brief  Read data with loop
   * @return Return 0 for success, others for error
   */
  int read_data() override;
  virtual ~TcpInput() = default;

 protected:
  /**
   * @brief  Read data of specified length into buf
   * @param buf     buffer for output data
   * @param len     specified lengthr
   * @return Return len for success, others for error
   */
  int read_fd_(char *buf, size_t len) override;

 private:
  TcpInput() = delete;
  TcpInput(const TcpInput &) = delete;
  TcpInput operator=(const TcpInput &) = delete;
  explicit TcpInput(InnoLidarClient *lidar, const void *param) : FileInput(lidar, param) {
  }
  bool read_next_packet_();

 private:
  LidarClientCommunication *lidar_comm_;
};

/**
 * @brief Queue of cached disordered frames
 */
class MisorderBufQueue {
  struct InnoDataPacketComparator {
    /**
     * @brief Operator functions for custom sorting
     */
    bool operator()(const InnoDataPacket *a, const InnoDataPacket *b) const {
      if (a->idx != b->idx) {
        return a->idx > b->idx;  // idx small in front
      } else {
        return a->sub_idx > b->sub_idx;  // sub_idx small in front
      }
    }
  };

 public:
  explicit MisorderBufQueue(size_t q_len = 10) {
    inno_log_verify(q_len > 0, "invalid q_len=%" PRI_SIZED, q_len);
    max_q_len_ = q_len;
  }
  ~MisorderBufQueue() {
  }
  MisorderBufQueue(const MisorderBufQueue &rhs) = delete;
  MisorderBufQueue &operator=(const MisorderBufQueue &rhs) = delete;

 public:
  inline void push(InnoDataPacket *job) {
    pq_.emplace(job);
  }

  inline void pop() {
    pq_.pop();
  }

  inline InnoDataPacket *top() {
    return pq_.top();
  }

  inline bool empty() {
    return pq_.empty();
  }

  inline bool full() {
    return pq_.size() > max_q_len_;
  }

  inline size_t size() {
    return pq_.size();
  }

 private:
  std::priority_queue<InnoDataPacket *, std::vector<InnoDataPacket *>, InnoDataPacketComparator> pq_;
  size_t max_q_len_ = 0;
};

class StageClientRead {
  friend InnoLidarClient;

 public:
  static const int32_t kResetMisorderQueueFrameIdxDiff = 10;
  /**
   * @brief Read stage main function
   * @param job     Ignored
   * @param ctx     StageClientRead object pointer
   * @param prefer  Ignored
   * @return Return 0 for success, others for error
   */
  static int process(void *job, void *ctx, bool prefer);

 public:
  /**
   * @brief StageClientRead constructor, read from file
   * @param l              InnoLidarClient pointer
   * @param filename       Source data filename
   * @param play_rate      Play rate
   * @param play_rate_x
   * @param rewind         Rewind
   * @param file_open      Specify file open function
   * @param file_read      Specify file read function
   * @param file_close     Specify file close function
   * @param ctx            Specify file operation functions context
   */
  StageClientRead(InnoLidarClient *l, LidarClientCommunication *lm, void *ctx);
  ~StageClientRead(void);

 public:
  /**
   * @brief Stop stage
   */
  void stop(void);
  /**
   * @brief Notify cleanup complete
   */
  void final_cleanup(void);
  /**
   * @brief Get stage current state
   * @return Return stage current state
   */
  enum InnoLidarBase::State get_state() {
    std::unique_lock<std::mutex> lk(mutex_);
    enum InnoLidarBase::State ret = state_;
    return ret;
  }
  /**
   * @brief Print stage stats
   */
  void print_stats(void) const;

 private:
  /**
   * @brief Initialize configurations
   * @param l InnoLidarClient
   */
  void init_(InnoLidarClient *l);
  /**
   * @brief Get lidar ID
   * @return Return lidar ID
   */
  const char *get_name_(void) const;

  /**
   * @brief Notify start reading
   */
  void start_reading_(void);
  /**
   * @brief The main process functoin
   * @param in_job Ignored
   * @return Return 0 for success, others for error
   */
  int process_job_(void *in_job);
  /**
   * @brief Send fault message to lidar
   */
  void send_fatal_message_();
  /**
   * @brief Wait until current state is STOPPING or STOPPED
   */
  void wait_until_stopping_();
  /**
   * @brief Get state STOPPING or STOPPED
   * @return Return true means current state is STOPPING or STOPPED, otherwise not
   */
  bool stopping_or_stopped_();
  /**
   * @brief Add packets to deliver stage
   * @param header InnoCommonHeader
   */
  void add_deliver_packet_(InnoCommonHeader *header);
  /**
   * @brief Add packets to deliver stage after correct
   * @param data_packet InnoDataPacket
   */
  int add_deliver_packet_after_correct_(InnoDataPacket *data_packet);
  /**
   * @brief create data input source according to different type
   * @param ctx InputParam context
   */
  bool create_input_(const void *ctx);
  /**
   * @brief Send the cached data_packet out
   */
  void flush_cached_packet();
  /**
   * @brief free all the cached data_packet
   */
  void free_cached_packet();

 private:
  InnoLidarClient *lidar_;

  StageClientReadConfig config_base_;
  StageClientReadConfig config_;

  MisorderBufQueue misorder_que_;
  uint64_t expected_sub_idx_ = -1;
  uint64_t expected_idx_ = -1;

  enum InputSource source_;
  LidarClientCommunication *lidar_comm_ = NULL;
  enum InnoLidarBase::State state_;
  std::mutex mutex_;
  std::condition_variable cond_;
  BaseInput *input_ = NULL;
};

}  // namespace innovusion

#endif  // SDK_CLIENT_STAGE_CLIENT_READ_H_
