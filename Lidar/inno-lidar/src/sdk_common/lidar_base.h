/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_COMMON_LIDAR_BASE_H_
#define SDK_COMMON_LIDAR_BASE_H_

#include <map>
#include <mutex>
#include <string>

#include <limits>

#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/resource_stats.h"
#include "sdk_common/ring_id_converter_interface.h"
#include "utils/config.h"
#include "utils/log.h"
#include "utils/utils.h"

namespace innovusion {
/**
 * @brief InnoLidarBase
 */
class InnoLidarBase {
 public:
  enum State {
    STATE_INIT = 0,
    STATE_READING,
    STATE_STOPPING,
    STATE_STOPPED,
  };
  static const uint8_t kDefaultFrameRate = 10;

 public:
  /**
   * @brief InnoLidarBase constructor
   * @param config_name   Lidar config name
   * @param name          Lidar ID
   */
  InnoLidarBase(const char *config_name, const char *name) : config_manage_(config_name) {
    name_ = strdup(name);
    handle_ = 0;
    get_host_time_ = NULL;
    message_callback_external_ = NULL;
    data_packet_callback_ = NULL;
    status_packet_callback_ = NULL;
    get_host_time_ = get_host_time_default;

    for (uint32_t i = 0; i < INNO_RECORDER_CALLBACK_TYPE_MAX; i++) {
      recorder_callbacks_[i] = NULL;
      recorder_callbacks_context_[i] = NULL;
    }
    cpusetsize_ = 0;
    cpuset_ = NULL;
    exclude_callback_thread_ = 0;

    last_stage_is_up_ = false;
    callback_context_ = NULL;
  }

  virtual ~InnoLidarBase() {
    if (name_) {
      free(name_);
      name_ = NULL;
    }
  }

 public:  // static methods
  /**
   * @brief Add Lidar base to Lidar base map
   * @param l Address of Lidar base
   * @return Return Lidar base handle
   */
  static int add_lidar(InnoLidarBase *l);

  /**
   * @brief Remove & delete Lidar base from map according to Liadr handle
   * @param handle Lidar handle to be removed
   * @return Return 0 if success, others for error
   */
  static int remove_lidar(int handle);

  /**
   * @brief Remove & delete all Lidar base from map
   * @return The count of Lidar removed
   */
  static int remove_lidar_all();

  /**
   * @brief Stop all Lidar
   * @return The count of Lidar stopped
   */
  static int stop_lidar_all();

  /**
   * @brief Get Lidar base address according to Lidar handle
   * @param handle Lidar handle
   * @return Address of target Lidar base
   */
  static InnoLidarBase *find_lidar(int handle);

  /**
   * @brief Get system time in second, it's the default host time function
   * @param context Ignored. The context of get host time function
   * @return Return system time in second
   */
  static double get_host_time_default(void *);

 public:
  /**
   * @brief Get current host time in second
   * @return Return current host time in second
   */
  inline double get_current_host_time() {
    return get_host_time_(callback_context_);
  }

  /**
   * @brief Get monotonic time in nanosecond
   * @return Return monotonic time in nanosecond
   */
  inline double get_monotonic_raw_time() {
    return InnoUtils::get_time_ns() / 1000000000.0;
  }

  /**
   * @brief Get monotonic time in microsecond
   * @return Return monotonic time in microsecond
   */
  inline static double get_monotonic_raw_time_us() {
    return InnoUtils::get_time_ns() / 1000.0;
  }

  /**
   * @brief Get monotonic time in millisecond
   * @return Return monotonic time in millisecond
   */
  inline static double get_monotonic_raw_time_ms() {
    return InnoUtils::get_time_ns() / 1000000.0;
  }

  /**
   * @brief Get monotonic time in second
   * @return Return monotonic time in second
   */
  inline uint32_t get_monotonic_raw_time_sec() {
    return InnoUtils::get_time_ns() / 1000000000;
  }

 public:
  /**
   * @brief Read ps register
   * @param off   The data offset
   * @param value The data container
   * @return Return 0 for success, others for error
   */
  virtual int read_ps_reg(uint16_t off, uint32_t *value) = 0;

  /**
   * @brief Read pl register
   * @param off   The data offset
   * @param value The data container
   * @return Return 0 for success, others for error
   */
  virtual int read_pl_reg(uint16_t off, uint32_t *value) = 0;

  /**
   * @brief Write ps register
   * @param off   The data offset
   * @param value The data container
   * @return Return 0 for success, others for error
   */
  virtual int write_ps_reg(uint16_t off, uint32_t value) = 0;

  /**
   * @brief Write pl register
   * @param off   The data offset
   * @param value The data container
   * @return Return 0 for success, others for error
   */
  virtual int write_pl_reg(uint16_t off, uint32_t value) = 0;

  /**
   * @brief Load params from yaml file
   * @param lidar_model   Lidar model
   * @param yaml_filename The params file Lidar read from
   * @return Return 0 for success, others for error
   */
  virtual int set_params_file(const char *lidar_model, const char *yaml_filename) = 0;

  /**
   * @brief Set key & value to config
   * @param name  Key
   * @param value Value
   * @return Return 0 for success, others for error
   */
  virtual int set_config_name_value(const char *name, const char *value) = 0;

  /**
   * @brief Set InnoReflectanceMode to Lidar server
   * @param mode InnoReflectanceMode
   * @return Return 0 for success, others for error
   */
  virtual int set_reflectance_mode(enum InnoReflectanceMode) = 0;

  /**
   * @brief Set InnoMultipleReturnMode to Lidar server
   * @param ret_mode InnoMultipleReturnMode
   * @return Return 0 for success, others for error
   */
  virtual int set_return_mode(InnoMultipleReturnMode ret_mode) = 0;

  /**
   * @brief Set ROI to Lidar server
   * @param h_angle ROI horizontal center (in [-60, 60] degrees)
   * @param v_angle ROI vertical center (in [-13, 13] degrees)
   *        Setting either horz_angle or vert_angle to
   *        kInnoNopROI, i.e. 10000.0
   *        will maintain the previous value for that angle.  Any other value
   *        outside the allowed range will result in a failure code return and
   *        no change in ROI center.
   * @return Return 0 for success, others for error
   */
  virtual int set_roi(double hori_angle, double v_angle) = 0;

  /**
   * @brief Get ROI from Lidar server
   * @param h_angle ROI horizon angle center
   * @param v_angle ROI vertical angle center
   * @return Return 0 for success, others for error
   */
  virtual int get_roi(double *h_angle, double *v_angle) = 0;

  /**
   * @brief Set mode to Lidar server and get the mode & status before set
   * @param mode                 The mode to be set
   * @param mode_before_change   Current mode before mode set
   * @param status_before_change Current status before mode set
   * @return Return 0 for success, others for error
   */
  virtual int set_mode(enum InnoLidarMode mode, enum InnoLidarMode *mode_before_change,
                       enum InnoLidarStatus *status_before_change) = 0;

  /**
   * @brief Get current mode & previous mode & current status from Lidar server
   * @param mode                  Address to store current mode
   * @param pre_mode              Address to store previous mode
   * @param status                Address to store current status
   * @param in_transition_mode_ms Time (in ms) stay in transition mode
   * @return Return 0 for success, others for error
   */
  virtual int get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode, enum InnoLidarStatus *status,
                              uint64_t *in_transition_mode_ms) = 0;

  /**
   * @brief Get anglehv table from Lidar server
   * @param anglehv_table     Buffer to store table
   * @return Return 0 for success, others for error
   */
  virtual int get_anglehv_table(InnoDataPacket *anglehv_table) = 0;

  /**
   * @brief Get attribute from Lidar server
   * @param attribute Name of the attribute
   * @param value     Buffer to store attribute value
   * @return Return 0 for success, others for error
   */
  virtual int get_attribute(const char *attribute, double *value) = 0;

  /**
   * @brief Get attribute from Lidar server
   * @param attribute Name of the attribute
   * @param buf       Buffer to store attribute string value
   * @param buf_size  Buffer size
   * @return Return 0 for success, others for error
   */
  virtual int get_attribute_string(const char *attribute, char *buf, size_t buf_size) = 0;

  /**
   * @brief Set lidar attribute
   * @param attribute Name of the attribute
   * @param buffer    Buffer to store the string value
   * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
   */
  virtual int set_attribute_string(const char *attribute, const char *buf) = 0;

  /**
   * @brief Set fault value to Lidar server
   * @param value_hex_str fault value
   * @return Return 0 for success, others for error
   */
  virtual int set_faults_save_raw(const std::string &value_hex_str) = 0;

  /**
   * @brief Set the velocity and angular velocity for motion correction
   * @param velocity_m_s[3]:           Velocities in x, y, z axis, unit is m/s
   * @param angular_velocity_rad_s[3]: Angular velocities in x, y, z axis, RAD/s
   * @return 0 means success, -1 if handle is invalid
   */
  virtual int set_motion_compensation(double velocity[3], double angular_velocity[3]) = 0;

  /**
   * @brief Specify the CPU affinity for all threads created by the
   *        library for this lidar instance. This function can only
   *        be called before inno_lidar_start() is called. Please see
   *        pthread_setaffinity_np() for reference.
   * @param cpusetsize              Size of *cpuset.
   * @param cpuset                  Cupset that will be passed to pthread_setaffinity_np().
   * @param exclude_callback_thread 1 means do not set affinity for callback
   *        thread, 0 otherwise.
   * @return 0 means successful stored the affinity settings that will be
   *         passed in when calling pthread_setaffinity_np(). It doesn't
   *         not guarantee that pthread_setaffinity_np() will be successful.
   *         1 invalid handle
   */
  virtual int thread_setaffinity_np(size_t cpusetsize, const cpu_set_t *cpuset, int exclude_callback_thread) = 0;

  /**
   * @brief Query lidar unit's state
   * @param state       Address to store InnoLidarState
   * @param error_code  Address to store error code. NULL means
   *        do not store error code
   * @return 0 means success, -1 if handle is invalid
   */
  virtual int get_fw_state(InnoLidarState *state, int *error_code) = 0;

  /**
   * @brief Query lidar unit's firmware version.
   * @param buffer     Buffer to store the firmware version.
   * @param Buffer_len Length of buffer, the recommended buffer_len is
   *        512, the size of buffer needs to be >= buffer_len.
   * @return 0 cannot get firmware version
   *        -1 invlid lidar handle
   *        -2 buffer_len is too small
   *        -3 source is file
   *        otherwise return the size of firmware version string
   *          stored in buffer, not include the trailing zero
   *        Sample content in the buffer:
   *           App Version: app-2.3.0-rc8.134
   *             build-time: 2019-08-14-18-19-25
   *           FPGA Datecode: 0x190814e2
   *             fpga-ver: 0x13
   *             fpga-rev: 0x0e
   *             board-rev: 0x2
   *           Firmware Version: 2.3.1-rc3-418.2019-08-15-17-41-40
   *             build-tag: 2.3.1-rc3-418
   *             build-time: 2019-08-15-17-41-40
   *             build-git-tag: 1.0.19
   */
  virtual int get_fw_version(char *buffer, int buffer_len) = 0;

  /**
   * @brief Query lidar unit's S/N
   * @param buffer     Buffer to store the S/N.
   * @param buffer_len Length of buffer, the recommended buffer_len is
   *        128, the size of buffer needs to be >= buffer_len.
   * @return 0 cannot get S/N
   *         -1 invlid lidar handle
   *         -2 buffer_len is too small
   *         -3 source is file
   *         otherwise return the size of S/N string
   *         stored in buffer, not include the trailing zero
   */
  virtual int get_sn(char *buffer, int buffer_len) = 0;

  /**
   * @brief Query lidar unit's model
   * @param buffer     Buffer to store the model.
   * @param buffer_len Length of buffer, the recommended buffer_len is
   *        32, the size of buffer needs to be >= buffer_len
   * @return 0 cannot get model
   *        -1 invlid lidar handle
   *        -2 buffer_len is too small
   *        -3 source is file
   *        otherwise return the size of model string
   *        stored in buffer, not include the trailing zero
   */
  virtual int get_model(char *buffer, int buffer_len) = 0;

  /**
   * @brief Start lidar
   */
  virtual int start() = 0;

  /**
   * @brief Stop lidar
   */
  virtual void stop() = 0;

  /**
   * @brief Print pipelie stage stats
   */
  virtual void print_stats() = 0;

  /**
   * @brief Get ring id converter
   * @return Address of RingIdConverterInterface
   */
  virtual RingIdConverterInterface *get_ring_id_converter() = 0;

  /**
   * @brief Get Lidar ID
   * @return Return Lidar ID in string
   */
  virtual const char *get_name(void) const {
    return name_;
  }

 public:
  /**
   * @brief Set the callback functions for message, data packet and status packet
   * @param message_callback Message callback function
   * @param data_callback    Data packet callback function
   * @param status_callback  Status packet callback function
   * @param get_host_time    Get host time callback function
   * @param callback_context Context that will be passed to callback functions
   */
  void set_callbacks(InnoMessageCallback message_callback, InnoDataPacketCallback data_callback,
                     InnoStatusPacketCallback status_callback, InnoHosttimeCallback get_host_time,
                     void *callback_context) {
    message_callback_external_ = message_callback;
    data_packet_callback_ = data_callback;
    status_packet_callback_ = status_callback;
    if (get_host_time) {
      get_host_time_ = get_host_time;
    }
    callback_context_ = callback_context;
    return;
  }

  /**
   * @brief Set the callback function for recorder
   * @param type     Callback type
   * @param callback Callback function
   * @param ctx      Context that will be passed to callback function
   * @return Return 0 means success, -1 means callback type is invalid
   */
  int set_recorder_callback(enum InnoRecorderCallbackType type, InnoRecorderCallback callback, void *ctx) {
    inno_log_panic_if_not(type < INNO_RECORDER_CALLBACK_TYPE_MAX && type > INNO_RECORDER_CALLBACK_TYPE_NONE,
                          "invalid data callback type %d", type);

    std::unique_lock<std::mutex> lk(recorder_callbacks_mutex_[type]);
    if (recorder_callbacks_[type]) {
      // already has callback
      return -1;
    }
    recorder_callbacks_[type] = callback;
    recorder_callbacks_context_[type] = ctx;
    return 0;
  }

  /**
   * [CST Bugfinder Defect ID 53728] Reviewed
   *
   * PolySpace report a defect here:
   * Unnecessary code, if-condition is always true.
   *
   * Avoid multithreaded competition consumption ,
   *
   * So ignore this defect
   */
  /**
   * @brief Check if recorder callback is set
   * @param type Callback type
   * @return Return true if callback is set, otherwise return false
   */
  inline bool has_recorder_callback(enum InnoRecorderCallbackType type) {
    if (recorder_callbacks_[type]) {
      std::unique_lock<std::mutex> lk(recorder_callbacks_mutex_[type]);
      if (recorder_callbacks_[type]) {
        return true;
      }
    }
    return false;
  }

  /**
   * [CST Bugfinder Defect ID 53727] Reviewed
   *
   * PolySpace report a defect here:
   * Unnecessary code, if-condition is always true.
   *
   * Avoid multi-threaded competition consumption ,
   *
   * So ignore this defect
   */
  /**
   * @brief Do recorder callback
   * @param type   Callback type
   * @param buffer Buffer to deliver to callback function
   * @param len    Buffer length
   * @return Return 0 means success, otherwise means failure
   */
  inline int do_recorder_callback(enum InnoRecorderCallbackType type, const char *buffer, int len) {
    if (recorder_callbacks_[type]) {
      InnoRecorderCallback callback = NULL;
      {
        std::unique_lock<std::mutex> lk(recorder_callbacks_mutex_[type]);
        callback = recorder_callbacks_[type];
      }
      if (callback) {
        // do not make the callback while hold the lock
        int ret = callback(handle_, recorder_callbacks_context_[type], type, buffer, len);
        if (ret != 0) {
          inno_log_info("recorder_callback[%d] return %d, cancel callback", type, ret);
          // this is the only place the reset the callbck
          std::unique_lock<std::mutex> lk(recorder_callbacks_mutex_[type]);
          recorder_callbacks_[type] = NULL;
          recorder_callbacks_context_[type] = NULL;
        }
        return ret;
      }
    }
    return 0;
  }

  /**
   * @brief Do message callback
   * @param error_level Message log level
   * @param code        Message code
   * @param fmt         Message format string
   * @param ...         Message format string arguments
   */
  void do_message_callback_fmt(enum InnoMessageLevel error_level, enum InnoMessageCode code, const char *fmt, ...) {
    va_list valist;
    va_start(valist, fmt);
    do_message_callback_v_(error_level, code, fmt, valist);
    va_end(valist);
  }

  /**
   * @brief Do data packet callback
   * @param data Data packet
   * @return Return 0 means success, otherwise means failure
   */
  int do_data_callback(const InnoDataPacket *data) {
    if (data_packet_callback_) {
      ++stat_point_data_packet_sent_;
      stat_point_sent_ += data->item_number;
      return data_packet_callback_(handle_, callback_context_, data);
    } else {
      return 0;
    }
  }

  /**
   * @brief Do status packet callback
   * @param error_level Message log level
   * @param code        Message code
   * @param error_message Message
   */
  void do_message_callback(enum InnoMessageLevel error_level, enum InnoMessageCode code, const char *error_message) {
    if (code != INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT) {
      inno_log_with_level(InnoLogLevel(error_level),
                          "message_callback: name=%s level=%d, "
                          "code=%d, message=%s",
                          name_, error_level, code, error_message);
    }
    if (message_callback_external_) {
      ++stat_message_packet_sent_;
      message_callback_external_(handle_, callback_context_,
                                 0,  // local message
                                 error_level, code, error_message);
    }
  }

  /**
   * @brief Do status packet callback
   * @param pkt Status packet
   */
  void do_status_callback(const InnoStatusPacket *pkt) {
    if (status_packet_callback_) {
      status_packet_callback_(handle_, callback_context_, pkt);
      stats_update_packet_bytes(ResourceStats::PACKET_TYPE_STATUS, 1, sizeof(InnoStatusPacket));
    }
  }

  /**
   * @brief Summary packets and bytes count for each packet type
   * @param type    PacketType
   * @param packet  Packet count
   * @param byte    Bytes count
   */
  virtual void stats_update_packet_bytes(enum ResourceStats::PacketType type, size_t packet, size_t byte) = 0;

  /**
   * @brief
   * @param cause
   */
  virtual void set_save_raw_data_flag(std::string cause) {
    // do nothing in LidarBase
    return;
  }

  /**
   * @brief Get read stage state
   * @return Return read stage state
   */
  virtual State get_state_() = 0;

 protected:
  /**
   * @brief Do message callback
   * @param error_level Message log level
   * @param code        Message code
   * @param fmt         Message format string
   * @param valist      Message format string arguments
   */
  void do_message_callback_v_(enum InnoMessageLevel error_level, enum InnoMessageCode code, const char *fmt,
                              va_list valist) {
    static const size_t kMaxMessageSize = 1024;
    char buffer[kMaxMessageSize];
    buffer[0] = 0;
    vsnprintf(buffer, kMaxMessageSize, fmt, valist);
    buffer[kMaxMessageSize - 1] = 0;
    do_message_callback(error_level, code, buffer);
  }

  /**
   * @brief Set play rate
   * @param rate Play rate
   */
  void set_play_rate_(int rate) {
    if (rate > 100) {
      play_rate_ = 0;
      play_rate_x_ = rate / 10000.0;
    } else if (rate >= 0) {
      play_rate_ = rate;
      play_rate_x_ = 0;
    } else {
    }
  }

 public:
  static std::mutex static_mutex_s;
  static std::map<int, InnoLidarBase *> lidars_s;
  static int max_handle_s;
  static uint32_t open_count_s;

 protected:
  int handle_;
  /* lidar ID */
  char *name_;

  int play_rate_;
  double play_rate_x_;

  /* callbacks*/
  InnoMessageCallback message_callback_external_;
  InnoDataPacketCallback data_packet_callback_;
  InnoStatusPacketCallback status_packet_callback_;
  InnoHosttimeCallback get_host_time_;
  InnoRecorderCallback recorder_callbacks_[INNO_RECORDER_CALLBACK_TYPE_MAX];
  void *recorder_callbacks_context_[INNO_RECORDER_CALLBACK_TYPE_MAX];
  std::mutex recorder_callbacks_mutex_[INNO_RECORDER_CALLBACK_TYPE_MAX];
  void *callback_context_;

  /* cpuset */
  size_t cpusetsize_;
  cpu_set_t *cpuset_ = NULL;
  bool exclude_callback_thread_;

  /* lidar basic configs */
  ConfigManager config_manage_;

  std::mutex last_stage_mutex_;
  bool last_stage_is_up_;

  /* stats */
  uint64_t stat_point_sent_{0};
  uint64_t stat_point_data_packet_sent_{0};
  uint64_t stat_message_packet_sent_{0};
};

}  // namespace innovusion
#endif  // SDK_COMMON_LIDAR_BASE_H_
