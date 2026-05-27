/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_COMMON_COMMUNICATION_H_
#define SDK_COMMON_COMMUNICATION_H_

#include <mutex>

#include "sdk_common/inno_lidar_packet.h"
#include "utils/inno_lidar_log.h"
#include "utils/net_manager.h"

namespace innovusion {

/**
 * @brief LidarCommunication
 */
class LidarCommunication : public NetManager {
 public:
  class StatusCounters {
   public:
    StatusCounters() {
      memset(this, 0, sizeof(*this));
    }
    ~StatusCounters();
    int set(const char *input);

    enum InnoTimeSyncConfig clock_config;
    uint64_t uptime;
    uint32_t stream_count;
    enum InnoLidarState stream_status;
    size_t data_sent;
    int32_t error_code;
    uint32_t idle_loop;
    uint32_t lose_counter_1;
    uint32_t lose_counter_2;
  };

 public:
  static const uint32_t kCapabilitiesRegister = 0x1104;
  static const uint32_t kDmaControlRegister = 0x1004;
  static const uint32_t kDmaRawDataSelected = 0x2;  // bit 1
  static const uint32_t kExtendedPacketFormat = 0x20000;
  static const int kNOPRoi = 10000;
  static constexpr double kDefaultTimeoutSec = 0.5;

 public:
  /**
   * @brief LidarCommunication constructor
   * @param ip           Lidar IP
   * @param port         Lidar port
   * @param service_port Lidar service port
   * @param timeout_sec  Connection timeout in seconds
   */
  LidarCommunication(const char *ip, uint16_t port, uint16_t service_port, double timeout_sec);
  ~LidarCommunication();

  /**
   * @brief Get section key value from Lidar
   * @param section        Section name
   * @param key            Key name
   * @param default_value  Default value
   * @return Return value get from Lidar, or default value if failed
   */
  double get_config_section_key_value(const char *section, const char *key, double default_value);

  /**
   * @brief Set section key value to Lidar
   * @param section Section name
   * @param key     Key name
   * @param value   Address of value to be set
   * @return Return 0 if success, others for failed
   */
  int set_config_section_key_value(const char *section, const char *key, const char *value);

  /**
   * @brief Read 32bits register from Lidar
   * @param addr  Register address
   * @param data  Buffer to store data
   * @return Return 0 if success, others for failed
   */
  int reg_read(uint32_t addr, uint32_t *data);

  /**
   * @brief Write 32bits register to Lidar
   * @param addr  Register address
   * @param data  Data to be written
   * @return Return 0 if success, others for failed
   */
  int reg_write(uint32_t addr, uint32_t data);

  inline int read_ps_reg(uint32_t addr, uint32_t *data) {
    uint32_t ps_addr = 0x1000 | addr;
    return reg_read(ps_addr, data);
  }

  inline int write_ps_reg(uint32_t addr, uint32_t data) {
    uint32_t ps_addr = 0x1000 | addr;
    return reg_write(ps_addr, data);
  }
  /**
   * @brief Get frame rate from Lidar
   * @return Return positive frame rate if success, others for failed
   */
  double get_frame_rate();

  /**
   * @brief Get galvo mode from Lidar
   * @return Return galvo
   */
  int get_galvo_mode();

  /**
   * @brief Get board name from Lidar
   * @param buffer     Buffer to store board name
   * @param buffer_len Buffer length
   * @return Return 0 if success, others for failed
   */
  int get_board_name(char *buffer, size_t buffer_len);

  /**
   * @brief Get DID from Lidar
   * @param buffer
   * @param buffer_len
   * @param did
   * @return
   */
  int get_did(char *buffer, size_t buffer_len, const char *did);

  /**
   * @brief Get FW squenece from Lidar
   * @param buffer     Buffer to store FW sequence
   * @param buffer_len Buffer length
   * @return Return 0 if success, others for failed
   */
  int get_fw_sequence(char *buffer, size_t buffer_len);

  /**
   * @brief Get FW version from Lidar
   * @param buffer     Buffer to store FW version
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_fw_version(char *buffer, size_t buffer_len);

  /**
   * @brief Get SN from Lidar
   * @param buffer      Buffer to store Lidar SN
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_sn(char *buffer, size_t buffer_len);
  int get_HW_config(int* hw_config);
  /**
   * @brief Get model from Lidar
   * @param buffer      Buffer to store model
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_model(char *buffer, size_t buffer_len);

  /**
   * @brief Get temperature from Lidar
   * @param buffer      Buffer to store Lidar temperature
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_temperature(char *buffer, size_t buffer_len);

  /**
   * @brief Get detector temperature from Lidar
   * @param buffer      Buffer to store detector temperature
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_detector_temps(char *buffer, size_t buffer_len);

  /**
   * @brief Set Inno Fault to Lidar
   * @param fid InnoLidarInFault
   * @return Return 0 for success, others for failed
   */
  int set_fw_inner_fault_do(enum InnoLidarInFault fid);

  /**
   * @brief Get Lidar motor speeds
   * @param buffer     Buffer to store motor speeds
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_motor_speeds(char *buffer, size_t buffer_len);

  /**
   * @brief Get motor config from Lidar
   * @param buffer      Buffer to store config
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_internal_motor_config(char *buffer, size_t buffer_len);

  /**
   * @brief Get galvo first period from Lidar
   * @param buffer     Buffer to store galvo first period
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_galvo_first_period(char *buffer, size_t buffer_len);

  /**
   * @brief Get quiet mode config from Lidar
   * @param buffer     Buffer to store quiet mode config
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_internal_quiet_config(char *buffer, size_t buffer_len);
  int get_saved_roi_enable(char *buffer, size_t buffer_len);
  /**
   * @brief Get calibration yaml from Lidar
   * @param buffer     Buffer to store calibration yaml
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for failed
   */
  int get_geo_yaml(char *buffer, size_t buffer_len);

  /**
   * @brief Get if external packet format from Lidar
   * @param format Buffer to store if external packet format
   * @return Return 0 for success, others for failed
   */
  int get_ext_packet_format(bool *format);

  /**
   * @brief Populate StatusCounters with Lidar response
   * @param counters  Buffer of StatusCounters to be populated
   * @param get_diff  True to update last lose counter, false to not update
   * @return Return 0 for success, others for failed
   */
  int get_status(StatusCounters *counters, bool get_diff);

  /**
   * @brief Get Lidar mode and status.
   *        This implementation is invalid yet.
   * @param mode      Buffer to store Lidar mode
   * @param pre_mode  Buffer to store Lidar previous mode
   * @param status    Buffer to store Lidar status
   * @return Return 0 for success, others for failed
   */
  int get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode, enum InnoLidarStatus *status);

  /**
   * @brief Get APD calibration status from Lidar
   * @param status Buffer to store APD calibration status
   * @return Return 0 for success, others for failed
   */
  int get_apd_cal_status(int *status);

  /**
   * @brief Get calibration mode id from Lidar
   * @param calibration_id Buffer to store calibration mode id
   * @return Return 0 for success, others for failed
   */
  int get_calibration_mode_id(int *calibration_id);

  /**
   * @brief Get Lidar mode in string
   * @param mode Lidar mode
   * @return Return string of Lidar mode
   */
  const char *get_mode_section(InnoLidarMode mode);
  const char *get_current_mode_section();
  /**
   * @brief Clear Fault
   * @param fid Fault ID to be cleared
   * @return Return 0 for success, others for failed
   */
  int clear_fw_inner_faults(uint32_t fid);
  double get_motor_speed_config();
  double get_ptp_time_stamping();

  // FALCON-I_UT04_SRC_SDK_013
  // int set_raw_capture_mode(bool);

  /**
   * @brief Set use raw mode to Lidar.
   *        If use raw mode, will raise a panic in this implementation.
   * @param use_raw_mode 0: disable raw mode, 1: enable raw mode
   * @return Return 0 for success, others for failed
   */
  int set_raw_capture_mode(bool use_raw_mode);

  /**
   * @brief Set vertical ROI centor to Lidar
   * @param angle Verical ROI centor
   * @return Return 0 for success, others for failed
   */
  int set_vertical_roi(double angle);

  /**
   * @brief Get vertical ROI centor from Lidar
   * @param angle Buffer to store verical ROI centor
   * @return Return 0 for success, others for failed
   */
  int get_vertical_roi(double *angle);

  /**
   * @brief Set reboot flag to Lidar
   * @param value Reboot flag
   * @return Return 0 for success, others for failed
   */
  int set_reboot(int value);

  /**
   * @brief Connect to Lidar and get streaming fd
   * @return Return 0 for success, others for failed
   */
  int open_streaming(void);

  /**
   * @brief Start streaming
   * @param is_tcp Is tcp
   * @return Return 0 for success, others for failed
   */
  int start_streaming(bool is_tcp);

  /**
   * @brief Close streaming fd
   */
  void close_streaming();

  /**
   * @brief Get streaming fd
   * @return Return streaming fd
   */
  int get_streaming_fd();

  /**
   * @brief  Send stop command to Lidar
   * @return Return 0 for success, others for failed
   */
  int send_stop();

  /**
   * @brief Set Lidar mode and get Lidar previous mode and status
   * @param mode     Lidar mode to set
   * @param pre_mode Buffer to store previous mode
   * @param status   Buffer to store Lidar status
   * @return Return 0 for success, others for failed
   */
  int set_mode(enum InnoLidarMode mode, enum InnoLidarMode *pre_mode, enum InnoLidarStatus *status);

 private:
  /**
   * @brief Send command to Lidar and get response
   * @param buffer     Buffer to store response
   * @param buffer_len Buffer length
   * @param cmd        Buffer send to Lidar
   * @return Return 0 for success, others for failed
   */
  int send_command_and_save_reply_(char *buffer, size_t buffer_len, const char *cmd);

  /**
   * @brief Send command to Lidar and get response
   * @param recv_buffer     Buffer to store response
   * @param recv_buffer_len Buffer length
   * @param send_buffer     Buffer send to Lidar
   * @return Return 0 for success, others for failed
   */
  int get_reply_info_(char *recv_buffer, size_t recv_buffer_len, const char *send_buffer);

 private:
  uint16_t service_port_;
  uint32_t first_ctrlor_lose_1_;
  uint32_t first_ctrlor_lose_2_;
  uint32_t last_ctrlor_lose_1_;
  uint32_t last_ctrlor_lose_2_;
  size_t get_status_called_cnt_;
  std::mutex mutex_;
  int streaming_fd_;
};
}  // namespace innovusion
#endif  // SDK_COMMON_COMMUNICATION_H_
