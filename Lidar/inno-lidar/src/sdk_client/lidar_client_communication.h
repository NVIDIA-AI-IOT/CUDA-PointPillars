/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_LIDAR_CLIENT_COMMUNICATION_H_
#define SDK_CLIENT_LIDAR_CLIENT_COMMUNICATION_H_

#include <string>

#include "sdk_common/inno_lidar_api.h"
#include "utils/config.h"
#include "utils/net_manager.h"

namespace innovusion {

class InnoLidarClient;

/**
 * @brief LidarClientCommunicationConfig
 */
class LidarClientCommunicationConfig : public Config {
 public:
  LidarClientCommunicationConfig() : Config() {
  }

  /**
   * @brief   Get config name
   * @return  Return config name
   */
  const char *get_type() const override {
    return "LidarClient_Communication";
  }

  /**
   * @brief Update configurations
   * @param key   Config key
   * @param value Config value
   * @return Ignored
   */
  int set_key_value_(const std::string &key, double value) override {
    SET_CFG(get_conn_timeout_sec);
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
  double get_conn_timeout_sec = 0.5;
  END_CFG_MEMBER()
};

/**
 * @brief LidarClientCommunication
 */
class LidarClientCommunication : public NetManager {
 public:
  static const int kInvalidVerticalRoi = 1000000;
  static const size_t kSmallBufferSize = 1024 * 8;
  static const size_t kMaxUrlSize = 1024;
  static const size_t kMaxReplySize = 4096 * 2;

 public:
  /**
   * @brief LidarClientCommunication constructor
   * @param client      Address of InnoLidarClient
   * @param ip          Lidar ip
   * @param port        Lidar port
   * @param timeout_sec Connection timeout
   */
  LidarClientCommunication(InnoLidarClient *client, const char *ip, unsigned short port, double timeout_sec);

  /**
   * @brief LidarClientCommunication constructor
   * @param ip          Lidar ip
   * @param port        Lidar port
   * @param timeout_sec Connection timeout
   */
  LidarClientCommunication(const char *ip, unsigned short port, double timeout_sec);
  ~LidarClientCommunication();

 public:
  /**
   * @brief Get attribute from Lidar server
   * @param attr       Name of the attribute
   * @param buffer_in  Buffer to store attribute value
   * @param buffer_len Buffer length
   * @return Return 0 for success, others for error
   */
  int get_attribute(const char *attr, char *buffer_in, size_t buffer_len);

  /**
   * @brief Get attribute from Lidar server
   * @param attribute Name of the attribute
   * @param value     Buffer to store attribute value
   * @return Return 0 for success, others for error
   */
  int get_attribute(const char *attribute, double *value);

  /**
   * @brief Get fw state from Lidar server
   * @param buffer      Buffer to store fw state
   * @param buffer_len  Buffer length
   * @return  0 for success, others for error
   */
  int get_fw_state(char *buffer, size_t buffer_len);

  /**
   * @brief Get fw version from Lidar server
   * @param buffer     Buffer to store fw version
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
  int get_fw_version(char *buffer, size_t buffer_len);

  /**
   * @brief Get SN from Lidar server
   * @param buffer     Buffer to store SN
   * @param buffer_len Length of buffer, the recommended buffer_len is
   *        128, the size of buffer needs to be >= buffer_len.
   * @return 0 cannot get S/N
   *         -1 invlid lidar handle
   *         -2 buffer_len is too small
   *         -3 source is file
   *         otherwise return the size of S/N string
   *         stored in buffer, not include the trailing zero
   */
  int get_sn(char *buffer, size_t buffer_len);

  /**
   * @brief Get model from Lidar server
   * @param buffer     Buffer to store model
   * @param buffer_len Length of buffer, the recommended buffer_len is
   *        32, the size of buffer needs to be >= buffer_len
   * @return 0 cannot get model
   *        -1 invlid lidar handle
   *        -2 buffer_len is too small
   *        -3 source is file
   *        otherwise return the size of model string
   *        stored in buffer, not include the trailing zero
   */
  int get_model(char *buffer, size_t buffer_len);

  /**
   * @brief Get debug level from Lidar server
   * @param debug_level Buffer to store debug level
   * @return Return 0 for success, others for error
   */
  int get_debug(InnoLogLevel *debug_level);

  /**
   * @brief Get lidar current working mode and status
   * @param mode                  Current work mode
   * @param pre_mode              Previous work mode
   * @param status                Work status
   * @param in_transition_mode_ms Time (in ms) stay in transition mode
   * @return Return 0 for success, others for error
   */
  int get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode, enum InnoLidarStatus *status,
                      uint64_t *in_transition_mode_ms);

  /**
   * @brief
   * @param port_data     Buffer to store data port
   * @param port_message  Buffer to store message port
   * @param port_status   Buffer to store status port
   * @param ip            Buffer to store lidar ip
   * @param ip_len        Buffer length of ip
   * @param ip2           Buffer to store local ip
   * @param ip2_len       Buffer length of ip2
   * @return Return 0 for success, others for error
   */
  int get_server_udp_ports_ip(int32_t *port_data, int32_t *port_message, int32_t *port_status, char *ip, size_t ip_len,
                              char *ip2, size_t ip2_len);

  /**
   * @brief Set Lidar server UDP port
   * @param port_data UDP port to be set
   * @return Return 0 for success, others for error
   */
  int set_server_udp_ports_ip(uint16_t port_data);

  /**
   * @brief Set ROI to Lidar server
   * @param h_roi ROI horizontal center (in [-60, 60] degrees)
   * @param v_roi ROI vertical center (in [-25, 25] degrees)
   *        Setting either horz_angle or vert_angle to
   *        kInnoNopROI, i.e. 10000.0
   *        will maintain the previous value for that angle.  Any other value
   *        outside the allowed range will result in a failure code return and
   *        no change in ROI center.
   * @return Return 0 for success, others for error
   */
  int set_roi(double h_roi, double v_roi);

  /**
   * @brief Get ROI from Lidar server
   * @param h_roi ROI horizon angle center
   * @param v_roi ROI vertical angle center
   * @return Return 0 for success, others for error
   */
  int get_roi(double *h_roi, double *v_roi);

  /**
   * @brief Set InnoReflectanceMode to Lidar server
   * @param val InnoReflectanceMode
   * @return Return 0 for success, others for error
   */
  int set_reflectance_mode(InnoReflectanceMode val);

  /**
   * @brief Set InnoMultipleReturnMode to Lidar server
   * @param val InnoMultipleReturnMode
   * @return Return 0 for success, others for error
   */
  int set_return_mode(InnoMultipleReturnMode val);

  /**
   * @brief Set debug level to Lidar server
   * @param val Debug level
   * @return Return 0 for success, others for error
   */
  int set_debug(InnoLogLevel val);

  /**
   * @brief Set mode to Lidar server
   * @param val The mode to be set
   * @return Return 0 for success, others for error
   */
  int set_mode(enum InnoLidarMode val);

  /**
   * @brief Set reboot command to Lidar server
   * @param value Set 1 for reboot, others will be ignored
   * @return Return 0 for success, others for error
   */
  int set_reboot(int value);

  /**
   * @brief Get UDP ip and raw port id from Lidar server
   * @param port_raw Buffer to store port id
   * @param ip       Buffer to store ip
   * @param ip_len   IP buffer length
   * @return Return 0 for success, others for error
   */
  int get_server_udp_raw_port(int32_t *port_raw, char *ip, size_t ip_len);

  /**
   * @brief Set UDP raw port to Lidar server
   * @param port_raw port id
   * @return Return 0 for success, others for error
   */
  int set_server_udp_raw_port(uint16_t port_raw);

  /**
   * @brief Set fault value to Lidar server
   * @param value fault value
   * @return Return 0 for success, others for error
   */
  int set_faults_save_raw(const std::string &value_hex_str);
  int set_faults_save_raw(uint64_t value);

  /**
   * @brief Set attribute to Lidar server
   * @param attribute Key
   * @param buf       Value
   * @return Return 0 for success, others for error
   */
  int set_attribute_string(const char *attribute, const char *buf);

  /// @brief get anglehv_table from Lidar server
  int get_anglehv_table(char *buffer, size_t buffer_len);

 private:
  /**
   * @brief Send http get request to Lidar server to set data
   * @param set_command Buffer of set command
   * @return Return 0 for success, others for error
   */
  int send_set_command_(const char *set_command);

  /**
   * @brief Send http get request to Lidar server to get data
   * @param name        Get param
   * @param buffer      Buffer to store result
   * @param buffer_len  Buffer length
   * @return Return 0 for success, others for error
   */
  int send_get_command_(const char *name, char *buffer, size_t buffer_len);
  /**
   * @brief Send http get request to Lidar server to get data
   * @param name        Get param
   * @param buffer      Buffer to store result
   * @param buffer_len  Buffer length
   * @param value       Get param value
   * @return Return 0 for success, others for error
   */
  int send_get_command_(const char *name, char *buffer, size_t buffer_len, const char *value);

 private:
  LidarClientCommunicationConfig config_base_;
  LidarClientCommunicationConfig config_;
  InnoLidarClient *lidar_client_;
};

}  // namespace innovusion
#endif  // SDK_CLIENT_LIDAR_CLIENT_COMMUNICATION_H_
