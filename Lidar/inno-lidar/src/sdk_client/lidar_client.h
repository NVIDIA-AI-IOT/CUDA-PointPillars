/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_LIDAR_CLIENT_H_
#define SDK_CLIENT_LIDAR_CLIENT_H_

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <sys/types.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <string>

#include "sdk_common/inno_lidar_api.h"
#include "sdk_client/raw_recorder.h"
#include "sdk_client/lidar_fault_check.h"
#include "sdk_common/lidar_base.h"
#include "sdk_common/resource_stats.h"
#include "utils/config.h"
#include "utils/inno_thread.h"
#include "utils/log.h"

namespace innovusion {
class ClientStats;
class ConsumerProducer;
class LidarClientCommunication;
class MemPool;
class StageClientRead;
class StageClientDeliver;
class StageClientDeliver2;
class StageClientRecorder;
class RingIdConverter;
class RawReceiver;

/**
 * @brief InnoLidarClient
 */
class InnoLidarClient : public InnoLidarBase {
  friend class StageClientRead;
  friend class StageClientDeliver;
  friend class StageClientDeliver2;
  friend class StageClientRecorder;
  friend class RawReceiver;
  friend class ClientStats;
  friend class BaseInput;
  friend class FileInput;
  friend class UdpInput;
  friend class TcpInput;
  friend class PcapInput;

 private:
  enum LidarSource {
    LIDAR_SOURCE_NONE = 0,
    LIDAR_SOURCE_FILE,
    LIDAR_SOURCE_LIVE,
    LIDAR_SOURCE_MAX,
  };

 public:
  // xxx todo: pick the right value
  static const size_t kSignalJobPoolSize = 15;
  static const size_t kAngleJobPoolSize = 20;
  static const size_t kDeliverMessageMaxSize = 60000;
  static const size_t kDeliverPointsMaxBlockNumber = 10000;
  static const size_t kDeliverMessageJobPoolSize = 10;
  static const size_t kDeliverPointsJobPoolSize = 10;
  static const size_t kDeliverStatusJobPoolSize = 10;
  static const size_t kMaxPacketSize = 65536;
  static const size_t kPacketPoolSize = 900;

 public:  // static methods
  /**
   * @brief Ignored
   * @param job     Ignored
   * @param ctx     Ignored
   * @param prefer  Ignored
   * @return Ignored
   */
  static int reader_func(void *job, void *ctx, bool prefer);

 public:
  /**
   * @brief InnoLidarClient constructor, read data from network
   * @param name      Lidar ID
   * @param lidar_ip  Lidar IP string, e.g. "172.168.1.10".
   * @param port      Lidar PORT, e.g. 8001.
   * @param use_tcp   Specify source type, true for TCP, false for UDP
   * @param udp_port  Specify UDP port to receive data
   */
  InnoLidarClient(const char *name, const char *lidar_ip, uint16_t port, bool use_tcp, uint16_t udp_port);

  /**
   * @brief InnoLidarClient constructor, read data from file
   * @param name      Lidar ID
   * @param filename  Reading data from the file
   * @param play_rate Play rate
   * @param rewind    Rewind
   * @param skip      Ignored
   */
  InnoLidarClient(const char *name, const char *filename, int play_rate, int rewind, int64_t skip);

  /**
   * @brief InnoLidarClient constructor, read data from file
   * @param name        Lidar ID
   * @param filename    Reading data from the file
   * @param play_rate   Play rate
   * @param rewind      Rewind
   * @param file_open   File open function
   * @param file_read   File read function
   * @param file_close  File close function
   * @param ctx         File operation functions context
   */
  InnoLidarClient(const char *name, void *ctx);
  ~InnoLidarClient();

 public:
  /**
   * @brief Read ps register, but do nothing in this implement
   * @param off   The data offset
   * @param value The data container
   * @return Return 1 always
   */
  int read_ps_reg(uint16_t off, uint32_t *value) override {
    return 1;
  }
  /**
   * @brief Read pl register, but do nothing in this implement
   * @param off   The data offset
   * @param value The data container
   * @return Return 1 always
   */
  int read_pl_reg(uint16_t off, uint32_t *value) override {
    return 1;
  }
  /**
   * @brief Write ps register, but do nothing in this implement
   * @param off   The data offset
   * @param value The data container
   * @return Return 1 always
   */
  int write_ps_reg(uint16_t off, uint32_t value) override {
    return 1;
  }
  /**
   * @brief Write pl register, but do nothing in this implement
   * @param off   The data offset
   * @param value The data container
   * @return Return 1 always
   */
  int write_pl_reg(uint16_t off, uint32_t value) override {
    return 1;
  }

  /**
   * @brief Load params from yaml file, but do nothing int this implement
   * @param lidar_model   Lidar model
   * @param yaml_filename The params file Lidar read from
   * @return Return 0 always
   */
  int set_params_file(const char *lidar_model, const char *yaml_filename) override {
    return 0;
  }
  /**
   * @brief Set key & value to config
   * @param name  Key
   * @param value Value
   * @return Return 0 for success, others for error
   */
  int set_config_name_value(const char *name, const char *value) override;
  /**
   * @brief Set InnoReflectanceMode to Lidar server
   * @param mode InnoReflectanceMode
   * @return Return 0 for success, others for error
   */
  int set_reflectance_mode(enum InnoReflectanceMode) override;

  /**
   * @brief Set InnoMultipleReturnMode to Lidar server
   * @param ret_mode InnoMultipleReturnMode
   * @return Return 0 for success, others for error
   */
  int set_return_mode(enum InnoMultipleReturnMode ret_mode) override;

  /**
   * @brief Set ROI to Lidar server
   * @param h_angle ROI horizontal center (in [-60, 60] degrees)
   * @param v_angle ROI vertical center (in [-25, 25] degrees)
   *        Setting either horz_angle or vert_angle to
   *        kInnoNopROI, i.e. 10000.0
   *        will maintain the previous value for that angle.  Any other value
   *        outside the allowed range will result in a failure code return and
   *        no change in ROI center.
   * @return Return 0 for success, others for error
   */
  int set_roi(double h_angle, double v_angle) override;

  /**
   * @brief Get ROI from Lidar server
   * @param h_angle ROI horizon angle center
   * @param v_angle ROI vertical angle center
   * @return Return 0 for success, others for error
   */
  int get_roi(double *h_angle, double *v_angle) override;

  /**
   * @brief Set mode to Lidar server and get the mode & status before set
   * @param mode                 The mode to be set
   * @param mode_before_change   Current mode before mode set
   * @param status_before_change Current status before mode set
   * @return Return 0 for success, others for error
   */
  int set_mode(enum InnoLidarMode mode, enum InnoLidarMode *mode_before_change,
               enum InnoLidarStatus *status_before_change) override;
  /**
   * @brief Get current mode & previous mode & current status from Lidar server
   * @param mode                  Address to store current mode
   * @param pre_mode              Address to store previous mode
   * @param status                Address to store current status
   * @param in_transition_mode_ms Time (in ms) stay in transition mode
   * @return Return 0 for success, others for error
   */
  int get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode, enum InnoLidarStatus *status,
                      uint64_t *in_transition_mode_ms) override;

  /**
   * @brief Get anglehv table from Lidar server
   * @param anglehv_table     Buffer to store table
   * @return Return 0 for success, others for error
   */
  int get_anglehv_table(InnoDataPacket *anglehv_table) override;

  /**
   * @brief Get attribute from Lidar server
   * @param attribute Name of the attribute
   * @param value     Buffer to store attribute value
   * @return Return 0 for success, others for error
   */
  int get_attribute(const char *attribute, double *value) override;

  /**
   * @brief Get attribute from Lidar server
   * @param attribute Name of the attribute
   * @param buf       Buffer to store attribute string value
   * @param buf_size  Buffer size
   * @return Return 0 for success, others for error
   */
  int get_attribute_string(const char *attribute, char *buf, size_t buf_size) override;

  /**
   * @brief Set lidar attribute
   * @param attribute Name of the attribute
   * @param buffer    Buffer to store the string value
   * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
   */
  int set_attribute_string(const char *attribute, const char *buf) override;

  /**
   * @brief Set fault value to Lidar server
   * @param value_hex_str fault value
   * @return Return 0 for success, others for error
   */
  int set_faults_save_raw(const std::string& value_hex_str) override;

  /**
   * @brief Set the velocity and angular velocity for motion correction
   * @param velocity_m_s[3]:           Velocities in x, y, z axis, unit is m/s
   * @param angular_velocity_rad_s[3]: Angular velocities in x, y, z axis, RAD/s
   * @return 0 means success, -1 if handle is invalid
   */
  int set_motion_compensation(double velocity[3], double angular_velocity[3]) override;

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
  int thread_setaffinity_np(size_t cpusetsize, const cpu_set_t *cpuset, int exclude_callback_thread) override;

  /**
   * @brief Query lidar unit's state
   * @param state       Address to store InnoLidarState
   * @param error_code  Address to store error code. NULL means
   *        do not store error code
   * @return 0 means success, -1 if handle is invalid
   */
  int get_fw_state(enum InnoLidarState *state, int *error_code) override;

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
  int get_fw_version(char *buffer, int buffer_len) override;

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
  int get_sn(char *buffer, int buffer_len) override;

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
  int get_model(char *buffer, int buffer_len) override;

  /**
   * @brief Start lidar
   */
  int start() override;

  /**
   * @brief Stop lidar
   */
  void stop() override;

  /**
   * @brief Print pipelie stage stats
   */
  void print_stats(void) override;

  /**
   * @brief Test some lidar command, such as fw_version/get_sn and etc.
   * @Return 0 for success, others for error
   */
  int before_read_start(void);

  /**
   * @brief Summary packets and bytes count for each packet type
   * @param type    PacketType
   * @param packet  Packet count
   * @param byte    Bytes count
   */
  void stats_update_packet_bytes(enum ResourceStats::PacketType type, size_t packet, size_t byte) override;

  /**
   * @brief Check galvo status
   * @param pkt          Packet to be checked
   * @param check_result A buffer to store check result
   * @return InnoFrameCheckProcess
   */
  InnoFrameCheckProcess get_check_galvo_mirror_result(const InnoDataPacket &pkt, InnoGalvoCheckResult *check_result);

  /**
   * @brief Get max distance check result in client
   * @param buf          Buffer to be checked
   * @param check_result Buffer to store check result
   * @return  0: checking
   *          1: check completed
   *          otherwise: error
   */
  int get_check_max_distance_result(const char *buf, InnoMaxDistanceResult *check_result);

  /**
   * @brief Check whether the points on a tunnel
   * @param  pkt InnoDataPacket type should be INNO_ITEM_TYPE_XYZ_POINTCLOUD
   * @Return 0 for success, others for error
   */
  int check_max_distance_on_tunnel(const InnoDataPacket &pkt);

  /**
   * @brief Set galvo sensor fault, it' temporarily closed
   * @Return 0 for success, others for error
   */
  int set_galvo_sensor_fault();

  /**
   * @brief Clear galvo sensor fault, it' temporarily closed
   * @Return 0 for success, others for error
   */
  int clear_galvo_sensor_fault();

  /**
   * @brief Set galvo mirror fault
   * @Return 0 for success, others for error
   */
  int set_galvo_mirror_fault();

  /**
   * @brief Clear galvo mirror fault
   * @Return 0 for success, others for error
   */
  int clear_galvo_mirror_fault();

  /**
   * @brief Set max distance fault
   * @Return 0 for success, others for error
   */
  int set_max_distance_fault();

  /**
   * @brief Clear max distance fault
   * @Return 0 for success, others for error
   */
  int clear_max_distance_fault();

  /**
   * @brief Set misalignment time in minutes
   * @param minutes misalgnment time
   * @Return 0 for success, others for error
   */
  int set_misalignment_time(int minutes);

  /**
   * @brief Clear misalignment time
   * @Return 0 for success, others for error
   */
  int clear_misalignment_time();

  /**
   * @brief Update ring id table
   */
  void update_ring_id_table(InnoDataPacket *pkt);

  /**
   * @brief Get ring id converter
   * @return Address of RingIdConverterInterface
   */
  RingIdConverterInterface *get_ring_id_converter() override;

  /**
   * @brief Get lidar ip
   * @return IP in in_addr_t
   */
  uint32_t get_specified_ip();

 protected:
  friend class LidarClientCommunication;

  /**
   * @brief Add config
   * @param c Address of config to be add
   */
  void add_config(Config *c);

  /**
   * @brief Remove config
   * @param c Address of config to be removed
   */
  void remove_config(Config *c);

 private:
  /**
   * @brief Init InnoLidarClient
   */
  void init_();

  /**
   * @brief Get read stage state
   * @return Read stage state
   */
  InnoLidarBase::State get_state_() override;

  /**
   * @brief Check if used as lidar
   * @return True for lidar, false for others
   */
  bool is_live_lidar_() const;

  /**
   * @brief Alloc buffer from memory pool with size of kMaxPacketSize bytes
   * @param size Verify required size is not larger than kMaxPacketSize
   * @return Buffer Address
   */
  void *alloc_buffer_(size_t size);

  /**
   * @brief Free buffer to memory pool
   * @param buffer Buffer to be free
   */
  void free_buffer_(void *buffer);

  /**
   * @brief Send job to deliver stage
   * @param job Address of inno packet
   */
  void add_deliver_job_(void *);

  /**
   * @brief Send job to deliver2 stage
   * @param job Address of inno packet
   */
  void add_deliver2_job_(void *);

  void add_recorder_job_(void *);
  /**
   * @brief Check if packet received from target lidar
   * @param specified_ip lidar ip
   * @param packet_ip    packet ip
   * @return True for packet received from target lidar, false for not
   */
  bool source_ip_check_(uint32_t specified_ip, uint32_t packet_ip);

  int attribute_force_xyz_pointcloud_(const char *buf);
  int attribute_force_vehicle_coordinate_(const char *buf);
  int attribute_save_raw_data_(const char *buf);
  int attribute_raw_data_save_path_(const char *buf);
  int attribute_faults_save_raw_(const char *buf);
  int attribute_steering_wheel_angle_(const char *buf);
  int attribute_vehicle_speed_(const char *buf);
  int attribute_galvo_ext_ref_(const char *buf);
  int attribute_use_ring_id_(const char *buf);

 private:
  /* source info */
  enum LidarSource lidar_source_;
  char *ip_;
  uint16_t port_;
  char *filename_;
  InnoDataPacket *anglehv_table_;
  bool anglehv_init_{false};
  /* communication with lidar */
  LidarClientCommunication *comm_;

  MemPool *packet_pool_;
  ConsumerProducer *cp_read_;
  ConsumerProducer *cp_deliver_;
  ConsumerProducer *cp_deliver2_;
  ConsumerProducer *cp_recorder_;
  StageClientRead *stage_read_;
  StageClientDeliver *stage_deliver_;
  StageClientDeliver2 *stage_deliver2_;
  StageClientRecorder *stage_recorder_;
  bool force_xyz_pointcloud_;

  ClientStats *client_stats_;

  // for time sync packet, client should buf ip port info
  // and set to server if server restart
  std::string time_sync_set_value{""};

  // The ptr of galvo mirror check
  InnoGalvoMirrorCheck *galvo_mirror_check_;
  // The ptr of max distance check
  InnoMaxDistanceCheck *max_distance_check_;

  // use ring id instead of <scan_line, ch> to identify each line
  RingIdConverter *ring_id_converter_{NULL};
  std::mutex ring_id_set_mutex_;
  // save raw4 data
  RawReceiver *raw_recorder_ = nullptr;
  InnoThread *it_raw_recorder_ = nullptr;
  std::string raw_recoder_save_path_;
  std::string faults_save_raw_ = {""};
};

}  // namespace innovusion
#endif  // SDK_CLIENT_LIDAR_CLIENT_H_
