/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include "sdk_common/lidar_communication.h"

#include "utils/log.h"
#include "utils/md5.h"

namespace innovusion {

LidarCommunication::StatusCounters::~StatusCounters() {
}

int LidarCommunication::StatusCounters::set(const char *input) {
  memset(this, 0, sizeof(*this));
  int ret = sscanf(input,
                   "time_config=%d up_time=%" PRI_SIZEX " err=0x%x stream_status=%d "
                   "stream_count=%u "
                   "data_sent=%" PRI_SIZEU " count1=%u count2=%u count3=%u",
                   reinterpret_cast<int*>(&clock_config),
                   &uptime,
                   &error_code,
                   reinterpret_cast<int*>(&stream_status),
                   &stream_count,
                   &data_sent,
                   &idle_loop,
                   &lose_counter_1,
                   &lose_counter_2);
  return ret == 9 ? 0 : -1;
}

LidarCommunication::LidarCommunication(const char *ip_addr, uint16_t port, uint16_t service_port, double timeout_sec)
    : NetManager(ip_addr, port, timeout_sec) {
  service_port_ = service_port;
  first_ctrlor_lose_1_ = 0;
  first_ctrlor_lose_2_ = 0;
  last_ctrlor_lose_1_ = 0;
  last_ctrlor_lose_2_ = 0;
  get_status_called_cnt_ = 0;
  streaming_fd_ = -1;
}

LidarCommunication::~LidarCommunication() {
}

int LidarCommunication::send_command_and_save_reply_(char *buffer, size_t buffer_len, const char *cmd) {
  char *recv_buffer = send_command_and_get_reply("%s", cmd);
  int ret;
  if (recv_buffer == NULL) {
    ret = 0;
  } else {
    size_t len = strlen(recv_buffer);
    if (buffer_len >= len + 1) {
      memcpy(buffer, recv_buffer, len + 1);
      ret = len;
    } else {
      ret = -2;
    }
    free(recv_buffer);
  }
  return ret;
}

double LidarCommunication::get_config_section_key_value(const char *section, const char *key, double default_value) {
  double ret = default_value;
  char *config_buffer = send_command_and_get_reply("%s %s %s", "get_i_config", section, key);
  if (config_buffer != NULL) {
    double value = 0;
    char pat[256];
    snprintf(pat, sizeof(pat), "[%s]\n%s = %%lf", section, key);
    pat[sizeof(pat) - 1] = 0;
    if (sscanf(config_buffer, pat, &value) == 1) {
      ret = value;
    } else {
      inno_log_error("invalid return for %s %s, %s, use default value %f", section, key, config_buffer, ret);
    }
    free(config_buffer);
  } else {
    inno_log_error("cannot get %s %s, use default value %f", section, key, ret);
  }
  return ret;
}

int LidarCommunication::set_config_section_key_value(const char *section, const char *key, const char *value) {
  set_default_timeout_sec(2.0);
  char *config_buffer = send_command_and_get_reply("%s %s %s %s", "set_i_config", section, key, value);
  set_default_timeout_sec(kDefaultTimeoutSec);
  if (config_buffer != NULL) {
    free(config_buffer);
    return 0;
  } else {
    inno_log_error("cannot set %s %s to %s", section, key, value);
    return -1;
  }
}

int LidarCommunication::set_reboot(int value) {
  inno_log_warning("lidar set reboot.");
  return send_command_and_free_reply("reboot %d", value);
}

int LidarCommunication::reg_read(uint32_t addr, uint32_t *data) {
  int ret = -1;
  char *recv_buffer = send_command_and_get_reply("reg_rd %x", addr);
  if (recv_buffer == NULL) {
    inno_log_error("Failed to reg_read 0x%x", addr);
  } else {
    if (sscanf(recv_buffer, "reg_rd: %x %x", &addr, data) != 2) {
      inno_log_error("Failed to reg_read 0x%x, invalid reply %s", addr, recv_buffer);
    } else {
      ret = 0;
    }
    free(recv_buffer);
  }
  return ret;
}

int LidarCommunication::reg_write(uint32_t addr, uint32_t data) {
  if (send_command_and_free_reply("reg_wr %x %x", addr, data) < 0) {
    inno_log_error("Failed to reg_write 0x%x 0x%x", addr, data);
    return -1;
  } else {
    return 0;
  }
}

/*
 * Get lidar mode section.
 * Return query section of lidar mode.
 * Return "quiet_mode", if lidar in protection mode.
 */
const char *LidarCommunication::get_mode_section(InnoLidarMode mode) {
  int calibration_mode_id = 0;
  switch (mode) {
    case InnoLidarMode::INNO_LIDAR_MODE_WORK_NORMAL:
      return "motor";
    case InnoLidarMode::INNO_LIDAR_MODE_WORK_SHORT_RANGE:
      return "shortrange_mode";
    case InnoLidarMode::INNO_LIDAR_MODE_WORK_CALIBRATION:
      if (get_calibration_mode_id(&calibration_mode_id) < 0) {
        inno_log_error("Failed to get calibration id.");
        return NULL;
      }
      return calibration_mode_id ? "calibration_mode" : "calibration_mode_1";
    case InnoLidarMode::INNO_LIDAR_MODE_PROTECTION:
      return "quiet_mode";
    case InnoLidarMode::INNO_LIDAR_MODE_WORK_QUIET:
      return "quiet_mode";
    case InnoLidarMode::INNO_LIDAR_MODE_WORK_EXHIBITION:
      return "exhibition_mode";
    default:
      return "motor";
  }
}

/*
* Get frame rate with lidar mode.
* If lidar in calibration mode,
* also need to get calibration mode id.
*/
const char *LidarCommunication::get_current_mode_section() {
  InnoLidarMode mode;
  InnoLidarMode pre_mode;
  InnoLidarStatus status;
  int ret = get_mode_status(&mode, &pre_mode, &status);
  if (ret < 0) {
    inno_log_error("Failed to get mode status.");
    return NULL;
  }
  return get_mode_section(mode);
}

/*
* Get frame rate with lidar mode.
* If lidar in calibration mode,
* also need to get calibration mode id.
*/
double LidarCommunication::get_frame_rate() {
  char model[2];
  int ret = get_model(model, sizeof(model));
  if (ret < 0) {
    inno_log_error("Failed to get lidar model");
    return -1.0;
  }
  // w:robinw, l: robine-lite
  if (strcmp(model, "w") == 0 || strcmp(model, "l") == 0) {
    char *recv_buffer = send_command_and_get_reply("get_frame_rate");
    if (recv_buffer == NULL) {
      inno_log_warning("Failed to get_frame_rate");
      return -1.0;
    } else {
      double frame_rate = 0;
      if (sscanf(recv_buffer, "%lf", &frame_rate) != 1) {
        inno_log_error("Invalid get_frame_rate reply %s", recv_buffer);
        free(recv_buffer);
        return -1.0;
      }
      free(recv_buffer);
      return frame_rate;
    }
  }

  const char *section = get_current_mode_section();
  if (!section) {
    inno_log_error("Failed to create section of get frame rate.");
    return -1.0;
  }
  return get_config_section_key_value(section, "galvo_framerate", 10);
}

int LidarCommunication::clear_fw_inner_faults(uint32_t fid) {
  return send_command_and_free_reply("clear_inner_faults %u",
                                      fid);
}

int LidarCommunication::get_galvo_mode() {
  return get_config_section_key_value("motor", "galvo_mode", 1);
}

int LidarCommunication::get_fw_version(char *buffer, size_t buffer_len) {
  int ret = send_command_and_save_reply_(buffer, buffer_len, "get_version");
  if (ret > 0) {
    return 0;
  } else {
    return ret - 1;
  }
}

int LidarCommunication::get_board_name(char *buffer, size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_hw_part_name");
}

int LidarCommunication::get_did(char *buffer, size_t buffer_len, const char *did) {
  char *recv_buffer = send_command_and_get_reply("get_did %s", did);
  if (recv_buffer == NULL) {
    inno_log_warning("Failed to get_did %s", did);
    return -1;
  } else {
    // HACK: got a response, from now on, change the
    // default timeout to 0.5 seconds
    set_default_timeout_sec(0.5);
    size_t len = strlen(recv_buffer);
    size_t prefix_chars = 14;
    size_t ignore_chars = prefix_chars + 12;
    if (len < ignore_chars) {
      inno_log_warning("DID not in FW?  Reply too short %s", recv_buffer);
      free(recv_buffer);
      return -2;
    }
    if (buffer_len < len - (ignore_chars - 1)) {
#ifdef _WIN32
      inno_log_warning("DID buffer too small");
#else
      inno_log_warning("DID buffer too small %zu < %zu", buffer_len, len - (ignore_chars - 1));
#endif
      free(recv_buffer);
      return -3;
    } else {
      // Success return value is a string
      //     00 [2 bytes did len] [2 bytes did #] [did val] [2 bytes CRC]
      // So did val skips char at front and end
      inno_log_info("get_did %s ==recv==> %s", did, recv_buffer);
      memcpy(buffer, recv_buffer + prefix_chars, len - ignore_chars + 1);
      free(recv_buffer);
      if (buffer[len - ignore_chars] == ' ') {
        buffer[len - ignore_chars] = '\0';
      }
      buffer[len - ignore_chars + 1] = '\0';
      inno_log_info("get_did %s ==ret==> %s", did, buffer);
      return 0;
    }
  }
}

int LidarCommunication::get_fw_sequence(char *buffer, size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_fw_seq_number");
}

int LidarCommunication::get_sn(char *buffer, size_t buffer_len) {
  char *recv_buffer = send_command_and_get_reply("get_sn");
  if (recv_buffer == NULL) {
    inno_log_warning("Failed to get_sn");
    return -1;
  } else {
    // HACK: got a response, from now on, change the
    // default timeout to 0.5 seconds
    set_default_timeout_sec(0.5);
    char sn[64];
    sn[sizeof(sn) - 1] = 0;
    if (sscanf(recv_buffer, "serial number: %63s", sn) != 1) {
      inno_log_error("Invalid get_sn reply %s", recv_buffer);
      free(recv_buffer);
      return -2;
    }
    free(recv_buffer);
    size_t len = strlen(sn);
    if (buffer_len < len + 1) {
      return -3;
    } else {
      memcpy(buffer, sn, len);
      buffer[len] = 0;
      return 0;
    }
  }
}

int LidarCommunication::get_HW_config(int* hw_config) {
  inno_log_verify(hw_config, "hw_config is NULL");
  char *recv_buffer = send_command_and_get_reply("get_HW_config");
  if (recv_buffer == NULL) {
    inno_log_warning("Failed to get_HW_config");
    return -1;
  } else {
    if (sscanf(recv_buffer, "HW_config: %d", hw_config) != 1) {
      inno_log_error("Invalid get_HW_config reply %s", recv_buffer);
      free(recv_buffer);
      return -2;
    }
    free(recv_buffer);
    return 0;
  }
}
int LidarCommunication::get_model(char *buffer, size_t buffer_len) {
  inno_log_verify(buffer_len >= 2, "buffer_len too small %" PRI_SIZEU "", buffer_len);
  char *recv_buffer = send_command_and_get_reply("get_model");
  if (recv_buffer == NULL) {
    inno_log_warning("Failed to get_model");
    return -1;
  } else {
    char model;
    if (sscanf(recv_buffer, "model: %c", &model) != 1) {
      inno_log_error("Invalid get_model reply %s", recv_buffer);
      free(recv_buffer);
      return -3;
    } else {
      // successful
      inno_log_verify(buffer_len >= 2, "buffer_len %" PRI_SIZEU, buffer_len);
      free(recv_buffer);
      buffer[0] = model;
      buffer[1] = 0;
      return 0;
    }
  }
}

int LidarCommunication::get_temperature(char *buffer, size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_temps");
}

int LidarCommunication::get_detector_temps(char *buffer, size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_det_temps");
}

int LidarCommunication::get_motor_speeds(char *buffer, size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_motor_speeds");
}

int LidarCommunication::get_internal_motor_config(char *buffer,
                                                  size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_i_config motor");
}

int LidarCommunication::get_galvo_first_period(char *buffer,
                                               size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_galvo_first_period");
}

int LidarCommunication::get_internal_quiet_config(char *buffer,
                                                  size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_i_config quiet_mode");
}

int LidarCommunication::get_saved_roi_enable(char *buffer,
                                                  size_t buffer_len) {
  return get_reply_info_(buffer, buffer_len, "get_i_config saved_roi enable");
}
int LidarCommunication::get_reply_info_(char *recv_buffer,
                                      size_t recv_buffer_len,
                                      const char *send_buffer) {
  char *recv = send_command_and_get_reply("%s", send_buffer);
  if (recv == NULL) {
    inno_log_warning("%s failed", send_buffer);
    return -1;
  } else {
    // HACK: got a response, from now on, change the
    // default timeout to 0.5 seconds
    set_default_timeout_sec(0.5);
    size_t len = strlen(recv);
    if (recv_buffer_len < len + 1) {
      free(recv);
      return -2;
    } else {
      memcpy(recv_buffer, recv, len);
      recv_buffer[len] = 0;
      free(recv);
      return 0;
    }
  }
}

int LidarCommunication::get_geo_yaml(char *buffer, size_t buffer_len) {
  int expect_md5 = 1;
  int ret = recv_length_buffer(buffer, buffer_len, expect_md5, "download_cal_file 0");
  if (ret < 0) {
    inno_log_error("failed to download_cal_file");
    // alarm_callback(INNO_ALARM_ERROR, INNO_ALARM_CODE_BAD_CONFIG_YAML,
    //               "cannot download from controller");
    return -1;
  } else if (ret <= 1) {
    inno_log_error("download_cal_file returns too small %d", ret);
    // alarm_callback(INNO_ALARM_ERROR, INNO_ALARM_CODE_BAD_CONFIG_YAML,
    //               "cannot download from controller");
    return -2;
  } else {
    return 0;
  }
}

int LidarCommunication::get_ext_packet_format(bool *is_extended_format) {
  inno_log_verify(is_extended_format, "is_extended_format should not be NULL");
  uint32_t data;
  if (reg_read(kCapabilitiesRegister, &data) < 0) {
    return -1;
  }
  if (data & kExtendedPacketFormat) {
    *is_extended_format = true;
  } else {
    *is_extended_format = false;
  }
  return 0;
}

int LidarCommunication::get_status(StatusCounters *counters, bool get_diff) {
  inno_log_verify(counters, "counters should not be NULL");
  char *recv_buffer = send_command_and_get_reply("get_status");
  if (recv_buffer == NULL) {
    inno_log_error("Cannot get_status");
    return -2;
  }
  if (counters->set(recv_buffer)) {
    inno_log_error("Cannot get_status, invalid buffer %s", recv_buffer);
    free(recv_buffer);
    return -1;
  } else {
    free(recv_buffer);
  }

  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (get_status_called_cnt_ == 0) {
      first_ctrlor_lose_1_ = counters->lose_counter_1;
      first_ctrlor_lose_2_ = counters->lose_counter_2;
      last_ctrlor_lose_1_ = counters->lose_counter_1;
      last_ctrlor_lose_2_ = counters->lose_counter_2;
    }

    if (get_diff) {
      uint32_t tmp1, tmp2;
      tmp1 = last_ctrlor_lose_1_;
      tmp2 = last_ctrlor_lose_2_;
      last_ctrlor_lose_1_ = counters->lose_counter_1;
      last_ctrlor_lose_2_ = counters->lose_counter_2;

      counters->lose_counter_1 -= tmp1;
      counters->lose_counter_2 -= tmp2;
    }
    get_status_called_cnt_ += 1;
  }

  // xxx todo: handle lose counters
  return 0;
}

int LidarCommunication::set_raw_capture_mode(bool use_raw_mode) {
  unsigned int data;

  if (use_raw_mode) {
    inno_log_panic("use raw_mode not implemented %d", use_raw_mode);
    return -1;
  } else {
    if (reg_read(kDmaControlRegister, &data) < 0) {
      return -1;
    }
    data = data & ~kDmaRawDataSelected;
    if (reg_write(kDmaControlRegister, data) < 0) {
      return -2;
    }
    return 0;
  }
}

int LidarCommunication::set_vertical_roi(double vertical_roi_angle) {
  set_default_timeout_sec(2.0);
  auto ret = send_command_and_free_reply("set_vertical_roi %f", vertical_roi_angle);
  set_default_timeout_sec(kDefaultTimeoutSec);
  return ret;
}

int LidarCommunication::get_vertical_roi(double *vertical_roi_angle) {
  inno_log_verify(vertical_roi_angle, "vertical_roi_angle");
  char *recv_buffer = send_command_and_get_reply("get_vertical_roi");
  int ret = 0;
  if (recv_buffer == NULL) {
    inno_log_error("Failed to get_vertical_roi");
    return -1;
  } else {
    if (sscanf(recv_buffer, "%lf", vertical_roi_angle) != 1) {
      inno_log_error("Failed to get_vertical_roi, invalid reply %s", recv_buffer);
      ret = -2;
    } else {
      ret = 0;
    }
    free(recv_buffer);
  }
  return ret;
}

int LidarCommunication::set_fw_inner_fault_do(enum InnoLidarInFault fid) {
  return send_command_and_free_reply("set_inner_fault_do %d", static_cast<int>(fid));
}

int LidarCommunication::set_mode(enum InnoLidarMode mode,
                                 enum InnoLidarMode *pre_mode,
                                 enum InnoLidarStatus *status) {
  set_default_timeout_sec(5.0);
  char *recv_buffer = send_command_and_get_reply("set_mode %d", mode);
  set_default_timeout_sec(kDefaultTimeoutSec);
  if (recv_buffer == NULL) {
    inno_log_error("Cannot set_mode %d", mode);
    return -2;
  }

  int pm;
  int st;
  int ret = 0;
  if (2 != sscanf(recv_buffer, "%d %d",
                  pre_mode ? reinterpret_cast<int *>(pre_mode) : &pm,
                  status ? reinterpret_cast<int *>(status) : &st)) {
    inno_log_error("bad reply for set_mode %s",
                   recv_buffer);
    ret = -2;
  }
  free(recv_buffer);
  return ret;
}

int LidarCommunication::get_calibration_mode_id(int *calibration_id) {
  char *recv_buffer = send_command_and_get_reply("calibration_mode_id");
  if (recv_buffer == NULL) {
    inno_log_error("Cannot get_calibration_mode_id");
    return -2;
  }
  if (sscanf(recv_buffer, "%d", calibration_id) != 1) {
    inno_log_error("bad reply for get_calibration_mode_id %s", recv_buffer);
    free(recv_buffer);
    return -2;
  }
  if (*calibration_id != 0 && *calibration_id != 1) {
    inno_log_error("invalid calibration mode id %s", recv_buffer);
    free(recv_buffer);
    return -2;
  }
  free(recv_buffer);
  return 0;
}

int LidarCommunication::get_mode_status(enum InnoLidarMode *mode,
                                        enum InnoLidarMode *pre_mode,
                                        enum InnoLidarStatus *status) {
  char *recv_buffer = send_command_and_get_reply("get_mode");
  if (recv_buffer == NULL) {
    inno_log_error("Cannot get_mode");
    return -2;
  }

  int md;
  int pm;
  int st;
  int ret = 0;
  if (3 != sscanf(recv_buffer, "lastmode=%d mode=%d status=%d",
                  pre_mode ? reinterpret_cast<int *>(pre_mode) : &pm,
                  mode ? reinterpret_cast<int *>(mode) : &md,
                  status ? reinterpret_cast<int *>(status) : &st)) {
    inno_log_error("bad reply for get_mode %s",
                   recv_buffer);
    ret = -2;
  }
  free(recv_buffer);
  return ret;
}

int LidarCommunication::get_apd_cal_status(int *status) {
  char *recv_buffer = send_command_and_get_reply("get_apdstreamingcal_status");
  if (recv_buffer == NULL) {
    inno_log_error("Cannot get_apdstreamingcal_status");
    return -2;
  }
  if (sscanf(recv_buffer, "%d", status) != 1) {
    inno_log_error("bad reply for get_apdstreamingcal_status %s",
                    recv_buffer);
    free(recv_buffer);
    return -2;
  }
  free(recv_buffer);
  return 0;
}

int LidarCommunication::open_streaming(void) {
  {
    std::unique_lock<std::mutex> lk(mutex_);
    inno_log_verify(streaming_fd_ == -1, "streaming_fd is already %d", streaming_fd_);
  }

  int fd = get_connection(0.5, 512 * 1024);
  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (fd < 0) {
      streaming_fd_ = -1;
      return -1;
    } else {
      streaming_fd_ = fd;
      return 0;
    }
  }
}

int LidarCommunication::start_streaming(bool is_tcp) {
  inno_log_verify(streaming_fd_ >= 0, "streaming_fd %d is not valid", streaming_fd_);
  int zero = 0;
  int ret = send_command_with_fd(streaming_fd_, NULL, &zero, is_tcp ? "start %hu" : "start direct %hu", service_port_);
  if (ret < 0) {
    inno_log_error("%s cannot send start (%d)", ip_, ret);
    return -10;
  } else {
    inno_log_info("%s start command sent", ip_);
    return 0;
  }
}

int LidarCommunication::get_streaming_fd() {
  return streaming_fd_;
}

void LidarCommunication::close_streaming() {
  std::unique_lock<std::mutex> lk(mutex_);
  if (streaming_fd_ >= 0) {
    close(streaming_fd_);
  }
  streaming_fd_ = -1;
}

int LidarCommunication::send_stop() {
  {
    std::unique_lock<std::mutex> lk(mutex_);
    inno_log_verify(streaming_fd_ == -1, "streaming_fd is already %d", streaming_fd_);
  }

  return send_command_and_free_reply("stop");
}

double LidarCommunication::get_motor_speed_config() {
  const char *section = get_current_mode_section();
  if (!section) {
    inno_log_error("Failed to get mode section");
    return -1.0;
  }
  return get_config_section_key_value(section, "mtr_f_rpm", -1.0);
}
double LidarCommunication::get_ptp_time_stamping() {
  return get_config_section_key_value("time", "time_stamping", 0);
}
}  // namespace innovusion
