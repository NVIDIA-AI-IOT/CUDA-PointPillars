/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include "sdk_client/lidar_client_communication.h"

#include <vector>

#include "sdk_client/lidar_client.h"
#include "utils/log.h"
#include "utils/md5.h"
namespace innovusion {

LidarClientCommunication::LidarClientCommunication(const char *ip, unsigned short port, double timeout_sec)
    : NetManager(ip, port, timeout_sec) {
}

LidarClientCommunication::LidarClientCommunication(InnoLidarClient *client, const char *ip, unsigned short port,
                                                   double timeout_sec)
    : NetManager(ip, port, timeout_sec) {
  lidar_client_ = client;
  lidar_client_->add_config(&config_base_);
  config_.copy_from_src(&config_base_);
}

LidarClientCommunication::~LidarClientCommunication() {
}

int LidarClientCommunication::send_set_command_(const char *set_command) {
  // set_command in the format of : name1=value1
  int status_code;
  char buffer[kSmallBufferSize] = {0};
  size_t buffer_len = sizeof(buffer);
  char url[kMaxUrlSize] = {0};
  size_t sn = snprintf(url, sizeof(url), "/command/?set_%s", set_command);
  if (sn >= sizeof(url)) {
    inno_log_panic("set_command too long %s", set_command);
    return -1;
  }
  char *content;
  int content_size;
  config_.copy_from_src(&config_base_);
  int ret = http_get(url, buffer, buffer_len, &status_code, &content, &content_size, config_.get_conn_timeout_sec);
  if (ret == 0) {
    if (status_code == 200) {
      return 0;
    } else {
      inno_log_warning("%s unsuccessful (HTML code %d) -- returned %s", url, status_code, buffer);
      return -2;
    }
  } else {
    return ret;
  }
}

int LidarClientCommunication::send_get_command_(const char *name, char *buffer, size_t buffer_len) {
  return send_get_command_(name, buffer, buffer_len, nullptr);
}

int LidarClientCommunication::send_get_command_(const char *name, char *buffer, size_t buffer_len, const char *value) {
  char url[kMaxUrlSize] = {0};
  char response[kMaxReplySize] = {0};
  size_t sn;
  if (value) {
    sn = snprintf(url, sizeof(url), "/command/?get_%s=%s", name, value);
  } else {
    sn = snprintf(url, sizeof(url), "/command/?get_%s", name);
  }
  if (sn >= sizeof(url)) {
    inno_log_panic("get_command too long %s=%s", name, value ? value : "");
    return -1;
  }
  char *content;
  int content_size;
  int status_code;
  config_.copy_from_src(&config_base_);
  int ret =
      http_get(url, response, sizeof(response), &status_code, &content, &content_size, config_.get_conn_timeout_sec);
  if (ret == 0) {
    if (status_code == 200) {
      if (content_size + 1 <= ssize_t(buffer_len)) {
        memmove(buffer, content, content_size + 1);
        return 0;
      } else {
        inno_log_error("get_command %s return too big %d %" PRI_SIZELU, name, content_size + 1, buffer_len);
        return -2;
      }
    } else {
      inno_log_warning("%s unsuccessful (HTML code %d) -- returned %s", url, status_code, buffer);
      return -3;
    }
  } else {
    return ret;
  }
}

int LidarClientCommunication::get_anglehv_table(char *buffer, size_t buffer_len) {
  char url[kMaxUrlSize];
  std::vector<char> response(buffer_len * 2);
  size_t sn;
  sn = snprintf(url, sizeof(url), "/command/?get_%s", "anglehv_table");
  char *content;
  int content_size;
  int status_code;
  config_.copy_from_src(&config_base_);
  int ret = http_get(url, response.data(), response.size(), &status_code, &content, &content_size,
                     config_.get_conn_timeout_sec);
  if (ret == 0) {
    if (status_code == 200) {
      if (content_size <= ssize_t(buffer_len)) {
        memmove(buffer, content, content_size);
        return 0;
      } else {
        inno_log_error("size mismatch %d %" PRI_SIZELU, content_size, buffer_len);
        return -2;
      }
    } else {
      inno_log_warning("%s unsuccessful (HTML code %d)", url, status_code);
      return -3;
    }
  } else {
    return ret;
  }
}

int LidarClientCommunication::get_attribute(const char *attr, char *buffer_in, size_t buffer_len) {
  inno_log_verify(attr && buffer_in, "%p %p", attr, buffer_in);
  char buffer[kSmallBufferSize];
  int ret = send_get_command_(attr, buffer, sizeof(buffer));
  if (ret == 0) {
    if (strlen(buffer) < buffer_len) {
      memcpy(buffer_in, buffer, strlen(buffer) + 1);
    } else {
      inno_log_warning("buffer too small %" PRI_SIZELU " %" PRI_SIZELU, strlen(buffer), buffer_len);
      ret = -1;
    }
  }
  return ret;
}

int LidarClientCommunication::get_attribute(const char *attr, double *value) {
  inno_log_verify(attr && value, "%p %p", attr, value);
  char buffer[kSmallBufferSize];
  int ret = send_get_command_(attr, buffer, sizeof(buffer));
  if (ret == 0) {
    if (1 == sscanf(buffer, "%lf", value)) {
      return 0;
    } else {
      return -1;
    }
  }
  return ret;
}

int LidarClientCommunication::get_fw_state(char *buffer, size_t buffer_len) {
  return send_get_command_("fw_state", buffer, buffer_len);
}

int LidarClientCommunication::get_fw_version(char *buffer, size_t buffer_len) {
  return send_get_command_("fw_version", buffer, buffer_len);
}

int LidarClientCommunication::get_sn(char *buffer, size_t buffer_len) {
  return send_get_command_("sn", buffer, buffer_len);
}

int LidarClientCommunication::get_model(char *buffer, size_t buffer_len) {
  return send_get_command_("model", buffer, buffer_len);
}

int LidarClientCommunication::get_debug(InnoLogLevel *debug_level) {
  char buffer[kSmallBufferSize];
  int ret = send_get_command_("debug", buffer, sizeof(buffer));
  if (ret == 0) {
    *debug_level = (InnoLogLevel)atoi(buffer);
  }
  return ret;
}

int LidarClientCommunication::get_roi(double *h_roi, double *v_roi) {
  inno_log_verify(h_roi, "h_roi");
  inno_log_verify(v_roi, "v_roi");

  char buffer[kSmallBufferSize];
  int ret = send_get_command_("roi", buffer, sizeof(buffer));
  if (ret == 0) {
    int ss = sscanf(buffer, "%lf,%lf", h_roi, v_roi);
    if (ss != 2) {
      ret = 1;
      inno_log_error("get_roi return %s", buffer);
    }
  }
  return ret;
}

int LidarClientCommunication::get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode,
                                              enum InnoLidarStatus *status, uint64_t *in_transition_mode_ms) {
  inno_log_verify(mode, "NULL pointer");
  inno_log_verify(pre_mode, "NULL pointer");
  inno_log_verify(status, "NULL pointer");
  inno_log_verify(in_transition_mode_ms, "NULL pointer");
  char buffer[kSmallBufferSize];
  int ret = send_get_command_("mode_status", buffer, sizeof(buffer));
  if (ret == 0) {
    if (4 == sscanf(buffer, "%d,%d,%d,%" PRI_SIZEU, reinterpret_cast<int *>(mode), reinterpret_cast<int *>(pre_mode),
                    reinterpret_cast<int *>(status), in_transition_mode_ms)) {
      return 0;
    } else {
      return -1;
    }
  }
  return ret;
}

int LidarClientCommunication::get_server_udp_ports_ip(int32_t *port_data, int32_t *port_message, int32_t *port_status,
                                                      char *ip, size_t ip_len, char *ip2, size_t ip2_len) {
  inno_log_verify(port_data, "NULL pointer");
  inno_log_verify(port_message, "NULL pointer");
  inno_log_verify(port_status, "NULL pointer");
  inno_log_verify(ip_len > 32, "%" PRI_SIZELU " must > 32", ip_len);
  ip[ip_len - 1] = 0;
  inno_log_verify(ip2_len > 32, "%" PRI_SIZELU " must > 32", ip2_len);
  ip2[ip2_len - 1] = 0;

  char buffer[kSmallBufferSize];
  char dummy[33];
  int ret = send_get_command_("udp_ports_ip", buffer, sizeof(buffer));
  if (ret != 0) {
    return -1;
  }

  if (6 == sscanf(buffer, "%d,%d,%d,%32[^,],%32[^,],%32s", port_data, port_message, port_status, ip, dummy, ip2)) {
    return 0;
  } else if (4 == sscanf(buffer, "%d,%d,%d,%32s", port_data, port_message, port_status, ip)) {
    ip2[0] = 0;
    return 0;
  } else {
    return -1;
  }
}

int LidarClientCommunication::set_roi(double h_roi, double v_roi) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "roi=%f,%f", h_roi, v_roi);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_reflectance_mode(InnoReflectanceMode val) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "reflectance_mode=%d", val);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_return_mode(InnoMultipleReturnMode val) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "return_mode=%d", val);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_debug(InnoLogLevel val) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "debug=%d", val);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_mode(enum InnoLidarMode val) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "mode=%d", val);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_reboot(int value) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "reboot=%d", value);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_server_udp_ports_ip(uint16_t port) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "udp_ports_ip=%hu,%hu,%hu", port, port, port);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::get_server_udp_raw_port(int32_t *port_raw, char *ip, size_t ip_len) {
  inno_log_verify(port_raw, "port_raw is NULL pointer");
  inno_log_verify(ip, "ip is NULL pointer");
  inno_log_verify(ip_len > 32, "ip_len: %" PRI_SIZELU " must > 32", ip_len);

  ip[ip_len - 1] = 0;

  char buffer[kSmallBufferSize]{0};
  int ret = send_get_command_("udp_raw_port", buffer, sizeof(buffer));
  if (ret == 0) {
    if (2 == sscanf(buffer, "%d,%32s", port_raw, ip)) {
      return 0;
    } else {
      return -1;
    }
  }

  return ret;
}

int LidarClientCommunication::set_server_udp_raw_port(uint16_t port_raw) {
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "udp_raw_port=%hu", port_raw);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_faults_save_raw(const std::string& value_hex_str) {
  char cmd[kSmallBufferSize];

  int sn = snprintf(cmd, sizeof(cmd), "faults_save_raw=%s", value_hex_str.c_str());
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_faults_save_raw(uint64_t value) {
  char cmd[kSmallBufferSize];

  int sn = snprintf(cmd, sizeof(cmd), "faults_save_raw=%" PRI_SIZED, value);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}

int LidarClientCommunication::set_attribute_string(const char *attribute, const char *buf) {
  inno_log_verify(attribute && buf, "attribute or buf is NULL");
  char cmd[kSmallBufferSize];
  int sn = snprintf(cmd, sizeof(cmd), "%s=%s", attribute, buf);
  inno_log_verify(sn < static_cast<int>(sizeof(cmd)), "exceed buf");
  return send_set_command_(cmd);
}


}  // namespace innovusion
