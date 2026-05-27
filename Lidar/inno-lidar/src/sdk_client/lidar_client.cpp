/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/lidar_client.h"

#ifndef _WIN32
#include <sys/time.h>
#endif

#include <vector>

#include "sdk_client/client_stats.h"
#include "sdk_client/lidar_client_communication.h"
#include "sdk_client/ring_id_converter/ring_id_converter.h"
#include "sdk_client/stage_client_deliver.h"
#include "sdk_client/stage_client_read.h"
#include "sdk_client/stage_client_recorder.h"
#include "sdk_common/inno_lidar_other_api.h"
#include "utils/consumer_producer.h"
#include "utils/mem_pool_manager.h"
#include "utils/utils.h"

namespace innovusion {

/**********************
 * constructor + destructor
 **********************/

InnoLidarClient::InnoLidarClient(const char *name, const char *lidar_ip, uint16_t port, bool use_tcp, uint16_t udp_port)
    : InnoLidarBase("LidarClient_", name) {
  init_();
  ip_ = strdup(lidar_ip);
  port_ = port;
  lidar_source_ = LIDAR_SOURCE_LIVE;

  comm_ = new LidarClientCommunication(this, ip_, port_, 0.5);
  inno_log_verify(comm_, "%s cannot allocate comm", name_);
  InputParam param;
  memset(&param, 0, sizeof(param));
  if (use_tcp) {
    param.tcp_param.source_type = SOURCE_TCP;
    strncpy(param.tcp_param.lidar_ip, lidar_ip, sizeof(param.tcp_param.lidar_ip) - 1);
    param.tcp_param.lidar_ip[sizeof(param.tcp_param.lidar_ip) - 1] = '\0';
    stage_read_ = new StageClientRead(this, comm_, &param);
  } else {
    param.udp_param.source_type = SOURCE_UDP;
    param.udp_param.udp_port = udp_port;
    strncpy(param.udp_param.lidar_ip, lidar_ip, sizeof(param.udp_param.lidar_ip) - 1);
    param.udp_param.lidar_ip[sizeof(param.udp_param.lidar_ip) - 1] = '\0';
    stage_read_ = new StageClientRead(this, comm_, &param);
  }

  inno_log_verify(stage_read_, "%s cannot allocate stage_read", name_);
  inno_log_info("%s uses live lidar at %s port=%hu udp_port=%hu", name_, ip_, port, udp_port);

  return;
}

InnoLidarClient::InnoLidarClient(const char *name, const char *filename, int play_rate, int rewind, int64_t skip)
    : InnoLidarBase("LidarClient_", name) {
  init_();
  filename_ = strdup(filename);
  lidar_source_ = LIDAR_SOURCE_FILE;
  comm_ = NULL;

  InputParam param;
  param.file_param.source_type = SOURCE_FILE;
  strncpy(param.file_param.filename, filename, sizeof(param.file_param.filename) - 1);
  param.file_param.filename[sizeof(param.file_param.filename) - 1] = '\0';
  param.file_param.play_rate = play_rate;
  param.file_param.rewind = rewind;
  param.file_param.skip = skip;

  stage_read_ = new StageClientRead(this, comm_, &param);
  inno_log_verify(stage_read_, "%s cannot allocate stage_read", name_);
  inno_log_info("%s open file %s, rate=%dMB/s %fX", name_, filename_, play_rate_, play_rate_x_);
  return;
}

InnoLidarClient::InnoLidarClient(const char *name, void *ctx) : InnoLidarBase("LidarClient_", name) {
  init_();
  InputParam *p = reinterpret_cast<InputParam *>(ctx);
  if (p->base_param.source_type == SOURCE_TCP || p->base_param.source_type == SOURCE_UDP) {
    comm_ = new LidarClientCommunication(this, ip_, port_, 0.5);
    inno_log_verify(comm_, "%s cannot allocate comm", name_);
  }

  stage_read_ = new StageClientRead(this, comm_, ctx);
  inno_log_verify(stage_read_, "%s cannot allocate stage_read", name_);
  return;
}

InnoLidarClient::~InnoLidarClient() {
  inno_log_info("%s close", name_);
  bool still_up;
  {
    std::unique_lock<std::mutex> lk(last_stage_mutex_);
    still_up = last_stage_is_up_;
  }
  inno_log_verify(!still_up, "forgot to call lidar stop?");

#if !(defined (_QNX_) || defined (_WIN32))
  if (it_raw_recorder_) {
    delete it_raw_recorder_;
    it_raw_recorder_ = nullptr;
  }

  if (raw_recorder_) {
    delete raw_recorder_;
    raw_recorder_ = nullptr;
  }
#endif

  if (ip_) {
    free(ip_);
    ip_ = NULL;
  }
  if (filename_) {
    free(filename_);
    filename_ = NULL;
  }
  if (comm_) {
    delete comm_;
    comm_ = NULL;
  }
  if (cpuset_) {
    free(cpuset_);
    cpuset_ = NULL;
  }

  if (stage_read_) {
    delete stage_read_;
    stage_read_ = NULL;
  }
  if (packet_pool_) {
    delete packet_pool_;
    packet_pool_ = NULL;
  }

  if (galvo_mirror_check_) {
    delete galvo_mirror_check_;
    galvo_mirror_check_ = NULL;
  }

  if (max_distance_check_) {
    delete max_distance_check_;
    max_distance_check_ = NULL;
  }

  if (ring_id_converter_) {
    delete ring_id_converter_;
    ring_id_converter_ = NULL;
  }

  if (anglehv_table_) {
    delete[] reinterpret_cast<char *>(anglehv_table_);
    anglehv_table_ = NULL;
  }
  return;
}

/**********************
 * private methods
 **********************/
void InnoLidarClient::init_() {
  // handle_
  inno_log_verify(handle_ == 0, "%s cannot init twice handle_=%d", name_, handle_);
  {
    std::unique_lock<std::mutex> lk(static_mutex_s);
    handle_ = ++max_handle_s;
  }

  comm_ = NULL;
  ip_ = NULL;
  port_ = 0;
  filename_ = NULL;

  play_rate_ = 0;
  play_rate_x_ = 0;

  cp_read_ = NULL;
  stage_deliver_ = NULL;
  stage_deliver2_ = NULL;
  packet_pool_ = new MemPool("packet_pool", kMaxPacketSize, kPacketPoolSize, 32);
  anglehv_table_ = reinterpret_cast<InnoDataPacket *>(new char[kInnoAngleHVTableMaxSize]);
  inno_log_verify(packet_pool_, "packet_pool");

  force_xyz_pointcloud_ = false;

  client_stats_ = NULL;
  galvo_mirror_check_ = NULL;
  max_distance_check_ = NULL;
}

InnoLidarBase::State InnoLidarClient::get_state_() {
  return stage_read_->get_state();
}

bool InnoLidarClient::is_live_lidar_() const {
  return comm_ != NULL;
}

void *InnoLidarClient::alloc_buffer_(size_t size) {
  inno_log_verify(size <= kMaxPacketSize, "%" PRI_SIZELU " too big", size);
  return packet_pool_->alloc();
}

void InnoLidarClient::free_buffer_(void *buffer) {
  packet_pool_->free(buffer);
}

void InnoLidarClient::add_deliver_job_(void *job) {
  cp_deliver_->add_job(job);
}

void InnoLidarClient::add_deliver2_job_(void *job) {
  cp_deliver2_->add_job(job);
}

void InnoLidarClient::add_recorder_job_(void *job) {
  cp_recorder_->add_job(job);
}

int InnoLidarClient::set_config_name_value(const char *name, const char *value) {
  return config_manage_.set_config_key_value(name, value, true);
}

int InnoLidarClient::set_reflectance_mode(enum InnoReflectanceMode mode) {
  if (mode <= INNO_REFLECTANCE_MODE_NONE || mode >= INNO_REFLECTANCE_MODE_MAX) {
    inno_log_error("Invalid ReflectanceMode %d", mode);
    return -1;
  }
  if (is_live_lidar_()) {
    return comm_->set_reflectance_mode(mode);
  } else {
    inno_log_info("file replay fake set reflectance");
    return 0;
  }
}

int InnoLidarClient::set_return_mode(enum InnoMultipleReturnMode mode) {
  if (mode <= INNO_MULTIPLE_RETURN_MODE_NONE || mode >= INNO_MULTIPLE_RETURN_MODE_MAX) {
    inno_log_error("Invalid MultipleReturnMode %d", mode);
    return -1;
  }
  if (is_live_lidar_()) {
    return comm_->set_return_mode(mode);
  } else {
    inno_log_info("file replay fake set return mode");
    return 0;
  }
}

int InnoLidarClient::set_roi(double h, double v) {
  if (is_live_lidar_()) {
    return comm_->set_roi(h, v);
  } else {
    inno_log_info("file replay fake set roi");
    return 0;
  }
}

int InnoLidarClient::get_roi(double *h, double *v) {
  if (is_live_lidar_()) {
    return comm_->get_roi(h, v);
  } else {
    *h = 0;
    *v = 0;
    return 0;
  }
}

int InnoLidarClient::set_mode(enum InnoLidarMode mode, enum InnoLidarMode *mode_before_change,
                              enum InnoLidarStatus *status_before_change) {
  if (is_live_lidar_()) {
    int ret = 0;
    if (mode_before_change) {
      enum InnoLidarMode pre_pre_mode;
      enum InnoLidarStatus status;
      uint64_t in_transition_mode_ms;
      ret = comm_->get_mode_status(mode_before_change, &pre_pre_mode,
                                   status_before_change ? status_before_change : &status, &in_transition_mode_ms);
    }
    if (ret == 0) {
      return comm_->set_mode(mode);
    } else {
      inno_log_warning("cannot get mode_before_change");
      return ret;
    }
  } else {
    if (mode_before_change) {
      *mode_before_change = INNO_LIDAR_MODE_WORK_NORMAL;
    }
    if (status_before_change) {
      *status_before_change = INNO_LIDAR_STATUS_NORMAL;
    }
    inno_log_info("file replay fake set mode");
    return 0;
  }
}

int InnoLidarClient::get_mode_status(enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode,
                                     enum InnoLidarStatus *status, uint64_t *in_transition_mode_ms) {
  if (is_live_lidar_()) {
    return comm_->get_mode_status(mode, pre_mode, status, in_transition_mode_ms);
  } else {
    if (mode) {
      *mode = INNO_LIDAR_MODE_WORK_NORMAL;
    }
    if (pre_mode) {
      *pre_mode = INNO_LIDAR_MODE_WORK_NORMAL;
    }
    if (status) {
      *status = INNO_LIDAR_STATUS_NORMAL;
    }
    inno_log_info("file replay return fake mode");
    return 0;
  }
}

int InnoLidarClient::get_anglehv_table(InnoDataPacket *inno_data) {
  int ret = -1;
  if (!anglehv_init_ && is_live_lidar_()) {
    ret = comm_->get_anglehv_table(reinterpret_cast<char *>(anglehv_table_), kInnoAngleHVTableMaxSize);

    if (ret == 0 && InnoPacketReader::verify_packet_crc32(reinterpret_cast<InnoCommonHeader *>(anglehv_table_))) {
      anglehv_init_ = true;
      memcpy(inno_data, anglehv_table_, anglehv_table_->common.size);
    }
  } else if (anglehv_init_) {
    memcpy(inno_data, anglehv_table_, anglehv_table_->common.size);
    ret = 0;
  } else if (is_live_lidar_()) {
    inno_log_error("get_anglehv_table failed");
    ret = -1;
  }
  return ret;
}

#define SERVER_ATTR_PREFIX "server_"
int InnoLidarClient::get_attribute(const char *attribute, double *value) {
  inno_log_verify(attribute && value, "%p, %p", attribute, value);
  // attributes start with server -- get from server
  if (InnoUtils::start_with(attribute, SERVER_ATTR_PREFIX)) {
    if (is_live_lidar_()) {
      // some server attributes should get with get_attribute_string
      // return -1 here to avoid send wrong request to server
      if (strcmp(attribute, "server_starting_log") == 0) {
        return -1;
      }
      return comm_->get_attribute(attribute + sizeof(SERVER_ATTR_PREFIX) - 1, value);
    } else {
      inno_log_info("play file mode not support set_%s", attribute);
      return -1;
    }
  }

  // attributes that need get from client
  if (strcmp(attribute, "is_live_direct_memory") == 0) {
    return -1;
  }

  // all others attribute -- send to server if it is live lidar
  if (is_live_lidar_()) {
    // backward compatibility -- if live lidar then send to server anyway
    return comm_->get_attribute(attribute, value);
  } else if (strcmp(attribute, "frame_rate") == 0) {
    *value = kDefaultFrameRate;
  } else if (strcmp(attribute, "reflectance_mode") == 0) {
    *value = INNO_REFLECTANCE_MODE_REFLECTIVITY;
  } else if (strcmp(attribute, "multiple_return") == 0 || strcmp(attribute, "return_mode") == 0) {
    *value = INNO_MULTIPLE_RETURN_MODE_SINGLE;
  } else if (strcmp(attribute, "enabled") == 0) {
    *value = 1;
  } else {
    return -1;
  }
  return 0;
}

int InnoLidarClient::get_attribute_string(const char *attribute, char *buf, size_t buf_size) {
  if (buf_size < 2) {
    inno_log_warning("buf_size too small %" PRI_SIZELU, buf_size);
    return -1;
  }
  // attributes start with server_ -- get from server
  if (InnoUtils::start_with(attribute, SERVER_ATTR_PREFIX)) {
    if (is_live_lidar_()) {
      return comm_->get_attribute(attribute + sizeof(SERVER_ATTR_PREFIX) - 1, buf, buf_size);
    } else {
      inno_log_info("play file mode not support get_%s", attribute);
      return 0;
    }
  }

  // attributes that need get from client
  if (strcmp(attribute, "is_live_direct_memory") == 0) {
    buf[0] = '0';
    buf[1] = '\0';
    return 0;
  } else if (strcmp(attribute, "is_pc_server") == 0) {
    size_t ret = snprintf(buf, buf_size, "client");
    if (ret >= buf_size) {
      buf[buf_size - 1] = 0;
    }
    return 0;
  } else if (strcmp(attribute, "is_live_lidar") == 0) {
    size_t ret = snprintf(buf, buf_size, is_live_lidar_() ? "yes" : "no");
    if (ret >= buf_size) {
      buf[buf_size - 1] = 0;
    }
    return 0;
  }
  // todo xxx buf maybe too small for some attributes such as inner_faults
  //  size_t ret = snprintf(buf, buf_size,
  //                        SET_ATTR_TIPS,
  //                        attribute, attribute, attribute);
  //  return ret < buf_size ? 0 : -1;
  // all others attribute -- send to server if it is live lidar
  if (is_live_lidar_()) {
    return comm_->get_attribute(attribute, buf, buf_size);
  } else {
    if (strcmp(attribute, "source_is_live_lidar") == 0) {
      buf[0] = '0';
      buf[1] = 0;
      return 0;
    }
    inno_log_info("play file mode not support get_%s", attribute);
    buf[0] = 0;
    return -1;
  }
}

int InnoLidarClient::set_faults_save_raw(const std::string& value_hex_str) {
#if defined (_QNX_) || defined (_WIN32)
  return 0;
#else
  faults_save_raw_ = value_hex_str;
  if (is_live_lidar_()) {
    inno_log_info("faults save raw: %s", value_hex_str.c_str());
    return comm_->set_faults_save_raw(value_hex_str);
  } else {
    inno_log_info("play file mode ignore setting faults save raw");
    return -1;
  }
#endif
}

int InnoLidarClient::attribute_force_xyz_pointcloud_(const char *buf) {
  if (strcmp(buf, "1") == 0) {
    force_xyz_pointcloud_ = true;
    return 0;
  } else if (strcmp(buf, "0") == 0) {
    force_xyz_pointcloud_ = false;
    return 0;
  } else {
    inno_log_error("invalid parameter: %s, must be 0 or 1", buf);
    return -1;
  }
}

int InnoLidarClient::attribute_force_vehicle_coordinate_(const char *buf) {
  if (strcmp(buf, "1") == 0) {
    InnoDataPacketUtils::set_vehicle_coordinate(1);
    return 0;
  } else if (strcmp(buf, "0") == 0) {
    InnoDataPacketUtils::set_vehicle_coordinate(0);
    return 0;
  } else {
    inno_log_error("invalid parameter: %s, must be 0 or 1", buf);
    return -1;
  }
}

int InnoLidarClient::attribute_save_raw_data_(const char *buf) {
#if defined (_QNX_) || defined (_WIN32)
  return 0;
#else
  std::unique_lock<std::mutex> lk(last_stage_mutex_);
  if (!last_stage_is_up_) {
    inno_log_error("%s is not working", name_);
    return -1;
  }
  // set udp port to listen
  int port = atoi(buf);
  if (port > 0) {
    // stop saving raw data thread if there already has a working thread
    // but the working thread are listening to a different port
    if (it_raw_recorder_ && raw_recorder_) {
      if (raw_recorder_->udp_port != port) {
        inno_log_info("stop listening to port %d", raw_recorder_->udp_port);
        it_raw_recorder_->shutdown();
        delete it_raw_recorder_;
        it_raw_recorder_ = nullptr;
        delete raw_recorder_;
        raw_recorder_ = nullptr;
        // stop server's raw4 sender
        if (comm_->set_server_udp_raw_port(-1) != 0) {
          inno_log_warning("stop raw data sender in lidar server failed");
          return -1;
        }
      } else {
        inno_log_info("already listen to port %d", port);
        return 0;
      }
    }
    // verify raw data save path has been set;
    // set up raw data saving thread
    inno_log_verify(!raw_recorder_ && !it_raw_recorder_, "raw_recorder and it_raw_recorder inited?");
    inno_log_verify(!raw_recoder_save_path_.empty(), "raw data save path not set!");
    // start server's raw4 sender
    if (comm_->set_server_udp_raw_port(port) != 0) {
      inno_log_warning("start raw data sender in lidar server failed");
      return -1;
    }
    // start raw recorder
    raw_recorder_ = new RawReceiver(this, port, raw_recoder_save_path_);
    it_raw_recorder_ = new InnoThread("raw_recorder", 0, 1, RawReceiver::start, raw_recorder_, 0, cpuset_);
    it_raw_recorder_->start();
    inno_log_info("start receiving raw data. port: %d", port);
    return 0;
  } else {
    inno_log_info("stop receiving raw data.");
    if (it_raw_recorder_) {
      it_raw_recorder_->shutdown();
      delete it_raw_recorder_;
      it_raw_recorder_ = nullptr;
    }
    if (raw_recorder_) {
      delete raw_recorder_;
      raw_recorder_ = nullptr;
    }
    // stop server's raw4 sender
    if (comm_->set_server_udp_raw_port(-1) != 0) {
      inno_log_warning("stop raw data sender in lidar server failed");
      return -1;
    }
    return 0;
  }
#endif
}

int InnoLidarClient::attribute_raw_data_save_path_(const char *buf) {
#if defined (_QNX_) || defined (_WIN32)
  return 0;
#else
  std::unique_lock<std::mutex> lk(last_stage_mutex_);
  if (!last_stage_is_up_) {
    inno_log_error("%s is not working", name_);
    return -1;
  }
  // set raw data save path
  // verify path is writable
  if (::access(buf, F_OK) != 0) {
#ifndef __MINGW64__
    inno_log_verify(::mkdir(buf, S_IRWXU) == 0, "create %s failed", buf);
#else
    inno_log_verify(::mkdir(buf) == 0, "create %s failed", buf);
#endif
  }
  inno_log_verify(::access(buf, R_OK) == 0, "path %s is not readable.", buf);
  inno_log_verify(::access(buf, W_OK) == 0, "path %s is not writable.", buf);
  raw_recoder_save_path_ = std::string(buf);
  if (raw_recorder_) {
    raw_recorder_->save_path = raw_recoder_save_path_;
  }
  inno_log_info("set raw data save path to: %s", buf);
  return 0;
#endif
}

int InnoLidarClient::attribute_faults_save_raw_(const char *buf) {
#if defined (_QNX_) || defined (_WIN32)
  return 0;
#else
  if (set_faults_save_raw(std::string(buf)) != 0) {
    inno_log_error("set faults_save_raw failed!");
    return -1;
  }
  return 0;
#endif
}

int InnoLidarClient::attribute_steering_wheel_angle_(const char *buf) {
  if (galvo_mirror_check_ == NULL) {
    galvo_mirror_check_ = new InnoGalvoMirrorCheck();
  }
  inno_log_verify(galvo_mirror_check_, "Invalid galvo_mirror_check_");
  char *end = NULL;
  if (strlen(buf) == 0) {
    inno_log_warning("Empty steering wheel angle!");
    galvo_mirror_check_->update_steering_wheel_angle(1001);
    galvo_mirror_check_->reset_check_times();
    return -1;
  }
  double v_steering_angle = strtod(buf, &end);
  if (strlen(end) > 0) {
    inno_log_warning("Invalid steering wheel angle: %s", buf);
    galvo_mirror_check_->update_steering_wheel_angle(1002);
    galvo_mirror_check_->reset_check_times();
    return -1;
  }
  galvo_mirror_check_->update_steering_wheel_angle(v_steering_angle);
  return 0;
}

int InnoLidarClient::attribute_vehicle_speed_(const char *buf) {
  if (galvo_mirror_check_ == NULL) {
    galvo_mirror_check_ = new InnoGalvoMirrorCheck();
  }
  inno_log_verify(galvo_mirror_check_, "Invalid galvo_mirror_check_");
  if (max_distance_check_ == NULL) {
    max_distance_check_ = new InnoMaxDistanceCheck();
  }
  inno_log_verify(max_distance_check_, "Invalid max_distance_check_");
  char *end = NULL;
  if (strlen(buf) == 0) {
    inno_log_warning("Empty vehicle speed!");
    galvo_mirror_check_->update_vehicle_speed(-1);
    galvo_mirror_check_->reset_check_times();
    max_distance_check_->update_vehicle_speed(-1);
    max_distance_check_->restart_check();
    return -1;
  }
  double v_speed = strtod(buf, &end);
  if (v_speed < 0 || strlen(end) > 0) {
    inno_log_warning("Invalid vehicle speed: %s", buf);
    galvo_mirror_check_->update_vehicle_speed(-1);
    galvo_mirror_check_->reset_check_times();
    max_distance_check_->update_vehicle_speed(-1);
    max_distance_check_->restart_check();
    return -1;
  }
  galvo_mirror_check_->update_vehicle_speed(v_speed);
  max_distance_check_->update_vehicle_speed(v_speed);
  return 0;
}

int InnoLidarClient::attribute_galvo_ext_ref_(const char *buf) {
  // Input galvo external reference,
  // the buf should be "x.xxxx,x.xxxx,x.xxxx,y.yyyy,y.yyyy,y.yyyy"
  // x.xxxx are rotation vector
  // y.yyyy are position offset
  if (galvo_mirror_check_ == NULL) {
    galvo_mirror_check_ = new InnoGalvoMirrorCheck();
  }
  inno_log_verify(galvo_mirror_check_, "Invalid galvo_ext_ref_");
  size_t buf_len = strlen(buf);
  if (buf_len == 0 || buf_len > 100) {
    inno_log_error("Invalid galvo ext ref: %s", buf);
    galvo_mirror_check_->reset_ref_idx();
    return -1;
  }
  double galvo_ext_ref[6] = {0};
  int got = sscanf(buf, "%lf,%lf,%lf,%lf,%lf,%lf", &galvo_ext_ref[0], &galvo_ext_ref[1], &galvo_ext_ref[2],
                   &galvo_ext_ref[3], &galvo_ext_ref[4], &galvo_ext_ref[5]);
  if (got != 6) {
    inno_log_error("Invalid galvo ext ref: %s, got = %d", buf, got);
    galvo_mirror_check_->reset_ref_idx();
    return -1;
  }
  galvo_mirror_check_->update_ext_ref(galvo_ext_ref[0], galvo_ext_ref[1], galvo_ext_ref[2], galvo_ext_ref[3],
                                      galvo_ext_ref[4], galvo_ext_ref[5]);
  return 0;
}

int InnoLidarClient::attribute_use_ring_id_(const char *buf) {
  if (strcmp(buf, "1") == 0) {
    if (!is_live_lidar_()) {
      inno_log_info("set use_ring_id=1 failed, not live lidar");
      return -1;
    }
    std::unique_lock<std::mutex> lk(ring_id_set_mutex_);
    if (!ring_id_converter_) {
      ring_id_converter_ = new RingIdConverter();
    }
    if (ring_id_converter_) {
        // ask server to send ring id table
      return comm_->set_attribute_string("ring_id_table", "1");
    } else {
      inno_log_error("create ring id coverter failed");
      return -1;
    }
  } else {
    inno_log_error("invalid parameter:%s, only support 1", buf);
    return -1;
  }
}

int InnoLidarClient::set_attribute_string(const char *attribute, const char *buf) {
  inno_log_verify(attribute && buf, "%p, %p", attribute, buf);

  // attributes start wih server_ -- send to server directly
  if (InnoUtils::start_with(attribute, SERVER_ATTR_PREFIX)) {
    if (is_live_lidar_()) {
      return comm_->set_attribute_string(attribute + sizeof(SERVER_ATTR_PREFIX) - 1, buf);
    } else {
      inno_log_info("play file mode ignore set_%s", attribute);
      return -1;
    }
  }

  // attributes that need set to client
  if (strcmp(attribute, "force_xyz_pointcloud") == 0) {
    return attribute_force_xyz_pointcloud_(buf);
  } else if (strcmp(attribute, "force_vehicle_coordinate") == 0) {
    return attribute_force_vehicle_coordinate_(buf);
  } else if (strcmp(attribute, "save_raw_data") == 0) {
    return attribute_save_raw_data_(buf);
  } else if (strcmp(attribute, "raw_data_save_path") == 0) {
    return attribute_raw_data_save_path_(buf);
  } else if (strcmp(attribute, "faults_save_raw") == 0) {
    return attribute_faults_save_raw_(buf);
  } else if (strcmp(attribute, "steering_wheel_angle") == 0) {
    return attribute_steering_wheel_angle_(buf);
  } else if (strcmp(attribute, "vehicle_speed") == 0) {
    return attribute_vehicle_speed_(buf);
  } else if (strcmp(attribute, "galvo_ext_ref") == 0) {
    return attribute_galvo_ext_ref_(buf);
  } else if (strcmp(attribute, "use_ring_id") == 0) {
    char model[2];
    int size = get_model(model, sizeof(model));
    if (strcmp(model, "k") == 0 || strcmp(model, "K") == 0 ||
        strcmp(model, "f") == 0) {  // k:falconK, K:falconK24,f:falconk2
      return attribute_use_ring_id_(buf);
    } else {
      inno_log_warning("Robin%s cannot support command: use_ring_id ", model);
      return 0;
    }
  }
  // all others attribute -- send to server
  if (is_live_lidar_()) {
    if (strcmp("time_sync_check", attribute) == 0) {
      time_sync_set_value = buf;
    }
    return comm_->set_attribute_string(attribute, buf);
  }

  return -1;
}

int InnoLidarClient::set_motion_compensation(double velocity[3], double a_velocity[3]) {
  return 1;
}

int InnoLidarClient::thread_setaffinity_np(size_t cpusetsize, const cpu_set_t *cpuset, int exclude_callback_thread) {
  inno_log_panic_if_not(cpusetsize > 0 && cpuset != NULL, "invalid calling parameter");
  cpusetsize_ = cpusetsize;
  cpuset_ = reinterpret_cast<cpu_set_t *>(malloc(cpusetsize));
  exclude_callback_thread_ = exclude_callback_thread;

  if (cpuset_ == NULL) {
    inno_log_error("cannot allocate memory for cpuset");
    return 2;
  }
  memcpy(cpuset_, cpuset, cpusetsize);
  return 0;
}

int InnoLidarClient::get_fw_state(InnoLidarState *state, int *error_code) {
  if (is_live_lidar_()) {
    char buffer[1024];
    int ret = comm_->get_fw_state(buffer, sizeof(buffer));
    if (ret == 0) {
      if (2 == sscanf(buffer, "%d,%d", reinterpret_cast<int *>(state), error_code)) {
        return 0;
      } else {
        return -1;
      }
    } else {
      return ret;
    }
  } else {
    // xxx todo: based on real streamin state
    *state = INNO_LIDAR_STATE_READY;
    *error_code = 0;
    inno_log_info("file replay fake get fw_state");
    return 0;
  }
}

int InnoLidarClient::get_fw_version(char *buffer, int buffer_len) {
  if (is_live_lidar_()) {
    return comm_->get_fw_version(buffer, buffer_len);
  } else {
    int ret = snprintf(buffer, buffer_len, "Inno Lidar Client");
    inno_log_info("file replay fake get fw_state");
    return ret < buffer_len ? 0 : -1;
  }
}

int InnoLidarClient::get_sn(char *buffer, int buffer_len) {
  if (is_live_lidar_()) {
    return comm_->get_sn(buffer, buffer_len);
  } else {
    int ret = snprintf(buffer, buffer_len, "c_file_replay");
    return ret < buffer_len ? 0 : -1;
  }
}

int InnoLidarClient::get_model(char *buffer, int buffer_len) {
  if (is_live_lidar_()) {
    return comm_->get_model(buffer, buffer_len);
  } else {
    int ret = snprintf(buffer, buffer_len, "i");
    return ret < buffer_len ? 0 : -1;
  }
}

int InnoLidarClient::start() {
  {
    std::unique_lock<std::mutex> lk(last_stage_mutex_);
    if (last_stage_is_up_) {
      inno_log_info("%s has started", name_);
      return 0;
    }
  }
  if (strncmp(inno_api_version(), INNO_SDK_VERSION_IN_HEADER, strlen(INNO_SDK_VERSION_IN_HEADER)) != 0) {
    do_message_callback_fmt(INNO_MESSAGE_LEVEL_INFO, INNO_MESSAGE_CODE_LIB_VERSION_MISMATCH,
                            "Header file inno_lidar_api.h and "
                            "libinnolidar.so file version mismatch: %s vs %s. ",
                            INNO_SDK_VERSION_IN_HEADER, inno_api_version());
  }
  inno_log_info("%s start", name_);
  inno_log_info("%s LIDAR SDK version is %s", name_, inno_api_version());
  inno_log_info("%s LIDAR SDK build tag is %s", name_, inno_api_build_tag());
  inno_log_info("%s LIDAR SDK build time is %s", name_, inno_api_build_time());
  inno_log_verify(!last_stage_is_up_, "%s forget to call stop before restart?", name_);
  if (before_read_start()) {
    inno_log_error("@@ %s before_read_start failed! @@", name_);
    return -1;
  }

  stage_deliver2_ = new StageClientDeliver2(this);
  inno_log_verify(stage_deliver2_, "stage_deliver2_");

  stage_deliver_ = new StageClientDeliver(this);
  inno_log_verify(stage_deliver_, "stage_deliver_");

  cp_deliver2_ = new ConsumerProducer("deliver2", 0, 1, StageClientDeliver2::process, stage_deliver2_, 200, 0, 100,
                                      cpusetsize_, exclude_callback_thread_ ? NULL : cpuset_);
  inno_log_verify(cp_deliver2_, "deliver2");

  bool can_drop = is_live_lidar_() || play_rate_ != 0 || play_rate_x_ != 0;

  cp_deliver_ = new ConsumerProducer("deliver", 0, 1, StageClientDeliver::process, stage_deliver_, 400,
                                     can_drop ? 400 : 0, 100, cpusetsize_, exclude_callback_thread_ ? NULL : cpuset_);
  inno_log_verify(cp_deliver_, "deliver");

  cp_recorder_ = new ConsumerProducer("recoder", 0, 1, StageClientRecorder::process, stage_deliver_, 200, 200, 0,
                                     cpusetsize_, exclude_callback_thread_ ? NULL : cpuset_);
  inno_log_verify(cp_deliver_, "recoder");

  int priority = 0;
#ifdef __linux__
  if (getuid() == 0) {
    // ruing as root
    priority = 1;
  }
#endif
  cp_read_ =
      new ConsumerProducer("read", priority, 1, StageClientRead::process, stage_read_, 2, 2, 0, cpusetsize_, cpuset_);
  inno_log_verify(cp_read_, "read");

  client_stats_ = new ClientStats(this);
  inno_log_verify(client_stats_, "client_stats");

  cp_deliver2_->start();
  {
    std::unique_lock<std::mutex> lk(last_stage_mutex_);
    last_stage_is_up_ = true;
  }
  cp_deliver_->start();
  cp_read_->start();
  cp_read_->add_job(NULL);
  cp_recorder_->start();
  inno_log_info("%s started", name_);
  return 0;
}

void InnoLidarClient::stop() {
  {
    std::unique_lock<std::mutex> lk(last_stage_mutex_);
    if (!last_stage_is_up_) {
      inno_log_info("%s is not start, don't need stop", name_);
      return;
    }
  }
  inno_log_info("%s stop stage_read_", name_);
  print_stats();

  stage_read_->stop();

  // wait until stage change to stopped
  inno_log_info("%s shutdown read", name_);
  cp_read_->shutdown();
  inno_log_info("%s shutdown deliver", name_);
  cp_deliver_->shutdown();

  inno_log_info("%s shutdown deliver", name_);
  cp_deliver2_->shutdown();

  inno_log_info("%s final cleanup", name_);
  stage_read_->final_cleanup();

  cp_recorder_->shutdown();
  std::unique_lock<std::mutex> lk(last_stage_mutex_);
  last_stage_is_up_ = false;
  #if !(defined (_QNX_) || defined (_WIN32))
  if (it_raw_recorder_ && !it_raw_recorder_->has_shutdown()) {
    inno_log_info("%s shutdown raw recorder", name_);
    it_raw_recorder_->shutdown();
  }
  #endif

  inno_log_info("%s final cleanup done", name_);

  delete (client_stats_);
  client_stats_ = NULL;

  delete cp_read_;
  cp_read_ = NULL;

  delete cp_deliver2_;
  cp_deliver2_ = NULL;

  delete cp_deliver_;
  cp_deliver_ = NULL;

  // DO NOT delete stage_read
  delete stage_deliver_;
  stage_deliver_ = NULL;

  delete stage_deliver2_;
  stage_deliver2_ = NULL;

  delete cp_recorder_;
  cp_recorder_ = NULL;
  inno_log_info("%s stopped", name_);
}

void InnoLidarClient::print_stats(void) {
  std::unique_lock<std::mutex> lk(last_stage_mutex_);
  if (last_stage_is_up_) {
    cp_read_->print_stats();
    cp_deliver_->print_stats();

    stage_read_->print_stats();
    stage_deliver_->print_stats();
    stage_deliver2_->print_stats();
  } else {
    inno_log_panic("last_stage is not up");
  }

  return;
}

int InnoLidarClient::before_read_start(void) {
  int ret;
  if (is_live_lidar_()) {
    char buffer[1024];

    std::vector<std::string> items = {"sw_version", "command_line", "fw_version",        "lidar_id",
                                      "debug",      "udp_ports_ip", "status_interval_ms"};
    for (auto item : items) {
      ret = comm_->get_attribute(item.c_str(), buffer, sizeof(buffer));
      if (ret) {
        inno_log_error("cannot get remote %s", item.c_str());
        return ret;
      } else {
        inno_log_info("%s remote %s: %s", get_name(), item.c_str(), buffer);
      }
    }

    // sn
    ret = comm_->get_sn(buffer, sizeof(buffer));
    if (ret) {
      inno_log_error("cannot get sn");
      return ret;
    } else {
      inno_log_info("%s serial number: %s", get_name(), buffer);
    }
    // get frame rate
    double frames_per_second_ = 0;
    int t = comm_->get_attribute("frame_rate", &frames_per_second_);
    if (t) {
      inno_log_error("cannot get frame_rate");
      return t;
    } else {
      inno_log_info("%s frame_rate: %f", get_name(), frames_per_second_);
    }
  }
  return 0;
}

void InnoLidarClient::add_config(Config *c) {
  inno_log_verify(c, "invalid config");
  config_manage_.add_config(c);
}

void InnoLidarClient::remove_config(Config *c) {
  inno_log_verify(c, "invalid config");
  config_manage_.remove_config(c);
}

void InnoLidarClient::stats_update_packet_bytes(enum ResourceStats::PacketType type, size_t packet, size_t byte) {
  inno_log_verify(client_stats_, "client_stats");
  client_stats_->update_packet_bytes(type, packet, byte);
}

/**
 * 1. Calculate the new normal vector after rotating
 *    the normal vector of fitting plane with rotation matrix.
 * 2. Calculate the included angle between new normal vector
 *    and the unit vector which normal to a horizontal plane.
 * 3. Compare the included angle with threshold.
 * 4. Calculate finally check result.
 */
InnoFrameCheckProcess InnoLidarClient::get_check_galvo_mirror_result(const InnoDataPacket &pkt,
                                                                     InnoGalvoCheckResult *check_result) {
  if (galvo_mirror_check_) {
    return galvo_mirror_check_->galvo_mirror_offset_check(pkt, check_result);
  } else {
    if (pkt.is_last_sub_frame) {
      inno_log_verify(check_result, "check_result nullptr");
      check_result->frame_check_code = INNO_GALVO_CHECK_FRAME_CODE_INVALID;
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_NO_EXT_REF;
      return InnoFrameCheckProcess::INNO_FRAME_CHECK_LAST_COMPELETED;
    } else {
      return InnoFrameCheckProcess::INNO_FRAME_CHECK_CONTINUE;
    }
  }
}

/**
 * Get max distance check result in client.
 * @return < 0: failed
 *           0: checking
 *           1: check completed
 */
int InnoLidarClient::get_check_max_distance_result(const char *buf, InnoMaxDistanceResult *check_result) {
  if (max_distance_check_) {
    return max_distance_check_->check_max_distance(buf, check_result);
  }
  return -1;
}

/**
 * Check whether the points on a tunnel
 * @param  pkt InnoDataPacket type should be INNO_ITEM_TYPE_XYZ_POINTCLOUD
 * @return < 0: failed
 *           0: success
 */
int InnoLidarClient::check_max_distance_on_tunnel(const InnoDataPacket &pkt) {
  if (max_distance_check_) {
    return max_distance_check_->check_data_on_tunnel(pkt);
  }
  return -1;
}

int InnoLidarClient::set_galvo_sensor_fault() {
  // Temporarily closed
  // if (comm_) {
  //   comm_->set_attribute_string("inner_fault",
  //     std::to_string(INNO_LIDAR_IN_FAULT_GALVO_SENSOR).c_str());
  //   return 0;
  // }
  inno_log_warning("Temporarily not support!");
  return -1;
}

int InnoLidarClient::clear_galvo_sensor_fault() {
  // Temporarily closed
  // if (comm_) {
  //   comm_->set_attribute_string("clear_inner_fault",
  //     std::to_string(INNO_LIDAR_IN_FAULT_GALVO_SENSOR).c_str());
  //   return 0;
  // }
  inno_log_warning("Temporarily not support!");
  return -1;
}

int InnoLidarClient::set_galvo_mirror_fault() {
  if (comm_) {
    return comm_->set_attribute_string("inner_fault", std::to_string(INNO_LIDAR_IN_FAULT_GALVO_MIRROR).c_str());
  }
  return -1;
}

int InnoLidarClient::clear_galvo_mirror_fault() {
  if (comm_) {
    return comm_->set_attribute_string("clear_inner_fault", std::to_string(INNO_LIDAR_IN_FAULT_GALVO_MIRROR).c_str());
  }
  return -1;
}

int InnoLidarClient::set_max_distance_fault() {
  if (comm_) {
    return comm_->set_attribute_string("inner_fault", std::to_string(INNO_LIDAR_IN_FAULT_MAX_DISTANCE).c_str());
  }
  return -1;
}

int InnoLidarClient::clear_max_distance_fault() {
  if (comm_) {
    return comm_->set_attribute_string("clear_inner_fault", std::to_string(INNO_LIDAR_IN_FAULT_MAX_DISTANCE).c_str());
  }
  return -1;
}

int InnoLidarClient::set_misalignment_time(int minutes) {
  if (comm_) {
    return comm_->set_attribute_string("set_misalignment", std::to_string(minutes).c_str());
  }
  return -1;
}

int InnoLidarClient::clear_misalignment_time() {
  if (comm_) {
    return comm_->set_attribute_string("heal_misalignment", "");
  }
  return -1;
}

void InnoLidarClient::update_ring_id_table(InnoDataPacket *pkt) {
  std::unique_lock<std::mutex> lk(ring_id_set_mutex_);
  if (ring_id_converter_) {
    ring_id_converter_->update_ring_id_table(pkt);
  }
}

RingIdConverterInterface *InnoLidarClient::get_ring_id_converter() {
  std::unique_lock<std::mutex> lk(ring_id_set_mutex_);
  if (ring_id_converter_) {
    return ring_id_converter_;
  } else {
    return nullptr;
  }
}

bool InnoLidarClient::source_ip_check_(uint32_t specified_ip, uint32_t packet_ip) {
  if (packet_ip != specified_ip) {
    static uint32_t packets_from_others = 0;
    packets_from_others++;
    if (packets_from_others % 8192 == 1) {
      unsigned char *p = reinterpret_cast<unsigned char *>(&packet_ip);
      unsigned char *p1 = reinterpret_cast<unsigned char *>(&specified_ip);
      inno_log_warning(
          "receive data packets from other lidar for %u times, "
          "current ip: "
          "%u"
          "."
          "%u"
          "."
          "%u"
          "."
          "%u, "
          "specified ip: "
          "%u"
          "."
          "%u"
          "."
          "%u"
          "."
          "%u",
          packets_from_others, p[0], p[1], p[2], p[3], p1[0], p1[1], p1[2], p1[3]);
    }
    return false;
  }
  return true;
}
uint32_t InnoLidarClient::get_specified_ip() {
  struct in_addr ip;
  int ret = NetManager::inno_inet_pton(ip_, &ip);
  inno_log_verify(ret == 1, "inno_inet_pton returns %d", ret);
  return ip.s_addr;
}

}  // namespace innovusion
