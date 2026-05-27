/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/stage_client_deliver.h"
#include <utility>

#include "sdk_client/lidar_client.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/consumer_producer.h"

namespace innovusion {

StageClientDeliver::StageClientDeliver(InnoLidarClient *l) {
  lidar_ = l;

  stats_total_jobs_ = 0;
  stats_dropped_jobs_ = 0;
  stats_data_jobs_ = 0;
  stats_message_jobs_ = 0;
  stats_status_jobs_ = 0;
  stats_points_ = 0;
  stats_2nd_return_points_ = 0;
  stats_frames_ = 0;
  // init galvo check
  has_set_galvo_mirror_fault_ = false;
  has_set_galvo_sensor_fault_ = false;
  check_galvo_times_ = 0;
  last_galvo_check_status_ = 0x00FF;
  last_galvo_check_angle_ = 0;
  xyz_data_packet_ = reinterpret_cast<InnoDataPacket*>(xyz_data_packet_buf_);
  lidar_->add_config(&config_base_);
  config_.copy_from_src(&config_base_);
}

StageClientDeliver::~StageClientDeliver(void) {
  lidar_->remove_config(&config_base_);
  // print frame loss rate at the end
  summary_package_.print_inno_data_packet_loss_rate(0);
}

int StageClientDeliver::process(void *in_job, void *ctx, bool prefer) {
  StageClientDeliver *s = reinterpret_cast<StageClientDeliver *>(ctx);
  return s->process_job_(reinterpret_cast<InnoCommonHeader *>(in_job), prefer);
}

int StageClientDeliver::process_job_(InnoCommonHeader *pkt, bool prefer) {
  stats_total_jobs_++;

  // only update stats for InnoDataPacket
  if (pkt->version.magic_number == kInnoMagicNumberDataPacket &&
      (reinterpret_cast<InnoDataPacket *>(pkt)->type != INNO_ITEM_TYPE_MESSAGE &&
       reinterpret_cast<InnoDataPacket *>(pkt)->type != INNO_ITEM_TYPE_MESSAGE_LOG &&
       reinterpret_cast<InnoDataPacket *>(pkt)->type != INNO_FALCON_RING_ID_TABLE)) {
    summary_package_.inno_data_packet_receive_stats(reinterpret_cast<const InnoDataPacket &>(*pkt));
  }

  if (!prefer) {
    stats_dropped_jobs_++;
    if (stats_dropped_jobs_ % 10 == 1) {
      //inno_log_warning("drop data in deliver stage.");
      print_stats();
    }
    if (stats_dropped_jobs_ % 100 == 1) {
      lidar_->cp_deliver_->print_stats();
    }
    lidar_->free_buffer_(pkt);
    return 0;
  }
  bool need_free_buffer = true;
  size_t n = pkt->size;
  inno_log_verify(n >= sizeof(InnoCommonHeader), "%" PRI_SIZELU " vs %" PRI_SIZELU, n, sizeof(InnoCommonHeader));
  if (pkt->version.magic_number == kInnoMagicNumberStatusPacket) {
    InnoStatusPacket *status_packet = reinterpret_cast<InnoStatusPacket *>(pkt);

    // xxx todo: handle different version
    if (n == sizeof(InnoStatusPacket)) {
      stats_status_jobs_++;
      if (lidar_->status_packet_callback_) {
        // inno_log_debug("status callback");
        if (InnoDataPacketUtils::check_status_packet(*status_packet, 0)) {
          status_packet->sensor_readings.galvo_status_client = last_galvo_check_status_;
          status_packet->sensor_readings.galvo_offset_angle_client = last_galvo_check_angle_;
          InnoPacketReader::set_packet_crc32(&status_packet->common);
        }
        lidar_->status_packet_callback_(lidar_->handle_, lidar_->callback_context_, status_packet);
      }
      lidar_->stats_update_packet_bytes(ResourceStats::PACKET_TYPE_STATUS, 1, n);
      // update ring_id table if mode changed
      if (status_packet->common.lidar_status == 0) {
        if (status_packet->common.lidar_mode != cur_lidar_mode_) {
          lidar_->update_ring_id_table(nullptr);
        }
        cur_lidar_mode_ = status_packet->common.lidar_mode;
      }
    } else {
      inno_log_warning("size mismatch %" PRI_SIZELU " %" PRI_SIZELU, n, sizeof(InnoStatusPacket));
    }
  } else if (pkt->version.magic_number == kInnoMagicNumberDataPacket) {
    InnoDataPacket *data_packet = reinterpret_cast<InnoDataPacket *>(pkt);
    if (n >= sizeof(InnoDataPacket)) {
      if (CHECK_SPHERE_POINTCLOUD_DATA(data_packet->type) || CHECK_XYZ_POINTCLOUD_DATA(data_packet->type)) {
        uint64_t start = InnoUtils::get_time_ns();
        uint64_t start_2 = 0;
        stats_data_jobs_++;
        uint64_t point_count_2nd_return = 0;

        // get anglehv_table
        if ((data_packet->type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
             data_packet->type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) && !lidar_->anglehv_init_) {
          char *table = new char[kInnoAngleHVTableMaxSize];
          lidar_->get_anglehv_table(reinterpret_cast<InnoDataPacket *>(table));
          delete[] table;
        }
        if (lidar_->data_packet_callback_) {
          // inno_log_debug("data callback %" PRI_SIZEU "", n);
          if (lidar_->force_xyz_pointcloud_ && CHECK_SPHERE_POINTCLOUD_DATA(data_packet->type)) {
            InnoDataPacket *xyz_pkt = NULL;
            size_t buf_size = 0;
            if (!config_.enable_stage_client_deliver2) {
              xyz_pkt = xyz_data_packet_;
              buf_size = kMaxXyzDataPacketBufSize;
            } else {
              xyz_pkt = reinterpret_cast<InnoDataPacket *>(lidar_->alloc_buffer_(kMaxInnoDataBufferSize));
              buf_size = kMaxInnoDataBufferSize;
            }
            if (InnoDataPacketUtils::convert_to_xyz_pointcloud(
                    *data_packet, xyz_pkt, buf_size, false,
                    ring_id_converter_ == NULL ? lidar_->get_ring_id_converter() : ring_id_converter_,
                    reinterpret_cast<char *>(
                        reinterpret_cast<InnoAngleHVTable *>(lidar_->anglehv_table_->payload)->table))) {
              if (lidar_->galvo_mirror_check_)
                check_lidar_fault_in_client_(*xyz_pkt);  // nio falconI/G check fault in client
              start_2 = InnoUtils::get_time_ns();
              convert_xyz_mean_ms_.add((start_2 - start) / 1000000.0);
              if (!config_.enable_stage_client_deliver2) {
                lidar_->data_packet_callback_(lidar_->handle_, lidar_->callback_context_, xyz_pkt);
              } else {
                lidar_->add_deliver2_job_(reinterpret_cast<void *>(xyz_pkt));
              }
              point_count_2nd_return = InnoDataPacketUtils::get_points_count_2nd_return(*xyz_pkt);
            } else {
              inno_log_error("cannot convert data_packet");
            }
          } else {
            start_2 = start;
            // nio falconI check fault in client
            if (lidar_->galvo_mirror_check_ && data_packet->type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
              check_lidar_fault_in_client_(*data_packet);
            } else if (data_packet->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
              if (lidar_->galvo_mirror_check_ && InnoDataPacketUtils::convert_to_xyz_pointcloud(
                    *data_packet,
                    xyz_data_packet_,
                     kMaxXyzDataPacketBufSize,
                    false)) {
                check_lidar_fault_in_client_(*xyz_data_packet_);
              }
            }

            if (!config_.enable_stage_client_deliver2) {
              lidar_->data_packet_callback_(lidar_->handle_, lidar_->callback_context_, data_packet);
            } else {
              need_free_buffer = false;
              lidar_->add_deliver2_job_(reinterpret_cast<void *>(data_packet));
            }
          }
          callback_mean_ms_.add((InnoUtils::get_time_ns() - start_2) / 1000000.0);
        }
        lidar_->stats_update_packet_bytes(ResourceStats::PACKET_TYPE_DATA, 1, n);
        int new_frame = 0;
        if (data_packet->is_last_sub_frame) {
          stats_frames_++;
          new_frame = 1;
          // print every 30 seconds
          if (stats_frames_ % (15 * 30) == 10) {
            print_stats();
            lidar_->cp_deliver_->print_stats();
          }
        }
        uint32_t point_count = InnoDataPacketUtils::get_points_count(*data_packet);
        lidar_->stats_update_packet_bytes(ResourceStats::PACKET_TYPE_POINT, new_frame, point_count);
        stats_points_ += point_count;
        stats_2nd_return_points_ += point_count_2nd_return;
      } else if (data_packet->type == INNO_ITEM_TYPE_MESSAGE || data_packet->type == INNO_ITEM_TYPE_MESSAGE_LOG) {
        char *p_addr = reinterpret_cast<char *>(const_cast<InnoDataPacket *>(data_packet));
        InnoMessage *messages = reinterpret_cast<InnoMessage *>(p_addr + sizeof(InnoDataPacket));
        InnoMessage *m = &messages[0];
        stats_message_jobs_++;
        if (n != data_packet->item_size + sizeof(InnoDataPacket) || data_packet->item_size <= sizeof(InnoMessage) ||
            data_packet->item_number != 1 || data_packet->item_size != m->size) {
          inno_log_warning("invalid message n=%" PRI_SIZELU
                           " item_size=%u item_num=%u m_size=%u, "
                           "%" PRI_SIZELU " %" PRI_SIZELU,
                           n, data_packet->item_size, data_packet->item_number, m->size, sizeof(InnoDataPacket),
                           sizeof(InnoMessage));
        } else {
          // zero-terminate message
          bool do_external_callback = true;
          m->content[m->size - sizeof(InnoMessage) - 1] = 0;
          if (m->code == INNO_MESSAGE_CODE_NEW_START) {
            #if !(defined (_QNX_) || defined (_WIN32))
            if (!lidar_->faults_save_raw_.empty()) {
              lidar_->set_faults_save_raw(lidar_->faults_save_raw_);
            }
            #endif
          } else if (m->code == INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT) {
            InnoMaxDistanceResult check_result{0};
            int ret = lidar_->get_check_max_distance_result(m->content, &check_result);
            if (ret == 1) {
              update_max_distance_check_result(check_result);
            }
            do_external_callback = false;
          } else if (m->code == INNO_MESSAGE_CODE_ROI_CHANGED) {
            // update ring-id table if using ring-id
            lidar_->update_ring_id_table(nullptr);
          }
          if (lidar_->message_callback_external_ && do_external_callback) {
            // inno_log_debug("message callback");
            lidar_->message_callback_external_(lidar_->handle_, lidar_->callback_context_, 1,
                                               InnoMessageLevel(m->level), InnoMessageCode(m->code), m->content);
          }
          lidar_->stats_update_packet_bytes(ResourceStats::PACKET_TYPE_MESSAGE, 1, n);
        }
      } else if (data_packet->type == INNO_FALCON_RING_ID_TABLE) {
        lidar_->update_ring_id_table(data_packet);
      } else {
        inno_log_warning("unknow data_packet type %u", data_packet->type);
      }
    } else {
      inno_log_warning("message size mismatch %" PRI_SIZELU " %" PRI_SIZELU, n, sizeof(InnoDataPacket));
    }
  } else {
    inno_log_warning("bad message, size=%" PRI_SIZELU, n);
  }
  if (need_free_buffer) {
    lidar_->free_buffer_(pkt);
  }
  return 0;
}

void StageClientDeliver::print_stats() const {
  // inno_log_info(
  //     "StageClientDeliever: "
  //     "convert_xyz mean/std/max/total=%.2fms/%.2f/%.2f/"
  //     "%" PRI_SIZEU
  //     " callback mean/std/max/total=%.2fms/%.2f/%.2f/%" PRI_SIZEU
  //     " total=%" PRI_SIZEU " total_dropped=%" PRI_SIZEU
  //     " data=%" PRI_SIZEU " message=%" PRI_SIZEU
  //     " status="
  //     "%" PRI_SIZEU " points=%" PRI_SIZEU " frames=%" PRI_SIZEU
  //     " points_2nd_return=%" PRI_SIZEU,
  //     convert_xyz_mean_ms_.mean(), convert_xyz_mean_ms_.std_dev(), convert_xyz_mean_ms_.max(),
  //     convert_xyz_mean_ms_.count(), callback_mean_ms_.mean(), callback_mean_ms_.std_dev(), callback_mean_ms_.max(),
  //     callback_mean_ms_.count(), stats_total_jobs_, stats_dropped_jobs_, stats_data_jobs_, stats_message_jobs_,
  //     stats_status_jobs_, stats_points_, stats_frames_, stats_2nd_return_points_);
}

const char *StageClientDeliver::get_name_(void) const {
  return lidar_->get_name();
}

/**
 * Start to check fault in client sdk.
 * @param pkt  input data packet
 * @param has_force_xyz  has converted to xyz point
 * @return int
 */
int StageClientDeliver::check_lidar_fault_in_client_(const InnoDataPacket &pkt) {
  if (pkt.type != INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
    return -1;
  }
  lidar_->check_max_distance_on_tunnel(pkt);
  if (pkt.scanner_direction == kInnoScanDirectionBottom2Top) {
    InnoGalvoCheckResult check_result{0};
    InnoFrameCheckProcess ret = lidar_->get_check_galvo_mirror_result(pkt, &check_result);
    if (ret == INNO_FRAME_CHECK_LAST_COMPELETED) {
      update_galvo_check_result_(check_result);
    }
  }
  return 0;
}

/**
 * Update the result of galvo checking
 * @param  check_result
 */
int StageClientDeliver::update_galvo_check_result_(const InnoGalvoCheckResult &check_result) {
  ++check_galvo_times_;
  /* save current status and angle */
  uint8_t save_speed = 0xFF;
  if (check_result.speed >= 0 && check_result.speed < 255) {
    save_speed = static_cast<uint8_t>(check_result.speed);
  }
  last_galvo_check_status_ = (check_result.fault_status << 12) | (check_result.frame_check_code << 8) | save_speed;
  last_galvo_check_angle_ = static_cast<uint16_t>(check_result.mean_deviated_angle * 100);
  char galvo_check_result_str[1024] = {0};
  int got = snprintf(galvo_check_result_str, sizeof(galvo_check_result_str),
                     "[Galvo Check] frame=%" PRI_SIZEU
                     ", code=%u,count=%u,"
                     "ext_ref:(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f),"
                     "fit_plane:(%.4f*x+%.4f*y+%.4f*z+%.4f=0),"
                     "rot_vector:(%.4f,%.4f,%.4f),"
                     "steering_angle=%.4f"
                     "speed=%.4f,speed_acc=%.4f,R2=%.4f,cos_angle=%.4f,"
                     "dev_angle=%.4f,mean_angle=%.4f,"
                     "angle_r2=%.4f,fault_status=%u,fault_times=%u,"
                     "valid_times=%d,us=%" PRI_SIZEU,
                     check_result.frame, static_cast<uint8_t>(check_result.frame_check_code), check_result.points_count,
                     check_result.input_vector.x, check_result.input_vector.y, check_result.input_vector.z,
                     check_result.input_delta.x, check_result.input_delta.y, check_result.input_delta.z,
                     check_result.ground_coeff.a, check_result.ground_coeff.b, check_result.ground_coeff.c,
                     check_result.ground_coeff.d, check_result.rotated_vector.x, check_result.rotated_vector.y,
                     check_result.rotated_vector.z, check_result.steering_wheel_angle, check_result.speed,
                     check_result.speed_acc, check_result.variance, check_result.cos_angle, check_result.deviated_angle,
                     check_result.mean_deviated_angle, check_result.deviated_angle_r2,
                     static_cast<uint8_t>(check_result.fault_status), check_result.fault_times,
                     check_result.valid_times, check_result.use_time_us);
  /* start to check fault */
  if (check_result.fault_status == INNO_GALVO_FAULT_STATUS_SET_GALVO_MIRROR ||
      check_result.fault_status == INNO_GALVO_FAULT_STATUS_SET_GALVO_SENSOR) {
    // Set galvo mirror fault angle > 3.0
    int ret = lidar_->set_galvo_mirror_fault();
    if (ret < 0) {
      inno_log_error("set_galvo_mirror_fault failed, ret = %d", ret);
    } else {
      has_set_galvo_mirror_fault_ = true;
    }
    if (check_result.fault_status == INNO_GALVO_FAULT_STATUS_SET_GALVO_SENSOR) {
      // Set galvo sensor fault
      // Temporarily closed
      // lidar_->set_galvo_sensor_fault();
      has_set_galvo_sensor_fault_ = true;
    }
    if (got < static_cast<int>(sizeof(galvo_check_result_str))) {
      lidar_->do_message_callback(INNO_MESSAGE_LEVEL_ERROR, INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT,
                                  galvo_check_result_str);
      inno_log_error("%s", galvo_check_result_str);
    }
    check_galvo_times_ = 1;
  } else if (check_result.fault_status == INNO_GALVO_FAULT_STATUS_NORMAL) {
    if (has_set_galvo_mirror_fault_) {
      int ret = lidar_->clear_galvo_mirror_fault();
      if (ret < 0) {
        inno_log_warning("clear_galvo_mirror_fault failed, ret = %d", ret);
      } else {
        has_set_galvo_mirror_fault_ = false;
      }
    }
    if (has_set_galvo_sensor_fault_) {
      // Temporarily closed
      // lidar_->clear_galvo_sensor_fault();
      has_set_galvo_sensor_fault_ = false;
    }
  }
  // [GALVO_CHECK] print every 20s
  if (check_galvo_times_ % 100 == 0 && got < static_cast<int>(sizeof(galvo_check_result_str))) {
    lidar_->do_message_callback(INNO_MESSAGE_LEVEL_INFO, INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT,
                                galvo_check_result_str);
    inno_log_info("%s", galvo_check_result_str);
  }
  /* Print galvo check status every 10s or use message callback*/
#ifdef _GALVO_CHECK_DEBUG_
  char callback_msg[1024];
  int ret = snprintf(callback_msg, sizeof(callback_msg),
                     "%lu\t%u\t%u\t"         // frame - points_count
                     "%lf\t%lf\t%lf\t"       // input_vector
                     "%lf\t%lf\t%lf\t%lf\t"  // ground_coeff
                     "%lf\t%lf\t%lf\t"       // rotated_vector
                     "%lf\t%lf\t%lf\t%lf\t"  // speed - cos_angle
                     "%lf\t%lf\t%lf\t"       // angle - angle_r2
                     "%u\t%u\t%u\t"          // fault_status - valid_times
                     "%lu",
                     check_result.frame, static_cast<uint8_t>(check_result.frame_check_code), check_result.points_count,
                     check_result.input_vector.x, check_result.input_vector.y, check_result.input_vector.z,
                     check_result.ground_coeff.a, check_result.ground_coeff.b, check_result.ground_coeff.c,
                     check_result.ground_coeff.d, check_result.rotated_vector.x, check_result.rotated_vector.y,
                     check_result.rotated_vector.z, check_result.speed, check_result.speed_acc, check_result.variance,
                     check_result.cos_angle, check_result.deviated_angle, check_result.mean_deviated_angle,
                     check_result.deviated_angle_r2, static_cast<uint8_t>(check_result.fault_status),
                     check_result.fault_times, check_result.valid_times, check_result.use_time_us);
  if (ret < static_cast<int>(sizeof(callback_msg))) {
    lidar_->do_message_callback(INNO_MESSAGE_LEVEL_DEBUG, INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT, callback_msg);
  }
#endif  // _GALVO_CHECK_DEBUG_
  return 0;
}

/**
 * Update the result of max distance checking
 * @param  check_result
 */
int StageClientDeliver::update_max_distance_check_result(const InnoMaxDistanceResult &check_result) {
  char max_distance_check_str[1024] = {0};
  int got = snprintf(max_distance_check_str, sizeof(max_distance_check_str),
                     "latest_frame=%" PRI_SIZEU
                     ","
                     "status=%u,v_count=%u,"
                     "mean_speed=%lf,mean_refl=%lf,"
                     "th_count=%d,th_normal_refl=%lf,th_fault_refl=%lf,"
                     "total_speed=%lf,total_refl=%lf,"
                     "th_min_speed=%lf,th_check_speed=%lf",
                     check_result.latest_frame_idx, static_cast<uint8_t>(check_result.fault_status),
                     check_result.valid_frame_count, check_result.mean_speed, check_result.mean_refl,
                     check_result.threshold_valid_count, check_result.threshold_normal_refl,
                     check_result.threshold_fault_refl, check_result.total_speed, check_result.total_refl,
                     check_result.threshold_min_speed, check_result.threshold_check_speed);
  if (got < static_cast<int>(sizeof(max_distance_check_str))) {
    inno_log_with_level(
        check_result.fault_status == INNO_MAX_DISTANCE_STATUS_SET_FAULT ? INNO_LOG_LEVEL_ERROR : INNO_LOG_LEVEL_INFO,
        "[Max-Distance]: %s", max_distance_check_str);
    lidar_->do_message_callback(INNO_MESSAGE_LEVEL_INFO, INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT,
                                max_distance_check_str);
  }
  if (check_result.fault_status == INNO_MAX_DISTANCE_STATUS_SET_FAULT) {
    int ret = lidar_->set_max_distance_fault();
    if (ret < 0) {
      inno_log_error("set_max_distance_fault failed, ret = %d", ret);
      return -1;
    }
  } else if (check_result.fault_status == INNO_MAX_DISTANCE_STATUS_NORMAL) {
    int ret = lidar_->clear_max_distance_fault();
    if (ret < 0) {
      inno_log_warning("clear_max_distance_fault failed, ret = %d", ret);
      return -1;
    }
    ret = lidar_->clear_misalignment_time();
    if (ret < 0) {
      inno_log_error("clear_misalignment_time failed, ret = %d", ret);
      return -1;
    }
  }
  // note: set_misalignment_time() must be called after
  // set_max_distance_fault(), clear_max_distance_fault() and
  // clear_misalignment_time()
  if (check_result.mean_speed >= InnoMaxDistanceCheck::kInnoMinHealSpeed) {
    int ret = lidar_->set_misalignment_time(3);  // 1800frames = 3minutes
    if (ret < 0) {
      inno_log_error("set_misalignment_time failed, ret = %d", ret);
      return -1;
    }
  }
  return 0;
}

StageClientDeliver2::StageClientDeliver2(InnoLidarClient *l) {
  lidar_ = l;
  stats_total_jobs_ = 0;
  stats_dropped_jobs_ = 0;
  stats_data_jobs_ = 0;
}

StageClientDeliver2::~StageClientDeliver2(void) {
}

int StageClientDeliver2::process(void *in_job, void *ctx, bool prefer) {
  StageClientDeliver2 *s = reinterpret_cast<StageClientDeliver2 *>(ctx);
  return s->process_job_(reinterpret_cast<InnoCommonHeader *>(in_job), prefer);
}

int StageClientDeliver2::process_job_(InnoCommonHeader *pkt, bool prefer) {
  stats_total_jobs_++;
  if (!prefer) {
    stats_dropped_jobs_++;
    if (stats_dropped_jobs_ % 10 == 1) {
      inno_log_warning("drop data in deliver2 stage.");
      print_stats();
    }
    if (stats_dropped_jobs_ % 100 == 1) {
      lidar_->cp_deliver2_->print_stats();
    }
    lidar_->free_buffer_(pkt);
    return 0;
  }
  size_t n = pkt->size;
  inno_log_verify(n >= sizeof(InnoCommonHeader), "%" PRI_SIZELU " vs %" PRI_SIZELU, n, sizeof(InnoCommonHeader));
  if (pkt->version.magic_number == kInnoMagicNumberDataPacket) {
    InnoDataPacket *data_packet = reinterpret_cast<InnoDataPacket *>(pkt);
    if (n >= sizeof(InnoDataPacket)) {
      if (CHECK_SPHERE_POINTCLOUD_DATA(data_packet->type) || CHECK_XYZ_POINTCLOUD_DATA(data_packet->type)) {
        if (lidar_->data_packet_callback_) {
          lidar_->data_packet_callback_(lidar_->handle_, lidar_->callback_context_, data_packet);
          lidar_->free_buffer_(pkt);
        } else {
          inno_log_panic("invalid! lidar_->data_packet_callback_ is NULL");
        }
      } else {
        inno_log_panic("invalid data_packet type %d", data_packet->type);
      }
    } else {
      inno_log_panic("invalid data_packet size %" PRI_SIZELU, n);
    }
  }
  return 0;
}

void StageClientDeliver2::print_stats() const {
//   inno_log_info(
//       "StageClientDeliever2: "
//       "total=%" PRI_SIZEU " total_dropped=%" PRI_SIZEU "data=%" PRI_SIZEU,
//       stats_total_jobs_, stats_dropped_jobs_, stats_data_jobs_);
}

}  // namespace innovusion
