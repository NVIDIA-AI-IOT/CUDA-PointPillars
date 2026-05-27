/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include "sdk_client/lidar_fault_check.h"
#include <float.h>

#include <random>
namespace innovusion {

/******************************************************************/
/*********************** Galvo Mirror Check ***********************/
/******************************************************************/

InnoGalvoMirrorCheck::InnoGalvoMirrorCheck() {
  init_();
}

InnoGalvoMirrorCheck::~InnoGalvoMirrorCheck() {
  check_galvo_ground_points_.clear();
}

void InnoGalvoMirrorCheck::init_() {
  input_vector_ = {0, 0, 0};
  input_delta_ = {0, 0, 0};
  ref_idx_ = 0;
  check_times_ = 0;
  check_speed_times_ = 0;
  vehicle_speed_ = 0.0;
  steering_wheel_angle_ = 0.0;
  speed_update_ts_ms_ = 0;
  steering_update_ts_ms_ = 0;
  frame_start_speed_ = 0.0;
  frame_start_speed_ts_ms_ = 0;
  frame_start_steering_angle_ = 0.0;
  frame_start_steering_angle_ts_ms_ = 0;
  last_vehicle_speed_ = 0.0;
  last_frame_idx_ = 0;
  need_ground_points_ = false;
  memset(check_angles_, 0, sizeof(check_angles_));
  memset(check_speeds_, 0, sizeof(check_speeds_));
  memset(total_speeds_, 0, sizeof(total_speeds_));
  check_galvo_ground_points_.reserve(kInnoGroundPointsReservedSize);
  check_galvo_ground_points_.clear();
}

/*
 * Reset set external reference index
 */
void InnoGalvoMirrorCheck::reset_ref_idx() {
  ref_idx_ = 0;
}

/*
 * @return  true: external reference has set
            false: not set
 */
bool InnoGalvoMirrorCheck::ref_has_set_() {
    return ref_idx_ > 0;
}

/**
 * Update external reference and convert the rotation vector to matrix
 * @param  vec_x Rotate around the X axis
 * @param  vec_y Rotate around the Y axis
 * @param  vec_z Rotate around the Z axis
 * @param  delta_x Reserved
 * @param  delta_y Reserved
 * @param  delta_z Reserved
 */
void InnoGalvoMirrorCheck::update_ext_ref(
            const double vec_x, const double vec_y, const double vec_z,
            const double delta_x, const double delta_y, const double delta_z) {
  input_vector_ = {vec_x, vec_y, vec_z};
  input_delta_ = {delta_x, delta_y, delta_z};
  ++ref_idx_;
  inno_log_info("Galvo ext reference, rotation vector (%lf, %lf, %lf), " \
                  "delta position (%lf, %lf, %lf), " \
                  "idx = %" PRI_SIZEU,
                  input_vector_.x, input_vector_.y, input_vector_.z,
                  input_delta_.x, input_delta_.y, input_delta_.z,
                  ref_idx_);
  // should be convert input rotation vector to rotation matrix
  if (convert_rotation_vector_to_matrix_() < 0) {
    reset_ref_idx();
    inno_log_warning("Convert to rotation matrix failed!");
  } else {
    inno_log_info("Galvo rotation matrix:" \
                "(%lf, %lf, %lf)," \
                "(%lf, %lf, %lf)," \
                "(%lf, %lf, %lf)",
                rotation_matrix_[0], rotation_matrix_[1], rotation_matrix_[2],
                rotation_matrix_[3], rotation_matrix_[4], rotation_matrix_[5],
                rotation_matrix_[6], rotation_matrix_[7], rotation_matrix_[8]);
  }
}

/**
 * Update speed from vehicle
 * @param  speed unit: km/h
 */
void InnoGalvoMirrorCheck::update_vehicle_speed(const double speed) {
  vehicle_speed_ = speed;
  speed_update_ts_ms_ = InnoUtils::get_time_ms();
}

/**
 * Update steering wheel angle
 * @param  angle unit?
 */
void InnoGalvoMirrorCheck::update_steering_wheel_angle(const double angle) {
  steering_wheel_angle_ = angle;
  steering_update_ts_ms_ = InnoUtils::get_time_ms();
}

/*
 * Reset check speed times
 */
void InnoGalvoMirrorCheck::reset_check_times() {
  check_times_ = 0;
}

/**
 * Galvo mirror offset check entrance
 * @param  pkt  input data packet which type
 *              should be INNO_ITEM_TYPE_XYZ_POINTCLOUD
 * @param  check_result  output galvo check result
 * @return InnoFrameCheckProcess
 */
InnoFrameCheckProcess InnoGalvoMirrorCheck::galvo_mirror_offset_check(const InnoDataPacket &pkt,
                                                                      InnoGalvoCheckResult *check_result) {
  if (pkt.scanner_direction != kInnoScanDirectionBottom2Top) {
    return INNO_FRAME_CHECK_CONTINUE;
  }
  // Check the external reference.
  // If no external reference return immediately.
  if (!ref_has_set_()) {
    if (last_frame_idx_ != pkt.idx) {
      last_frame_idx_ = pkt.idx;
      inno_log_verify(check_result, "Invalid check_result!");
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
      check_result->frame_check_code = INNO_GALVO_CHECK_FRAME_CODE_INVALID_EXT_REF;
      reset_check_times();
      return INNO_FRAME_CHECK_LAST_COMPELETED;
    }
    return INNO_FRAME_CHECK_CONTINUE;
  }
  // Init check process flag
  InnoFrameCheckProcess check_process = INNO_FRAME_CHECK_CONTINUE;
  if (last_frame_idx_ != pkt.idx) {
    // Check last frame and ready to new frame
    uint64_t idx_diff = pkt.idx - last_frame_idx_;
    check_process = INNO_FRAME_CHECK_LAST_COMPELETED;
    check_result->frame = last_frame_idx_;
    last_frame_idx_ = pkt.idx;
    check_result->speed = frame_start_speed_;
    check_result->speed_update_ts_ms = frame_start_speed_ts_ms_;
    check_result->steering_wheel_angle = frame_start_steering_angle_;
    check_result->steering_update_ts_ms = frame_start_steering_angle_ts_ms_;
    bool need_to_check_galvo_mirror = true;
    // Check the update time of speed.
    // If speed update timeout return immediately.
    uint64_t curr_timestamp_ms = InnoUtils::get_time_ms();
    uint64_t time_diff = curr_timestamp_ms - check_result->speed_update_ts_ms;
    uint64_t time_diff1 = curr_timestamp_ms - check_result->steering_update_ts_ms;
    if (time_diff > kInnoUpdateSpeedMaxDelayTime || time_diff1 > kInnoUpdateSteeringMaxDelayTime) {
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
      check_result->frame_check_code = INNO_GALVO_CHECK_FRAME_CODE_EXT_TIMEOUT;
      need_to_check_galvo_mirror = false;
      reset_check_times();
    }
    // Check the diff of frame idx.
    // No need return, just continue to check current frame.
    if (need_to_check_galvo_mirror && idx_diff > kInnoFrameIdxMaxDiff) {
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
      check_result->frame_check_code = INNO_GALVO_CHECK_FRAME_CODE_LAST_FRAME_IDX_OUT;
      need_to_check_galvo_mirror = false;
      reset_check_times();
    }
    // Start to check galvo mirror offset for last frame
    if (need_to_check_galvo_mirror) {
      check_result->input_vector = input_vector_;
      check_result->input_delta = input_delta_;
      check_result->points_count = check_galvo_ground_points_.size();
      uint64_t start_time = InnoUtils::get_time_us();
      cal_and_get_galvo_check_result_(check_galvo_ground_points_, check_result);
      check_result->use_time_us = InnoUtils::get_time_us() - start_time;
    }
    // NEW FRAME
    frame_start_speed_ = vehicle_speed_;
    frame_start_speed_ts_ms_ = speed_update_ts_ms_;
    frame_start_steering_angle_ = steering_wheel_angle_;
    frame_start_steering_angle_ts_ms_ = steering_update_ts_ms_;
    time_diff = InnoUtils::get_time_ms() - speed_update_ts_ms_;
    time_diff1 = InnoUtils::get_time_ms() - steering_update_ts_ms_;
    if (frame_start_speed_ >= kInnoMinGalvoCheckSpeed &&
        frame_start_steering_angle_ <= kInnoGalvoCheckMaxSteeringAngle && time_diff < kInnoUpdateSpeedMaxDelayTime &&
        time_diff1 < kInnoUpdateSteeringMaxDelayTime) {
      need_ground_points_ = true;
    } else {
      need_ground_points_ = false;
    }
    check_galvo_ground_points_.clear();
  }
  // start to collect ground points
  if (need_ground_points_) {
    collect_ground_points_(pkt);
  }
  return check_process;
}

/**
 * Start to collect ground points
 * The type of InnoDataPacket should be INNO_ITEM_TYPE_XYZ_POINTCLOUD
 */
int InnoGalvoMirrorCheck::collect_ground_points_(
                                  const InnoDataPacket &pkt) {
  // If ground points too much skip!
  if (check_galvo_ground_points_.size() > kInnoGroundValidMaxCount) {
    return -1;
  }
  return collect_ground_points_process_(pkt);
}

/**
 * Collect ground points process
 */
int InnoGalvoMirrorCheck::collect_ground_points_process_(const InnoDataPacket &pkt) {
  bool available_to_check = false;
  const uint32_t point_size = pkt.item_number;
  char *p_addr = reinterpret_cast<char *>(const_cast<InnoDataPacket *>(&pkt));
  InnoXyzPoint *xyz_points = reinterpret_cast<InnoXyzPoint *>(p_addr + sizeof(InnoDataPacket));
  if (point_size > 10) {
    const InnoXyzPoint &point0 = xyz_points[point_size / 3];
    const InnoXyzPoint &point1 = xyz_points[point_size / 2];
    const InnoXyzPoint &point2 = xyz_points[point_size / 3 * 2];
    // (check) direction 1: bottom -> top, only check scan_id range 0 - 5
    // (not check) direction 0: top -> bottom
    if (point0.scan_id < kInnoCheckBottomMaxScanId1 || point1.scan_id < kInnoCheckBottomMaxScanId1 ||
        point2.scan_id < kInnoCheckBottomMaxScanId1) {
      available_to_check = true;
    }
    if (available_to_check) {
      uint32_t i = 0;
      for (i = 0; i < point_size; i++) {
        const InnoXyzPoint &point = xyz_points[i];
        // 1. Check the road within 3 - 25m in front of the vehicle
        // 2. point.x: under the vehicle, point.x is perpendicular to the ground
        // 3. point.y: left and right sides of the vehicle
        // 4. scan id < 2, default add to ground points
        if (point.y < kInnoCheckLeftEdge || point.y > kInnoCheckRightEdge) {
          continue;
        }
        if (point.scan_id < kInnoCheckBottomScanId1) {
          check_galvo_ground_points_.push_back({point.x, point.y, point.z, point.ts_10us});
        } else {
          if (point.z < kInnoCheckMinDistance || point.z > kInnoCheckMaxDistance) {
            continue;
          }
          if (point.scan_id < kInnoCheckBottomMaxScanId1 && (point.scan_idx % kInnoGroundDivideScanIdxSize == 0)) {
            check_galvo_ground_points_.push_back({point.x, point.y, point.z, point.ts_10us});
          }
        }
      }
    }
  }
  return 0;
}

/**
 * Fit the ground plane by input points, and calculate the included angle
 * of the vector after rotated and z=0 plane.
 * Then check the galvo mirror offset.
 * @param  ground_points input the points of ground
 * @param  check_result  output the check result of this frame
 * @return int < 0: failed  0: success
 */
int InnoGalvoMirrorCheck::cal_and_get_galvo_check_result_(
                                      const InnoCheckPoints &ground_points,
                                      InnoGalvoCheckResult *check_result) {
  uint8_t check_angle_idx
          = static_cast<uint8_t>(check_times_ % kInnoGalvoCheckMaxTimes);
  uint8_t check_time_idx
          = static_cast<uint8_t>(check_speed_times_ % kInnoSpeedCheckMaxTimes);
  double curr_speed = check_result->speed;
  double curr_str_whl_angle_abs = fabs(check_result->steering_wheel_angle);
  uint32_t point_size = check_result->points_count;
  uint32_t i = 0;
  check_speeds_[check_time_idx] = fabs(curr_speed - last_vehicle_speed_);
  last_vehicle_speed_ = curr_speed;
  ++check_speed_times_;
  /* skip this frame if steering wheel angle more than threshold */
  if (curr_str_whl_angle_abs > kInnoGalvoCheckMaxSteeringAngle) {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_INVALID_STEERING_ANGLE;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
    if (curr_str_whl_angle_abs > kInnoGalvoCheckSteeringResetAngle) {
      /* the steering wheel angle too big to reset check times */
      reset_check_times();
    } else  {
      check_angles_[check_angle_idx] = 0;
      total_speeds_[check_angle_idx] = curr_speed;
      ++check_times_;
    }
    return -1;
  }
  /* skip this frame if speed not enough */
  if (curr_speed < kInnoMinGalvoCheckSpeed) {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_INVALID_SPEED;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
    reset_check_times();
    return -1;
  }
  /******** Check points threshold ********/
  if (point_size < kInnoGroundValidMinCount) {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_FIT_COUNT_NOT_ENOUGH;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
    check_angles_[check_angle_idx] = 0;
    total_speeds_[check_angle_idx] = curr_speed;
    ++check_times_;
    return -1;
  }
  /******** Check speed acc ********/
  if (check_speed_times_ > kInnoSpeedCheckMaxTimes) {
    double speed_acc = 0;
    for (i = 0; i < kInnoSpeedCheckMaxTimes; i++) {
      speed_acc += check_speeds_[i];
    }
    check_result->speed_acc = speed_acc / kInnoSpeedCheckMaxTimes;
    if (check_result->speed_acc > 0.8) {
      check_result->frame_check_code
                      = INNO_GALVO_CHECK_FRAME_CODE_SPEED_CHANGE_QUICKLY;
      check_result->fault_status
                            = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
      check_angles_[check_angle_idx] = 0;
      total_speeds_[check_angle_idx] = curr_speed;
      ++check_times_;
      return -1;
    }
  } else {
    check_result->mean_deviated_angle = 0.0;
    check_result->deviated_angle_r2 = 0.0;
    check_result->frame_check_code
                  = INNO_GALVO_CHECK_FRAME_CODE_CHECK_COUNT_NOT_ENOUGH;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_CHECKING;
    check_result->use_time_us = 0;
    return -1;
  }
  /******** Fit plane ********/
  double vector_module = 0.0;
  double temp_value = 0.0;
  InnoGroundCoeff coeff_plane = {0, 0, 0, 0};
  InnoVector3D rotated_vector = {0, 0, 0};
  // 1s = 100000 * 10us
  double distance_per_10us = curr_speed / 3.6 / 100000;
  int fit_ret = fitting_plane_ransac_least_(ground_points,
                                           distance_per_10us,
                                           &coeff_plane);
  if (fit_ret < 0) {
    // inno_log_info("fit invalid! ret=%d", fit_ret);
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_FIT_INVALID;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
    check_angles_[check_angle_idx] = 0;
    total_speeds_[check_angle_idx] = curr_speed;
    ++check_times_;
    return -1;
  }
  /******** rotated normal vector of ground plane ********/
  rotated_vector.x = rotation_matrix_[0] * coeff_plane.a
                   + rotation_matrix_[1] * coeff_plane.b
                   + rotation_matrix_[2] * coeff_plane.c;
  rotated_vector.y = rotation_matrix_[3] * coeff_plane.a
                   + rotation_matrix_[4] * coeff_plane.b
                   + rotation_matrix_[5] * coeff_plane.c;
  rotated_vector.z = rotation_matrix_[6] * coeff_plane.a
                   + rotation_matrix_[7] * coeff_plane.b
                   + rotation_matrix_[8] * coeff_plane.c;
  check_result->variance
              = get_fitting_plane_variance_(ground_points, coeff_plane);
  check_result->ground_coeff = coeff_plane;
  check_result->rotated_vector = rotated_vector;
  vector_module = calculate_vector_module_(rotated_vector);
  /******** check valid angle ********/
  if (vector_module > 0) {
    temp_value = vehicle_plane_vector.x * rotated_vector.x
               + vehicle_plane_vector.y * rotated_vector.y
               + vehicle_plane_vector.z * rotated_vector.z;
    check_result->cos_angle = fabs(temp_value) / vector_module;
    check_result->deviated_angle
                    = acos(check_result->cos_angle) / M_PI * 180.0;
    if (check_result->deviated_angle > kInnoMaxDeviatedAngle
        || check_result->deviated_angle <= 0) {
      check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_INVALID_ANGLE;
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
      check_angles_[check_angle_idx] = 0;
      total_speeds_[check_angle_idx] = curr_speed;
      ++check_times_;
      return -1;
    }
  } else {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_FIT_INVALID;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_INVALID_CHECK_SKIP;
    check_angles_[check_angle_idx] = 0;
    total_speeds_[check_angle_idx] = curr_speed;
    ++check_times_;
    return -1;
  }
  /******** update check code of this frame ********/
  if (check_result->deviated_angle > kInnoGalvoMinOffsetAngle) {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_DEVIATED;
  } else {
    check_result->frame_check_code
                        = INNO_GALVO_CHECK_FRAME_CODE_NORMAL;
  }
  /* here start to update current frame status and check angles array */
  check_angles_[check_angle_idx] = check_result->deviated_angle;
  total_speeds_[check_angle_idx] = curr_speed;
  ++check_times_;
  if (check_times_ > kInnoGalvoCheckMaxTimesIndex) {
    uint8_t valid_check_count = 0;
    uint8_t fault_occur_times = 0;
    double deviated_angle_r2 = 0.0;
    double mean_valid_angle = 0.0;
    double average_speed = 0.0;
    /* get fault occurs times and valid angle count */
    for (i = 0; i < kInnoGalvoCheckMaxTimes; i++) {
      if (check_angles_[i] > kInnoGalvoMinOffsetAngle) {
        ++fault_occur_times;
      }
      if (check_angles_[i] > 0) {
        ++valid_check_count;
        mean_valid_angle += check_angles_[i];
      }
      average_speed += total_speeds_[i];
    }
    check_result->fault_times = fault_occur_times;
    check_result->valid_times = valid_check_count;

    if (valid_check_count < kInnoGalvoMinCheckTimes) {
      check_result->mean_deviated_angle = 0;
      check_result->deviated_angle_r2 = 0;
      check_result->fault_status
              = INNO_GALVO_FAULT_STATUS_VALID_COUNT_NOT_ENOUGH;
      return 0;
    }

    mean_valid_angle = mean_valid_angle / valid_check_count;
    if (mean_valid_angle <= kInnoGalvoMinOffsetAngle) {
      check_result->mean_deviated_angle = mean_valid_angle;
      check_result->deviated_angle_r2 = 0;
      check_result->fault_status = INNO_GALVO_FAULT_STATUS_NORMAL;
      return 0;
    }
    average_speed = average_speed / valid_check_count;
    if (average_speed < kInnoMinGalvoVehicleSpeed) {
      check_result->mean_deviated_angle = mean_valid_angle;
      check_result->deviated_angle_r2 = 0;
      check_result->fault_status
                 = INNO_GALVO_FAULT_STATUS_SPEED_NOT_ENOUGH;
      return 0;
    }
    /* calculate mean angle and angle r2 */
    if (valid_check_count >= kInnoGalvoMinCheckTimes2) {
      for (i = 0; i < kInnoGalvoCheckMaxTimes; i++) {
        if (check_angles_[i] > 0) {
          deviated_angle_r2 += (check_angles_[i] - mean_valid_angle)
                            * (check_angles_[i] - mean_valid_angle);
        }
      }
      deviated_angle_r2 = deviated_angle_r2 / valid_check_count;
      if (deviated_angle_r2 > kInnoGalvoMaxAngleR2Rest) {
        mean_valid_angle = 0.0;
        check_result->fault_status
                  = INNO_GALVO_FAULT_STATUS_VALID_ANGLE_R2_TOO_BIG;
        reset_check_times();
      } else if (deviated_angle_r2 > kInnoGalvoMaxAngleR2) {
        mean_valid_angle = 0.0;
        check_result->fault_status
                  = INNO_GALVO_FAULT_STATUS_VALID_ANGLE_R2_TOO_BIG;
      } else {
        if (mean_valid_angle > kInnoGalvoMinOffsetAngle2) {
          check_result->fault_status = INNO_GALVO_FAULT_STATUS_SET_GALVO_SENSOR;
        } else if (mean_valid_angle > kInnoGalvoMinOffsetAngle) {
          check_result->fault_status = INNO_GALVO_FAULT_STATUS_SET_GALVO_MIRROR;
        } else {
          /******** Clear Fault ********/
          check_result->fault_status = INNO_GALVO_FAULT_STATUS_NORMAL;
        }
      }
    } else {
      /* valid angle count not enough */
      deviated_angle_r2 = 0.0;
      mean_valid_angle = 0.0;
      check_result->fault_status
                 = INNO_GALVO_FAULT_STATUS_VALID_COUNT_NOT_ENOUGH;
    }
    /* update check result information */
    check_result->mean_deviated_angle = mean_valid_angle;
    check_result->deviated_angle_r2 = deviated_angle_r2;
  } else {
    /******** In Checking ********/
    check_result->mean_deviated_angle = 0.0;
    check_result->deviated_angle_r2 = 0.0;
    check_result->fault_status = INNO_GALVO_FAULT_STATUS_CHECKING;
  }
  return 0;
}

inline double InnoGalvoMirrorCheck::calculate_vector_module_(
                                    const InnoVector3D &vector_3d) {
    return sqrt(vector_3d.x * vector_3d.x
              + vector_3d.y * vector_3d.y
              + vector_3d.z * vector_3d.z);
}

/**
 * Get the variance of fitting ground
 * @param  ground_points
 * @param  coeff_plane
 * @return double  variance of ground points
 */
inline double InnoGalvoMirrorCheck::get_fitting_plane_variance_(
                               const InnoCheckPoints &ground_points,
                               const InnoGroundCoeff &coeff_plane) {
    const size_t points_size = ground_points.size();
    if (points_size > 0) {
      double temp_value = 0.0;
      double plane_variance = 0.0;
      for (size_t i = 0; i < points_size; i++) {
        InnoPointXYZT point = ground_points[i];
        temp_value = point.x * coeff_plane.a + point.y * coeff_plane.b
                              + coeff_plane.c * point.z + coeff_plane.d;
        plane_variance += temp_value * temp_value;
      }
      return plane_variance / points_size;
    }
    return -1.0;
}

/**
 * Convert rotation vector to matrix
 * Rodrigues's Formula:
 * R = cos(θ)*MI + (1 - cos(θ))*v*vT + sin(θ)*vX
 *
 *      | 1 0 0 |                     |x|
 * MI = | 0 1 0 |  v * vT = |x y z| * |y|
 *      | 0 0 1 |                     |z|
 *
 *      | 0 -z  y|
 * vX = | z  0 -x| is antisymmetric matrix
 *      |-y  x  0|
 */
int InnoGalvoMirrorCheck::convert_rotation_vector_to_matrix_() {
  double theta = sqrt(input_vector_.x * input_vector_.x
                    + input_vector_.y * input_vector_.y
                    + input_vector_.z * input_vector_.z);
  if (theta < INNO_EPSLION_VALUE) {
    return -1;
  }
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);
  double cos_theta1 = 1.0 - cos_theta;
  double itheta = theta ? 1.0 / theta : 0.0;
  double vec_x = input_vector_.x * itheta;
  double vec_y = input_vector_.y * itheta;
  double vec_z = input_vector_.z * itheta;
  double matrix_vvT[9] = {0};
  double matrix_vX[9] = {0};
  double matrix_unit[9] = {0};
  //                        |x|
  // matrix_vvT = |x y z| * |y| * cos(1 - θ)
  //                        |z|
  matrix_vvT[0] = vec_x * vec_x * cos_theta1;
  matrix_vvT[1] = vec_x * vec_y * cos_theta1;
  matrix_vvT[2] = vec_x * vec_z * cos_theta1;
  matrix_vvT[3] = vec_x * vec_y * cos_theta1;
  matrix_vvT[4] = vec_y * vec_y * cos_theta1;
  matrix_vvT[5] = vec_y * vec_z * cos_theta1;
  matrix_vvT[6] = vec_x * vec_z * cos_theta1;
  matrix_vvT[7] = vec_y * vec_z * cos_theta1;
  matrix_vvT[8] = vec_z * vec_z * cos_theta1;
  //             | 0 -z  y|
  // matrix_vX = | z  0 -x| * sin(θ)
  //             |-y  x  0|
  matrix_vX[0] = 0;
  matrix_vX[1] = -vec_z * sin_theta;
  matrix_vX[2] = vec_y * sin_theta;
  matrix_vX[3] = vec_z * sin_theta;
  matrix_vX[4] = 0;
  matrix_vX[5] = -vec_x * sin_theta;
  matrix_vX[6] = -vec_y * sin_theta;
  matrix_vX[7] = vec_x * sin_theta;
  matrix_vX[8] = 0;
  //               | 1 0 0 |
  // matrix_unit = | 0 1 0 | * cos(θ)
  //               | 0 0 1 |
  matrix_unit[0] = 1 * cos_theta;
  matrix_unit[4] = 1 * cos_theta;
  matrix_unit[8] = 1 * cos_theta;

  for (int i = 0; i < 9; i++) {
    rotation_matrix_[i] = matrix_unit[i] + matrix_vvT[i] + matrix_vX[i];
  }
  return 0;
}

/**
 * Convert the rotation vector to matrix
 * Not use now
 */
int InnoGalvoMirrorCheck::convert_rotation_angle_to_matrix_() {
  rotation_matrix_[0] = cos(input_vector_.y) * cos(input_vector_.z);
  rotation_matrix_[1] =
        cos(input_vector_.z) * sin(input_vector_.x) * sin(input_vector_.y)
        - cos(input_vector_.x) * sin(input_vector_.z);
  rotation_matrix_[2] =
        sin(input_vector_.x) * sin(input_vector_.z)
        + cos(input_vector_.x) * cos(input_vector_.z) * sin(input_vector_.y);
  rotation_matrix_[3] = cos(input_vector_.y) * sin(input_vector_.z);
  rotation_matrix_[4] =
        cos(input_vector_.x) * cos(input_vector_.z)
        + sin(input_vector_.x) * sin(input_vector_.y) * sin(input_vector_.z);
  rotation_matrix_[5] =
        cos(input_vector_.x) * sin(input_vector_.y) * sin(input_vector_.z)
        - cos(input_vector_.z) * sin(input_vector_.z);
  rotation_matrix_[6] = -sin(input_vector_.y);
  rotation_matrix_[7] = cos(input_vector_.y) * sin(input_vector_.x);
  rotation_matrix_[8] = cos(input_vector_.x) * cos(input_vector_.y);
  return 0;
}

/**
 * Fitting plane by least square
 * @param  ground_points  input ground points
 * @param  coeff_plane  output the coeff of plane equation
 * @return < 0 failed, 0 success
 */
int InnoGalvoMirrorCheck::fitting_plane_from_points_(
                            const InnoCheckPoints &ground_points,
                            InnoGroundCoeff *coeff_plane) {
  const int points_count = static_cast<int>(ground_points.size());
  if (points_count < 3) {
    // inno_log_info("Not enough points to fitting plane, %d", points_count);
    return -1;
  }
  InnoPointXYZT sum = {0, 0, 0};
  for (int i = 0; i < points_count; i++) {
    sum.x += ground_points[i].x;
    sum.y += ground_points[i].y;
    sum.z += ground_points[i].z;
  }
  InnoPointXYZT centroid = {0, 0, 0};
  centroid.x = sum.x / static_cast<double>(points_count);
  centroid.y = sum.y / static_cast<double>(points_count);
  centroid.z = sum.z / static_cast<double>(points_count);
  double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (int i = 0; i < points_count; i++) {
    InnoPointXYZT temp;
    temp.x = ground_points[i].x - centroid.x;
    temp.y = ground_points[i].y - centroid.y;
    temp.z = ground_points[i].z - centroid.z;

    xx += temp.x * temp.x;
    xy += temp.x * temp.y;
    xz += temp.x * temp.z;
    yy += temp.y * temp.y;
    yz += temp.y * temp.z;
    zz += temp.z * temp.z;
  }
  double detX = yy * zz - yz * yz;
  double detY = xx * zz - xz * xz;
  double detZ = xx * yy - xy * xy;
  double detMax = std::max(std::max(detX, detY), detZ);
  if (detMax <= 0) {
    coeff_plane->a = 0;
    coeff_plane->b = 0;
    coeff_plane->c = 0;
    coeff_plane->d = 0;
    return -2;
  } else {
    InnoVector3D normal_vector{};
    double a = 0, b = 0;
    if (fabs(detMax - detX) < INNO_EPSLION_VALUE) {
      a = static_cast<double>((xz * yz - xy * zz) / detX);
      b = static_cast<double>((xy * yz - xz * yy) / detX);
      normal_vector = {1.0, a, b};
    } else if (fabs(detMax - detY) < INNO_EPSLION_VALUE) {
      a = static_cast<double>((yz * xz - xy * zz) / detY);
      b = static_cast<double>((xy * xz - yz * xx) / detY);
      normal_vector = {a, 1.0, b};
    } else {
      a = static_cast<double>((yz * xy - xz * yy) / detZ);
      b = static_cast<double>((xz * xy - yz * xx) / detZ);
      normal_vector = {a, b, 1.0};
    }

    double vector_module = calculate_vector_module_(normal_vector);
    if (vector_module < INNO_EPSLION_VALUE) {
      return -3;
    }
    normal_vector.x /= vector_module;
    normal_vector.y /= vector_module;
    normal_vector.z /= vector_module;

    coeff_plane->a = normal_vector.x;
    coeff_plane->b = normal_vector.y;
    coeff_plane->c = normal_vector.z;
    coeff_plane->d = -(coeff_plane->a * centroid.x
                    + coeff_plane->b * centroid.y
                    + coeff_plane->c * centroid.z);
  }
  return 0;
}

/**
 * Use 3 points to fit plane
 * @param  p0  input point 1
 * @param  p1  input point 2
 * @param  p2  input point 3
 * @param  plane  output the coeff of plane
 * @return double  output module
 */
double InnoGalvoMirrorCheck::estimate_param_(const InnoPointXYZT &p0,
                                            const InnoPointXYZT &p1,
                                            const InnoPointXYZT &p2,
                                            InnoGroundCoeff *plane) {
    std::vector<double> vec1{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
    std::vector<double> vec2{p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    plane->a = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    plane->b = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    plane->c = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    plane->d = -(plane->a * p0.x + plane->b * p0.y + plane->c * p0.z);
    return std::sqrt(plane->a * plane->a
                     + plane->b * plane->b
                     + plane->c * plane->c);
}

/**
 * Compute all distance of points to plane
 * @param  in_points input ground points
 * @param  plane input plane coeff
 * @param  module input module
 * @param  valid_count  output valid points count
 * @return  double  output mean distance
 */
double InnoGalvoMirrorCheck::compute_distance_(
                            const InnoCheckPoints &in_points,
                            const InnoGroundCoeff &plane,
                            const double &module,
                            uint32_t *valid_count) {
  if (module <= 0) {
    return -1.0;
  }
  size_t points_size = in_points.size();
  if (points_size < 3) return -1;
  uint32_t index = 0;
  uint32_t valid_index = 0;
  uint32_t step = 1;
  if (points_size > 400) {
    step = 2;
  }
  double total = 0.0;
  while (index < points_size) {
    double distance = std::abs((plane.a * in_points[index].x
                                  + plane.b * in_points[index].y
                                  + plane.c * in_points[index].z
                                  + plane.d) / module);
    if (distance < kInnoGroundDistanceThreshold2) {
      ++valid_index;
    }
    index += step;
    total += distance;
  }
  if (index == 0) {
    return -1.0;
  }
  *valid_count = valid_index * step;
  return total / index / step;
}

/**
 * Quick fit plane by RANSAC
 * @param  in_points  input points
 * @param  threshold_distance  min distance threshold
 * @param  is_fine  true: more times to fit
 * @param  out_plane  output plane coeff
 * @return  < 0 failed, 0 success
 */
int InnoGalvoMirrorCheck::quick_ransac_plane_(
                                const InnoCheckPoints &in_points,
                                InnoGroundCoeff *out_plane) {
  uint32_t points_size = static_cast<uint32_t>(in_points.size());
  if (points_size < 3) {
    return -1;
  }
  uint32_t max_times = 30;
  std::default_random_engine randomEngine;
  std::uniform_int_distribution<uint32_t> uniform{0, points_size - 1};
  randomEngine.seed(10U);
  uint32_t valid_threshold = static_cast<uint32_t>(points_size
                                         * kInnoCheckValidFitPercent);
  // uint32_t max_valid_threshold = static_cast<uint32_t>(num * 0.95);
  uint32_t valid_count = 0;
  // uint32_t max_valid_count = 0;
  int radom0 = 0;
  int radom1 = 0;
  int radom2 = 0;
  InnoGroundCoeff plane_coeff = {0, 0, 0, 0};
  while (max_times--) {
    radom0 = uniform(randomEngine);
    radom1 = uniform(randomEngine);
    radom2 = uniform(randomEngine);
    double module = estimate_param_(in_points[radom0], in_points[radom1],
                                   in_points[radom2], &plane_coeff);
    if (module < DBL_MIN) {
        continue;
    }
    double meanDistance
         = compute_distance_(in_points, plane_coeff, module, &valid_count);
    if (meanDistance < 0) {
        continue;
    }
    // inno_log_info("meanDistance=%lf, valid_count=%d, threshold=%d",
    //               meanDistance, valid_count, consensusThreshold);
    if (valid_count > valid_threshold) {
      // max_valid_count = valid_count;
      out_plane->a = plane_coeff.a;
      out_plane->b = plane_coeff.b;
      out_plane->c = plane_coeff.c;
      out_plane->d = plane_coeff.d;
      // if (max_valid_count > max_valid_threshold) {
      //   break;
      // }
      return 0;
    }
  }
  return -2;
  // if (max_valid_count > valid_threshold) {
  //   return 0;
  // } else {
  //   return -2;
  // }
}

/**
 * Fit plane by ransac and least square
 * @param  in_points  input ground points
 * @param  coeff  output finally plane coeff
 * @return  < 0 failed, 0 success
 */
int InnoGalvoMirrorCheck::fitting_plane_ransac_least_(
                           const InnoCheckPoints &in_points,
                           const double delta_z_per_10us,
                           InnoGroundCoeff *coeff) {
  int ret = quick_ransac_plane_(in_points, coeff);
  if (ret < 0) {
    // inno_log_warning("!quick_ransac_plane ret=%d", ret);
    return -1;
  }
  InnoCheckPoints filter_points;
  double module = std::sqrt(coeff->a * coeff->a
                          + coeff->b * coeff->b
                          + coeff->c * coeff->c);
  if (module <= 0) {
    return -2;
  }
  double delta_z = 0.0;
  for (auto &p : in_points) {
    double distance = (coeff->a * p.x
                     + coeff->b * p.y
                     + coeff->c * p.z + coeff->d) / module;
    if (std::abs(distance) < kInnoGroundDistanceThreshold1) {
      // motion compensation
      // bottom -> top, very close to start time
      if (p.reserved < 100) {
        delta_z = delta_z_per_10us * p.reserved;
      }
      filter_points.push_back({p.x, p.y, p.z + delta_z});
    }
  }
  ret = fitting_plane_from_points_(filter_points, coeff);
  return ret;
}

/******************************************************************/
/*********************** Max Distance Check ***********************/
/******************************************************************/

/**
 * Construct a new Max Distance Check Object
 */
InnoMaxDistanceCheck::InnoMaxDistanceCheck() {
  vehicle_speed_ = 0;
  distance_speed_update_ts_ms_ = 0;
  last_frame_idx_ = 0;
  check_idx_ = 0;
  total_speed_ = 0;
  total_refl_ = 0;
  valid_check_count_ = 0;
  data_last_frame_idx_ = 0;
  on_tunnel_ = false;
  tunnel_collect_size_ = 0;
  frame_points_size_ = 0;
}

/**
 * Destroy the Max Distance Check Object
 */
InnoMaxDistanceCheck::~InnoMaxDistanceCheck() {
}

/*
 * Restart check
 */
void InnoMaxDistanceCheck::restart_check() {
  check_idx_ = 0;
  total_speed_ = 0;
  total_refl_ = 0;
  valid_check_count_ = 0;
  last_frame_idx_ = 0;
  on_tunnel_times_ = 0;
}

/**
 * Update the lateset speed from vehicle
 * @param  speed  input speed
 */
void InnoMaxDistanceCheck::update_vehicle_speed(const double speed) {
  vehicle_speed_ = speed;
  distance_speed_update_ts_ms_
                 = InnoUtils::get_time_ms();
}

/**
 * Check whether the points on tunnel
 * @param  pkt  InnoDataPacket
 */
int InnoMaxDistanceCheck::check_data_on_tunnel(const InnoDataPacket &pkt) {
  uint64_t time_diff = InnoUtils::get_time_ms()
                                         - distance_speed_update_ts_ms_;
  if (vehicle_speed_ <= 0 || time_diff > kInnoUpdateSpeedMaxDelayTime) {
    return -1;
  }
  if (pkt.type != INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
    return -1;
  }
  if (data_last_frame_idx_ != pkt.idx) {
    data_last_frame_idx_ = pkt.idx;
    double percent = frame_points_size_ > kInnoMinTunnelPointCount ?
                     tunnel_collect_size_ * 1.0 / frame_points_size_ : 0;
    if (percent > kInnoOnTunnelMinPercent) {
      on_tunnel_ = true;
    } else {
      on_tunnel_ = false;
    }
    inno_log_trace("frame_points_size_=%" PRI_SIZEU ", " \
                  "tunnel_collect_size_=%"  PRI_SIZEU  ", " \
                  "percent = %f",
                            frame_points_size_,
                            tunnel_collect_size_,
                            percent);
    tunnel_collect_size_ = 0;
    frame_points_size_ = 0;
  }
  char *p_addr = reinterpret_cast<char*>(const_cast<InnoDataPacket*>(&pkt));
  InnoXyzPoint *xyz_points = reinterpret_cast<InnoXyzPoint *>
                    (p_addr + sizeof(InnoDataPacket));
  const uint32_t point_size = pkt.item_number;
  uint32_t i = 0;
  for (i = 0; i < point_size; ++i) {
    const InnoXyzPoint &point = xyz_points[i];
    if (point.z <= 0 || point.z > 150) {
      continue;
    }
    ++frame_points_size_;
    if (point.y > kInnoTunnelLeft && point.y < kInnoTunnelRight) {
      ++tunnel_collect_size_;
    }
  }
  return 0;
}

/**
 * Check max distance entrance
 * @return < 0: failed
 *           0: checking
 *           1: check completed
 */
int InnoMaxDistanceCheck::check_max_distance(
                            const char *buf,
                            InnoMaxDistanceResult *check_result) {
  inno_log_verify(buf, "NULL pointer");
  uint32_t max_check_times = 0;
  uint32_t threshold_check_times = 0;
  double threshold_normal_refl = 0;
  double threshold_fault_refl = 0;
  double threshold_min_speed = 0;
  double threshold_start_check_speed = 0;
  uint64_t frame_idx = 0;
  uint32_t points_count = 0;
  double frame_refl = 0.0;

  int got = sscanf(buf, "%u,%u,%lf,%lf,%lf,%lf,"
#if defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32)
                         "%llu,"
#else
                         "%lu,"
#endif
                         "%u,%lf",
                                       &max_check_times,
                                       &threshold_check_times,
                                       &threshold_normal_refl,
                                       &threshold_fault_refl,
                                       &threshold_min_speed,
                                       &threshold_start_check_speed,
                                       &frame_idx,
                                       &points_count,
                                       &frame_refl);
  if (got != 9
      || max_check_times == 0
      || threshold_check_times == 0
      || threshold_normal_refl < INNO_EPSLION_VALUE
      || threshold_fault_refl < INNO_EPSLION_VALUE
      || threshold_min_speed < INNO_EPSLION_VALUE
      || threshold_start_check_speed < INNO_EPSLION_VALUE) {
    // inno_log_warning("Invalid max distance params: %s", buf);
    restart_check();
    return -1;
  }
  if (last_frame_idx_ >= frame_idx) {
    // invalid frame idx
    restart_check();
    return -1;
  }
  // Check frame
  memset(check_result, 0, sizeof(InnoMaxDistanceResult));
  check_result->latest_frame_idx = frame_idx;
  check_result->threshold_valid_count = threshold_check_times;
  check_result->threshold_normal_refl = threshold_normal_refl;
  check_result->threshold_fault_refl = threshold_fault_refl;
  check_result->threshold_min_speed = threshold_min_speed;
  check_result->threshold_check_speed = threshold_start_check_speed;
  check_result->fault_status = INNO_MAX_DISTANCE_STATUS_CHECKING;
  last_frame_idx_ = frame_idx;
  // Check the update time of speed.
  // If speed update timeout return immediately.
  uint64_t time_diff = InnoUtils::get_time_ms() - distance_speed_update_ts_ms_;
  if (time_diff > kInnoUpdateSpeedMaxDelayTime) {
    check_result->fault_status = INNO_MAX_DISTANCE_STATUS_SPEED_TIMEOUT;
    restart_check();
    return -1;
  } else {
    if (check_idx_ < max_check_times) {
      // Checking...
      total_speed_ += vehicle_speed_;
      if (vehicle_speed_ >= kInnoMinHealSpeed
          && frame_refl > INNO_EPSLION_VALUE
          && points_count > 0) {
        total_refl_ += frame_refl;
        ++valid_check_count_;
      }
      if (on_tunnel_) {
        ++on_tunnel_times_;
      }
      ++check_idx_;
    }
    if (check_idx_ >= max_check_times) {
      // Checking Done!
      check_result->latest_check_idx = check_idx_;
      check_result->total_speed = total_speed_;
      check_result->total_refl = total_refl_;
      check_result->mean_speed = total_speed_ / check_idx_;
      check_result->valid_frame_count = valid_check_count_;
      check_result->mean_refl
            = valid_check_count_ > 0 ? total_refl_ / valid_check_count_ : 0;
      if (check_result->mean_speed >= kInnoMinHealSpeed
          && check_result->mean_refl >= threshold_normal_refl) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_NORMAL;
      } else if (check_result->mean_speed < kInnoMinHealSpeed) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_SPEED_TOO_LOW;
      } else if (valid_check_count_ < threshold_check_times) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_INVALID_COUNT;
      } else if (check_result->mean_speed < threshold_min_speed) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_INVALID_SPEED;
      } else if (on_tunnel_times_ >= max_check_times / 3) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_ON_TUNNEL;
      } else if (check_result->mean_refl < threshold_fault_refl) {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_SET_FAULT;
      } else {
        check_result->fault_status = INNO_MAX_DISTANCE_STATUS_MID;
      }
      restart_check();
      return 1;
    }
  }
  return 0;
}
}  // namespace innovusion
