/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_common/inno_lidar_packet_utils.h"

#include <algorithm>
#include <cmath>

#include "sdk_common/nps_adjustment.h"
#include "sdk_common/robin_nps_adjustment.h"
#include "utils/inno_lidar_log.h"
#include "utils/math_tables.h"
#include "utils/net_manager.h"
#include "utils/utils.h"
#include "sdk_common/inno_lidar_api.h"
int InnoDataPacketUtils::v_angle_offset_[INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD][kInnoChannelNumber];
int InnoDataPacketUtils::init_ = InnoDataPacketUtils::init_f();
int8_t InnoDataPacketUtils::nps_adjustment_[kVTableSize_][kHTableSize_][kInnoChannelNumber][kXZSize_];
const double InnoDataPacketUtils::kAdjustmentUnitInMeter_ = 0.0025;
const double InnoDataPacketUtils::kAdjustmentUnitInMeterRobin_ = 0.001;
int8_t InnoDataPacketUtils::robine_nps_adjustment_[kRobinEScanlines_][kHTableSize_][kXYZSize_];
int8_t InnoDataPacketUtils::robinw_nps_adjustment_[kRobinWScanlines_][kHTableSize_][kXYZSize_];
int8_t InnoDataPacketUtils::robinw_nps_adjustment_pin_[kRobinWDistSize_][kRobinWScanlines_][kHTableSize_][kXYZSize_];
// interval boundary 7.5, 8.5, 9.5, 12.5, 17.5, 25, 40cm for tables at d = 7, 8, 9, 10, 15, 20, 30, 50cm
const uint32_t InnoDataPacketUtils::kRobinDist_[kRobinWDistSize_-1] = {30, 34, 38, 50, 70, 100, 160};
int8_t InnoDataPacketUtils::robinelite_nps_adjustment_[kRobinEliteScanlines_][kHTableSize_][kXYZSize_];
int8_t InnoDataPacketUtils::vehicle_coordinate_ = 0;

// for robinW
const uint8_t InnoDataPacketUtils::robinw_channel_mapping[48] = {
  0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
  1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45,
  2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46,
  3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
};

const uint8_t InnoDataPacketUtils::robine_channel_mapping[48] = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
};

const uint8_t InnoDataPacketUtils::robinelite_channel_mapping[96] = {
  1, 17, 33, 48, 64, 80,
  3, 19, 35, 50, 66, 82,
  5, 21, 37, 52, 68, 84,
  7, 23, 39, 54, 70, 86,
  9, 25, 41, 56, 72, 88,
  11, 27, 43, 58, 74, 90,
  13, 29, 45, 60, 76, 92,
  15, 31, 47, 62, 78, 94,
  0, 2, 4, 49, 51, 53,
  6, 8, 10, 55, 57, 59,
  12, 14, 16, 61, 63, 65,
  18, 20, 22, 67, 69, 71,
  24, 26, 28, 73, 75, 77,
  30, 32, 34, 79, 81, 83,
  36, 38, 40, 85, 87, 89,
  42, 44, 46, 91, 93, 95,
};

uint32_t InnoPacketReader::calculate_packet_crc32(const InnoCommonHeader *header) {
  uint32_t crc = innovusion::InnoUtils::crc32_start();
  crc = innovusion::InnoUtils::crc32_do(crc, &header->version, sizeof(header->version));
  size_t off = offsetof(struct InnoCommonHeader, size);
  crc = innovusion::InnoUtils::crc32_do(crc, &header->size, header->size - off);
  return innovusion::InnoUtils::crc32_end(crc);
}

void InnoPacketReader::set_packet_crc32(InnoCommonHeader *header) {
  header->checksum = calculate_packet_crc32(header);
}

bool InnoPacketReader::verify_packet_crc32(const InnoCommonHeader *header) {
  uint32_t checksum = calculate_packet_crc32(header);
  if (header->checksum != checksum) {
    inno_log_warning("checksum mismatch 0x%x vs 0x%x", header->checksum, checksum);
    return false;
  } else {
    return true;
  }
}

uint32_t InnoPacketReader::calculate_http_crc32(const char *buffer, uint32_t length, bool append) {
  return innovusion::InnoUtils::calculate_http_crc32(buffer, length, append);
}

int InnoPacketReader::verify_http_crc32(const char *buffer, const char *url) {
  return innovusion::InnoUtils::verify_http_crc32(buffer, url);
}

int InnoDataPacketUtils::init_f(void) {
  init_f_falcon();
  init_f_robin();
  return 0;
}

// init the nps_adjustment_
int InnoDataPacketUtils::init_f_falcon(void) {
  for (uint32_t ich = 0; ich < kInnoChannelNumber; ich++) {
    v_angle_offset_[INNO_ITEM_TYPE_SPHERE_POINTCLOUD][ich] = ich * kInnoFaconVAngleDiffBase;
    // falconII NT3
    v_angle_offset_[INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD][ich] = ich * kInnoFaconVAngleDiffBase;
  }
  // init the nps_adjustment_
  size_t input_size = (kVTableEffeHalfSize_ * 2 + 1) * (kHTableEffeHalfSize_ * 2 + 1) * 2 * kInnoChannelNumber;
  inno_log_verify(sizeof(kInnoPs2Nps) == input_size * sizeof(double), "kInnoPs2Nps");

  inno_log_verify(kVTableSize_ >= kVTableEffeHalfSize_ * 2 + 1, "kVTableSize_");
  inno_log_verify(kHTableSize_ >= kHTableEffeHalfSize_ * 2 + 1, "kHTableSize_");
  memset(nps_adjustment_, 0, sizeof(nps_adjustment_));
  static double k_max[2] = {-100, -100};
  static double k_min[2] = {100, 100};
  for (uint32_t v = 0; v < kVTableEffeHalfSize_ * 2 + 1; v++) {
    for (uint32_t h = 0; h < kHTableEffeHalfSize_ * 2 + 1; h++) {
      for (uint32_t ich = 0; ich < kInnoChannelNumber; ich++) {
        for (uint32_t xz = 0; xz < kXZSize_; xz++) {
          double k = kInnoPs2Nps[xz][ich][v][h];
          double u = k / kAdjustmentUnitInMeter_;
          double q = std::floor(u + 0.5);
          nps_adjustment_[v][h][ich][xz] = q;
          // inno_log_debug("k=%f q=%f %d",
          //                k, q, nps_adjustment_[v][h][ich][xz]);
          k_max[xz] = std::max(k_max[xz], k);
          k_min[xz] = std::min(k_min[xz], k);
        }
      }
    }
  }
  // nps: x_adj=[-0.028944, -0.010921] range=0.018023, z_adj=[0.027061,0.037247] range=0.010186
  /*
  inno_log_info("nps: x_adj=[%f, %f] range=%f, "
                "z_adj=[%f,%f] range=%f",
                k_min[0], k_max[0],
                k_max[0] - k_min[0],
                k_min[1], k_max[1],
                k_max[1] - k_min[1]);
  */
  return 0;
}

int InnoDataPacketUtils::init_f_robin(void) {
  for (uint32_t ich = 0; ich < kInnoChannelNumber; ich++) {
    v_angle_offset_[INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD][ich] = ich * kInnoRobinEVAngleDiffBase;
    v_angle_offset_[INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD][ich] = ich * kInnoRobinWVAngleDiffBase;
  }

  init_robine_nps_adjustment_();
  init_robinw_nps_adjustment_();
  init_robinw_nps_adjustment_pin_();

  init_robinelite_nps_adjustment_();

  return 0;
}

void InnoDataPacketUtils::init_robine_nps_adjustment_() {
  double k_max[3] = {-200, -200, -200};
  double k_min[3] = {200, 200, 200};
  size_t input_size = kRobinEScanlines_ * (kHRobinTableEffeHalfSize_ * 2 + 1) * kXYZSize_;
  inno_log_verify(sizeof(robinE_kInnoPs2Nps) == input_size * sizeof(double), "robinE_kInnoPs2Nps");
  inno_log_verify(kHTableSize_ >= kHRobinTableEffeHalfSize_ * 2 + 1, "kHTableSize_");
  memset(robine_nps_adjustment_, 0, sizeof(robine_nps_adjustment_));
  for (uint32_t scan_id = 0; scan_id < kRobinEScanlines_; scan_id++) {
    for (uint32_t h = 0; h < kHRobinTableEffeHalfSize_ * 2 + 1; h++) {
      for (uint32_t xyz = 0; xyz < kXYZSize_; xyz++) {
        double k = robinE_kInnoPs2Nps[xyz][scan_id][h];
        double u = k * 0.001 / kAdjustmentUnitInMeterRobin_;
        double q = std::floor(u + 0.5);
        robine_nps_adjustment_[scan_id][h][xyz] = q;
        // inno_log_verify(q <= 127 && q >= -128, "q %f out of range", q);
        // inno_log_debug("k=%f q=%f %d", k, q, robine_nps_adjustment_[scan_id][h][xyz]);
        k_max[xyz] = std::max(k_max[xyz], k);
        k_min[xyz] = std::min(k_min[xyz], k);
      }
    }
  }

  // inno_log_info(
  //     "robin E nps: x_adj=[%f, %f] range=%f, y_adj=[%f, %f] range=%f, "
  //     "z_adj=[%f,%f] range=%f",
  //     k_min[0], k_max[0], k_max[0] - k_min[0], k_min[1], k_max[1], k_max[1] - k_min[1], k_min[2], k_max[2],
  //     k_max[2] - k_min[2]);
}

void InnoDataPacketUtils::init_robinw_nps_adjustment_() {
  double k_max[3] = {-200, -200, -200};
  double k_min[3] = {200, 200, 200};
  size_t input_size = kRobinWScanlines_ * (kHRobinTableEffeHalfSize_ * 2 + 1) * kXYZSize_;
  inno_log_verify(sizeof(robinW_kInnoPs2Nps) == input_size * sizeof(double), "robinW_kInnoPs2Nps");
  inno_log_verify(kHTableSize_ >= kHRobinTableEffeHalfSize_ * 2 + 1, "kHTableSize_");
  memset(robinw_nps_adjustment_, 0, sizeof(robinw_nps_adjustment_));
  for (uint32_t scan_id = 0; scan_id < kRobinWScanlines_; scan_id++) {
    for (uint32_t h = 0; h < kHRobinTableEffeHalfSize_ * 2 + 1; h++) {
      for (uint32_t xyz = 0; xyz < kXYZSize_; xyz++) {
        double k = robinW_kInnoPs2Nps[xyz][scan_id][h];
        double u = k * 0.001 / kAdjustmentUnitInMeterRobin_;
        double q = std::floor(u + 0.5);
        robinw_nps_adjustment_[scan_id][h][xyz] = q;
        inno_log_verify(q <= 127 && q >= -128, "q %f out of range", q);
        // inno_log_debug("k=%f q=%f %d", k, q, robinw_nps_adjustment_[scan_id][h][xyz]);
        k_max[xyz] = std::max(k_max[xyz], k);
        k_min[xyz] = std::min(k_min[xyz], k);
      }
    }
  }

  // inno_log_info(
  //     "robin W nps: x_adj=[%f, %f] range=%f, y_adj=[%f, %f] range=%f, "
  //     "z_adj=[%f,%f] range=%f",
  //     k_min[0], k_max[0], k_max[0] - k_min[0], k_min[1], k_max[1], k_max[1] - k_min[1], k_min[2], k_max[2],
  //     k_max[2] - k_min[2]);
}

void InnoDataPacketUtils::init_robinw_nps_adjustment_pin_() {
  double k_max[3] = {-200, -200, -200};
  double k_min[3] = {200, 200, 200};
  size_t input_size = kRobinWDistSize_ * kRobinWScanlines_ * (kHRobinTableEffeHalfSize_ * 2 + 1) * kXYZSize_;
  inno_log_verify(sizeof(robinW_kInnoPs2Nps_PIN) == input_size * sizeof(double), "robinW_kInnoPs2Nps_PIN");
  inno_log_verify(kHTableSize_ >= kHRobinTableEffeHalfSize_ * 2 + 1, "kHTableSize_");
  memset(robinw_nps_adjustment_pin_, 0, sizeof(robinw_nps_adjustment_pin_));
  for (uint32_t d = 0; d < kRobinWDistSize_; d++) {
    for (uint32_t scan_id = 0; scan_id < kRobinWScanlines_; scan_id++) {
      for (uint32_t h = 0; h < kHRobinTableEffeHalfSize_ * 2 + 1; h++) {
        for (uint32_t xyz = 0; xyz < kXYZSize_; xyz++) {
          double k = robinW_kInnoPs2Nps_PIN[d][xyz][scan_id][h];
          double u = k * 0.001 / kAdjustmentUnitInMeterRobin_;
          double q = std::floor(u + 0.5);
          robinw_nps_adjustment_pin_[d][scan_id][h][xyz] = q;
          inno_log_verify(q <= 127 && q >= -128, "q %f out of range", q);
          // inno_log_debug("k=%f q=%f %d", k, q, robinw_nps_adjustment_[d][scan_id][h][xyz]);
          k_max[xyz] = std::max(k_max[xyz], k);
          k_min[xyz] = std::min(k_min[xyz], k);
        }
      }
    }
  }

  // inno_log_info(
  //     "robin W nps: x_adj=[%f, %f] range=%f, y_adj=[%f, %f] range=%f, "
  //     "z_adj=[%f,%f] range=%f",
  //     k_min[0], k_max[0], k_max[0] - k_min[0], k_min[1], k_max[1], k_max[1] - k_min[1], k_min[2], k_max[2],
  //     k_max[2] - k_min[2]);
}

void InnoDataPacketUtils::init_robinelite_nps_adjustment_() {
  double k_max[3] = {-200, -200, -200};
  double k_min[3] = {200, 200, 200};
  size_t input_size = kRobinEliteScanlines_ * (kHRobinTableEffeHalfSize_ * 2 + 1) * kXYZSize_;
  inno_log_verify(sizeof(robinElite_kInnoPs2Nps) == input_size * sizeof(double), "robinElite_kInnoPs2Nps");
  inno_log_verify(kHTableSize_ >= kHRobinTableEffeHalfSize_ * 2 + 1, "kHTableSize_");
  memset(robinelite_nps_adjustment_, 0, sizeof(robinelite_nps_adjustment_));
  for (uint32_t scan_id = 0; scan_id < kRobinEliteScanlines_; scan_id++) {
    for (uint32_t h = 0; h < kHRobinTableEffeHalfSize_ * 2 + 1; h++) {
      for (uint32_t xyz = 0; xyz < kXYZSize_; xyz++) {
        double k = robinElite_kInnoPs2Nps[xyz][scan_id][h];
        double u = k * 0.001 / kAdjustmentUnitInMeterRobin_;
        double q = std::floor(u + 0.5);
        robinelite_nps_adjustment_[scan_id][h][xyz] = q;
        inno_log_verify(q <= 127 && q >= -128, "q %f out of range", q);
        // inno_log_debug("k=%f q=%f %d", k, q, robinelite_nps_adjustment_[scan_id][h][xyz]);
        k_max[xyz] = std::max(k_max[xyz], k);
        k_min[xyz] = std::min(k_min[xyz], k);
      }
    }
  }

  // inno_log_info(
  //     "robin Elite nps: x_adj=[%f, %f] range=%f, y_adj=[%f, %f] range=%f, "
  //     "z_adj=[%f,%f] range=%f",
  //     k_min[0], k_max[0], k_max[0] - k_min[0], k_min[1], k_max[1], k_max[1] - k_min[1], k_min[2], k_max[2],
  //     k_max[2] - k_min[2]);
}

inline void InnoDataPacketUtils::lookup_xz_adjustment_(const InnoBlockAngles &angles, uint32_t ch, double *x,
                                                       double *z) {
  uint32_t v = angles.v_angle / 512;
  uint32_t h = angles.h_angle / 512;
  v += kVTableEffeHalfSize_;
  h += kHTableEffeHalfSize_;
  // avoid index out-of-bound
  v = v & (kVTableSize_ - 1);
  h = h & (kHTableSize_ - 1);
  int8_t *addr_x = &nps_adjustment_[v][h][ch][0];
  int8_t *addr_z = addr_x + 1;
  *x = *addr_x * kAdjustmentUnitInMeter_;
  *z = *addr_z * kAdjustmentUnitInMeter_;
  return;
}

/*
@brief for robin scan
*/
inline void InnoDataPacketUtils::lookup_xyz_adjustment_(const InnoBlockAngles &angles, uint32_t scan_id,
                                                        uint32_t radius_unit, uint32_t firing, double adj[],
                                                        InnoItemType type) {
  int8_t (*robin_nps_adjustment)[kHTableSize_][kXYZSize_];
  if (type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD) {
    robin_nps_adjustment = robine_nps_adjustment_;
  } else if (type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD && firing == 0 &&
             (scan_id == 1 || scan_id == 25 || scan_id == 49 || scan_id == 73 ||
             scan_id == 97 || scan_id == 121 || scan_id == 145 || scan_id == 169)) {
    int d_index = kRobinWDistSize_-1;
    for (int i = 0; i < kRobinWDistSize_-1; i++) {
      if (radius_unit < kRobinDist_[i]) {
        d_index = i;
        break;
      }
    }
    // use the middle scanline in that range for current pin implementation
    scan_id += 10;
    robin_nps_adjustment = robinw_nps_adjustment_pin_[d_index];
  } else if (type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
    robin_nps_adjustment = robinelite_nps_adjustment_;
  } else {
    robin_nps_adjustment = robinw_nps_adjustment_;  // robinw_nps_adjustment_
  }

  int h_angle = angles.h_angle + (kHRobinTableEffeHalfSize_ << kTableShift_);
  int h_index = h_angle >> kTableShift_;
  // avoid index out-of-bound
  h_index = h_index & (kHTableSize_ - 1);
  if (h_index > kHTableSize_ - 2) {
    h_index = kHTableSize_ - 2;
  }

  int h_offset = h_angle & kTableMask_;
  int h_offset2 = kTableStep_ - h_offset;

  for (int i = 0; i < 3; i++) {
    int u = robin_nps_adjustment[scan_id][h_index][i];
    int v = robin_nps_adjustment[scan_id][h_index + 1][i];
    int t = u * h_offset2 + v * h_offset + kTableHalfStep_;
    int w = t >> kTableShift_;
    adj[i] = w * kAdjustmentUnitInMeterRobin_;
#if 0
    inno_log_info("i %d h_angle: %f/%f h_index: %d h_offset: %d %d u: %d v: %d t: %d w: %d adj: %f",
                   i, angles.h_angle*0.00549, h_angle*0.00549, h_index, h_offset, h_offset2, u, v, t, w, adj[i]);
#endif
  }

  return;
}

void InnoDataPacketUtils::get_xyzr_meter(const InnoBlockAngles angles, const uint32_t radius_unit,
                                         const uint32_t channel, InnoXyzrD *result, InnoItemType type,
                                         uint32_t firing, bool long_distance_mode) {
  if (type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    result->radius = (radius_unit << long_distance_mode) * kMeterPerInnoDistanceUnit200;
  } else {
    result->radius = radius_unit * kMeterPerInnoDistanceUnit400;
  }
  double t;
  if (angles.v_angle >= 0) {
    t = result->radius * innovusion::MathTables::lookup_cos_table_in_unit(angles.v_angle);
    result->x = result->radius * innovusion::MathTables::lookup_sin_table_in_unit(angles.v_angle);
  } else {
    t = result->radius * innovusion::MathTables::lookup_cos_table_in_unit(-angles.v_angle);
    result->x = -result->radius * innovusion::MathTables::lookup_sin_table_in_unit(-angles.v_angle);
  }
  if (angles.h_angle >= 0) {
    result->y = t * innovusion::MathTables::lookup_sin_table_in_unit(angles.h_angle);
    result->z = t * innovusion::MathTables::lookup_cos_table_in_unit(angles.h_angle);
  } else {
    result->y = -t * innovusion::MathTables::lookup_sin_table_in_unit(-angles.h_angle);
    result->z = t * innovusion::MathTables::lookup_cos_table_in_unit(-angles.h_angle);
  }

  // falcon nps adjustment
  if (type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD || type == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD) {
    double x_adj, z_adj;
    lookup_xz_adjustment_(angles, channel, &x_adj, &z_adj);
    // inno_log_debug("adjust %f %f", x_adj, z_adj);
    result->x += x_adj;
    result->z += z_adj;
  } else if (type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD || type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD ||
             type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
             type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
    double adj[3];
    lookup_xyz_adjustment_(angles, channel, radius_unit, firing, adj, type);
    result->x += adj[0];
    result->y += adj[1];
    result->z += adj[2];
  }

  return;
}

InnoDataPacket *InnoDataPacketUtils::convert_to_xyz_pointcloud_malloced(const InnoDataPacket &src,
                                                                        RingIdConverterInterface *ring_id_converter,
                                                                        const char *hvangle_table) {
  // 1. calculate max size and allocate new data packet
  size_t new_pkt_size =
      get_data_packet_size(src.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD ? INNO_ITEM_TYPE_XYZ_POINTCLOUD
                                                                        : INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD,
                           InnoDataPacketUtils::get_max_points_count(src), INNO_MULTIPLE_RETURN_MODE_NONE);

  InnoDataPacket *new_pkt = reinterpret_cast<InnoDataPacket *>(malloc(new_pkt_size));

  if (!new_pkt) {
    return NULL;
  }
  // 2. convert SPHERE_POINTCLOUD to XYZ_POINTCLOUD
  bool ret = convert_to_xyz_pointcloud(src, new_pkt, new_pkt_size, false, ring_id_converter, hvangle_table);
  if (ret) {
    return new_pkt;
  } else {
    free(new_pkt);
    return NULL;
  }
}

bool InnoDataPacketUtils::convert_to_xyz_pointcloud(const InnoDataPacket &src, InnoDataPacket *dest, size_t dest_size,
                                                    bool crc_disable, RingIdConverterInterface *ring_id_converter,
                                                    const char *hvangle_table) {
  if (!CHECK_SPHERE_POINTCLOUD_DATA(src.type)) {
    inno_log_warning("invalid type %u", src.type);
    return false;
  }

  uint32_t dummy_count = 0;

  size_t required_size = 0;
  uint16_t time_adjust_10us = 0;

  required_size = sizeof(InnoDataPacket);
  if (required_size > dest_size) {
    inno_log_warning("not enough size %" PRI_SIZELU " %" PRI_SIZELU, required_size, dest_size);
    return false;
  }
  memcpy(dest, &src, sizeof(InnoDataPacket));
  if (src.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    dest->type = INNO_ITEM_TYPE_XYZ_POINTCLOUD;
    dest->item_size = sizeof(InnoXyzPoint);
  } else if (src.type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD) {
    dest->type = INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD;
    dest->item_size = sizeof(InnoEnXyzPoint);
  } else if (src.type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
    dest->type = INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD;
    dest->item_size = sizeof(InnoEnXyzPoint);
  } else {
    dest->type += 1;  // robin & falconIII InnoItemType xyz = sphere+1
    dest->item_size = sizeof(InnoEnXyzPoint);
  }
  dest->item_number = 0;


  {
#define CONVERT_FN(ctx, p, blk, pt, full_angles, ch, m)                                                       \
  do {                                                                                                        \
    if (pt.radius > 0) {                                                                                      \
      InnoXyzPoint *xyz_points = reinterpret_cast<InnoXyzPoint *>(dest->payload);                             \
      InnoXyzPoint &ipt = xyz_points[dest->item_number];                                                      \
      required_size += sizeof(InnoXyzPoint);                                                                  \
      if (required_size > dest_size) {                                                                        \
        inno_log_warning("not enough size %" PRI_SIZELU " %" PRI_SIZELU, required_size, dest_size);           \
        return false;                                                                                         \
      }                                                                                                       \
      get_xyz_point(blk.header, pt, full_angles.angles[ch], ch, &ipt, src.long_distance_mode);                     \
      ipt.multi_return = m;                                                                                   \
      ipt.is_2nd_return = pt.is_2nd_return;                                                                   \
      ipt.ts_10us += time_adjust_10us;                                                                        \
      if (ring_id_converter) {                                                                                \
        ipt.ring_id = ring_id_converter->get_ring_id((InnoLidarMode)p.common.lidar_mode, p.scanner_direction, \
                                                     blk.header.scan_id, ch);                                 \
      }                                                                                                       \
      dest->item_number++;                                                                                    \
    }                                                                                                         \
  } while (0)

#define CONVERT_EN_FN(ctx, p, blk, pt, full_angles, ch, m)                                          \
  do {                                                                                              \
    if (pt.radius > 0) {                                                                            \
      InnoEnXyzPoint *xyz_points = reinterpret_cast<InnoEnXyzPoint *>(dest->payload);               \
      InnoEnXyzPoint &ipt = xyz_points[dest->item_number];                                          \
      required_size += sizeof(InnoEnXyzPoint);                                                      \
      if (required_size > dest_size) {                                                              \
        inno_log_warning("not enough size %" PRI_SIZELU " %" PRI_SIZELU, required_size, dest_size); \
        return false;                                                                               \
      }                                                                                             \
      get_xyz_point(blk.header, pt, full_angles.angles[ch], ch, &ipt, (InnoItemType)src.type);      \
      ipt.multi_return = m;                                                                         \
      ipt.is_2nd_return = pt.is_2nd_return;                                                         \
      ipt.timestamp_s = ipt.ts_10us / k10UsInSecond + dest->common.ts_start_us / kUsInSecond;       \
      dest->item_number++;                                                                          \
    }                                                                                               \
  } while (0)

    if (src.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      ITERARATE_INNO_DATA_PACKET_CPOINTS(CONVERT_FN, NULL, &src, dummy_count);
    } else if (src.type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
               src.type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
      ITERARATE_INNO_DATA_PACKET_CO_CPOINTS(CONVERT_EN_FN, NULL, &src, dummy_count, hvangle_table);
    } else {
      ITERARATE_INNO_DATA_PACKET_EN_CPOINTS(CONVERT_EN_FN, NULL, &src, dummy_count);
    }
  }
  dest->common.size = required_size;

  if (crc_disable == false) {
    InnoPacketReader::set_packet_crc32(&dest->common);
  }

  return true;
}

bool InnoDataPacketUtils::check_data_packet(const InnoDataPacket &pkt, size_t size) {
  if (pkt.common.version.magic_number != kInnoMagicNumberDataPacket) {
    inno_log_warning("bad magic %x", pkt.common.version.magic_number);
    return false;
  }
  if (size && pkt.common.size > size) {
    inno_log_warning("bad size %" PRI_SIZELU " %u", size, pkt.common.size);
    return false;
  }
  bool is_data = true;
  switch (pkt.type) {
    case INNO_ITEM_TYPE_SPHERE_POINTCLOUD:
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        if (pkt.item_size != sizeof(InnoBlock1)) {
          inno_log_warning("bad block1 item size %u, sizeof(InnoBlock1): %" PRI_SIZELU, pkt.item_size,
                           sizeof(InnoBlock1));
          return false;
        }
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        if (pkt.item_size != sizeof(InnoBlock2)) {
          inno_log_warning("bad block2 item size %u", pkt.item_size);
          return false;
        }
      } else {
        inno_log_warning("bad return_mode %u", pkt.multi_return_mode);
        return false;
      }
      break;
    case INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD:
    case INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD:
    case INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD:
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        if (pkt.item_size != sizeof(InnoEnBlock1)) {
          inno_log_warning("bad block1 item size %u, sizeof(InnoEnBlock1): %" PRI_SIZELU, pkt.item_size,
                           sizeof(InnoEnBlock1));
          return false;
        }
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        if (pkt.item_size != sizeof(InnoEnBlock2)) {
          inno_log_warning("bad block2 item size %u", pkt.item_size);
          return false;
        }
      } else {
        inno_log_warning("bad return_mode %u", pkt.multi_return_mode);
        return false;
      }
      break;
    case INNO_ITEM_TYPE_XYZ_POINTCLOUD:
      if (pkt.item_size != sizeof(InnoXyzPoint)) {
        inno_log_warning("bad InnoXyzPoint item size %u", pkt.item_size);
        return false;
      }
      break;
    case INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD:
      if (pkt.item_size != sizeof(InnoEnXyzPoint)) {
        inno_log_warning("bad InnoEnXyzPoint item size %u", pkt.item_size);
        return false;
      }
      break;
    case INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD:
    case INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD:
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        if (pkt.item_size != sizeof(InnoCoBlock1)) {
          inno_log_warning("bad block1 item size %u, sizeof(InnoEnBlock1): %" PRI_SIZELU, pkt.item_size,
                           sizeof(InnoCoBlock1));
          return false;
        }
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        if (pkt.item_size != sizeof(InnoCoBlock2)) {
          inno_log_warning("bad block2 item size %u", pkt.item_size);
          return false;
        }
      } else {
        inno_log_warning("bad return_mode %u", pkt.multi_return_mode);
        return false;
      }
      break;
    default:
      is_data = false;
      break;
  }

  if (is_data) {
    size_t s =
        get_data_packet_size(InnoItemType(pkt.type), pkt.item_number, InnoMultipleReturnMode(pkt.multi_return_mode));
    if (pkt.common.size != s) {
      inno_log_warning("bad size %" PRI_SIZELU " %u", s, pkt.common.size);
      return false;
    }
    if (!InnoPacketReader::verify_packet_crc32(&pkt.common)) {
      inno_log_warning("crc32 mismatch for data packet");
      return false;
    }
    if (pkt.common.version.major_version > kInnoMajorVersionDataPacket) {
      inno_log_error("please upgrade client sdk, lidar protocol major version:%u sdk major version:%u",
                     pkt.common.version.major_version, kInnoMajorVersionDataPacket);
      return false;
    } else if (pkt.common.version.major_version == kInnoMajorVersionDataPacket &&
               pkt.common.version.minor_version > kInnoMinorVersionDataPacket) {
      static int count = 0;
      if (count++ % 100 == 0)
        inno_log_warning("please upgrade client sdk, lidar protocol minor version:%u sdk minor version:%u",
                         pkt.common.version.minor_version, kInnoMinorVersionDataPacket);
    }
    return true;
  } else if (pkt.type == INNO_ITEM_TYPE_MESSAGE || pkt.type == INNO_ITEM_TYPE_MESSAGE_LOG) {
    if (pkt.item_number != 1) {
      inno_log_warning("bad item_number %u", pkt.item_number);
      return false;
    }
    const InnoMessage *messages = reinterpret_cast<const InnoMessage *>(pkt.payload);
    if (static_cast<uint32_t>(pkt.item_size) != messages[0].size || pkt.item_size <= sizeof(InnoMessage)) {
      inno_log_warning("bad message size %u %u", pkt.item_size, messages[0].size);
      return false;
    }
    if (pkt.common.size != pkt.item_size + sizeof(InnoDataPacket)) {
      inno_log_warning("bad message size %u, %" PRI_SIZELU, pkt.common.size, pkt.item_size + sizeof(InnoDataPacket));
      return false;
    }
    if (!InnoPacketReader::verify_packet_crc32(&pkt.common)) {
      inno_log_warning("crc32 mismatch for message packet");
      return false;
    }
    return true;
  } else if (pkt.type == INNO_ROBINW_ITEM_TYPE_ANGLEHV_TABLE || pkt.type == INNO_ROBINE_LITE_TYPE_ANGLEHV_TABLE) {
    if (!InnoPacketReader::verify_packet_crc32(&pkt.common)) {
      inno_log_warning("crc32 mismatch for INNO_ROBINW_ITEM_TYPE_ANGLEHV_TABLE packet");
      return false;
    }
    return true;
  } else {
    inno_log_warning("bad type %u", pkt.type);
    return false;
  }
}

bool InnoDataPacketUtils::check_status_packet(const InnoStatusPacket &pkt, size_t size) {
  if (pkt.common.version.magic_number != kInnoMagicNumberStatusPacket) {
    inno_log_warning("bad magic %x", pkt.common.version.magic_number);
    return false;
  }
  if (size && pkt.common.size > size) {
    inno_log_warning("bad size status packet %" PRI_SIZELU " %u", size, pkt.common.size);
    return false;
  }
  if (pkt.common.size != sizeof(pkt)) {
    inno_log_warning("bad size status packet %u %" PRI_SIZELU, pkt.common.size, sizeof(pkt));
    return false;
  }

  if (!InnoPacketReader::verify_packet_crc32(&pkt.common)) {
    inno_log_warning("crc32 mismatch for status packet");
    return false;
  }
  return true;
}

int InnoDataPacketUtils::printf_status_packet(const InnoStatusPacket &pkt, char *buffer, size_t buffer_size) {
  if (!buffer) {
    return -1;
  }

  auto &c = pkt.counters;
  int ret_c =
      snprintf(buffer, buffer_size,
               "counters: "
               "point_data_packet_sent=%" PRI_SIZEU
               " point_sent=%" PRI_SIZEU
               " message_packet_sent=%" PRI_SIZEU
               " raw_data_read=%" PRI_SIZEU
               " total_frame=%" PRI_SIZEU
               " total_polygon_rotation=%" PRI_SIZEU
               " total_polygon_facet=%" PRI_SIZEU
               " power_up_time_in_second=%u "
               "process_up_time_in_second=%u "
               "lose_ptp_sync=%u "
               "bad_data[4]=%u/%u/%u/%u "
               "data_drop[8]=%u/%u/%u/%u/%u/%u/%u/%u "
               "signals[8]=%u/%u/%u/%u/%u/%u/%u/%u "
               "latency_10us_average[6]=%hu/%hu/%hu/%hu/%hu/%hu "
               "latency_10us_variation[6]=%hu/%hu/%hu/%hu/%hu/%hu "
               "latency_10us_max[6]=%hu/%hu/%hu/%hu/%hu/%hu "
               "big_latency_frame=%u "
               "bad_frame=%u "
               "big_gap_frame=%u "
               "small_gap_frame=%u "
               "cpu_percentage=%hu "
               "mem_percentage=%hu "
               "netstat_rx speed/drop/err=%hukBps/%hu/%hu "
               "netstat_tx speed/drop/err=%hukBps/%hu/%hu "
               "sys_cpu_percentage=%hu/%hu/%hu/%hu "
               "motor[5]=%hu/%hu/%hu/%hu/%hu "
               "galvo[5]=%hu/%hu/%hu/%hu/%hu "
               "in_faults=0x%" PRI_SIZEX ";",
               c.point_data_packet_sent, c.point_sent, c.message_packet_sent, c.raw_data_read, c.total_frame,
               c.total_polygon_rotation, c.total_polygon_facet, c.power_up_time_in_second, c.process_up_time_in_second,
               c.lose_ptp_sync,
               // c.bad_data[4]
               c.bad_data[0], c.bad_data[1], c.bad_data[2], c.bad_data[3],
               // data_drop[8]
               c.data_drop[0], c.data_drop[1], c.data_drop[2], c.data_drop[3], c.data_drop[4], c.data_drop[5],
               c.data_drop[6], c.data_drop[7],
               // signals[8]
               c.in_signals[0], c.in_signals[1], c.in_signals[2], c.in_signals[3], c.in_signals[4], c.in_signals[5],
               c.in_signals[6], c.in_signals[7],
               // latency_10us_average[6]
               c.latency_10us_average[0], c.latency_10us_average[1], c.latency_10us_average[2],
               c.latency_10us_average[3], c.latency_10us_average[4], c.latency_10us_average[5],
               // latency_10us_variation[6]
               c.latency_10us_variation[0], c.latency_10us_variation[1], c.latency_10us_variation[2],
               c.latency_10us_variation[3], c.latency_10us_variation[4], c.latency_10us_variation[5],
               // latency_10us_max[6]
               c.latency_10us_max[0], c.latency_10us_max[1], c.latency_10us_max[2], c.latency_10us_max[3],
               c.latency_10us_max[4], c.latency_10us_max[5], c.big_latency_frame, c.bad_frame, c.big_gap_frame,
               c.small_gap_frame, c.cpu_percentage, c.mem_percentage,
               // netstat
               c.netstat_rx_speed_kBps, c.netstat_rx_drop, c.netstat_rx_err, c.netstat_tx_speed_kBps, c.netstat_tx_drop,
               c.netstat_tx_err,
               // usage of each proc
               c.sys_cpu_percentage[0], c.sys_cpu_percentage[1], c.sys_cpu_percentage[2], c.sys_cpu_percentage[3],
               // motor[5]
               c.motor[0], c.motor[1], c.motor[2], c.motor[3], c.motor[4],
               // galvo[5]
               c.galvo[0], c.galvo[1], c.galvo[2], c.galvo[3], c.galvo[4],
               // in_faults
               pkt.in_faults.faults);

  if (ret_c < 0) {  // error
    return ret_c;
  }

  if (static_cast<size_t>(ret_c) >= buffer_size) {
    buffer[buffer_size - 1] = 0;
    return buffer_size - 1;
  }

  // InnoStatusSensorReadings sensor_readings;
  const InnoStatusSensorReadings &s = pkt.sensor_readings;
  int ret_s = snprintf(
      buffer + ret_c, buffer_size - ret_c,
      "sensor_readings: "
      "t_fpga=%hd t_laser=%hd "
      "t_adc=%hd t_board=%hd "
      "t_det=%hd/%hd/%hd/%hd "
      "t_other=%hd/%hd/%hd "
      "heater_ma=%hu "
      "m_rpm=%u g_fpm=%u "
      "m_total=%" PRI_SIZEU " g_total=%" PRI_SIZEU
      " moisture=%hu/%hu "
      "window_blockage=%hu/%hu "
      "motor=%hu/%hu/%hu/%hu/%hu/%hu "
      "galvo=%hu/%hu/%hu/%hu/%hu/%hu "
      "laser=%hu/%hu/%hu/%hu/%hu/%hu "
      "galvo_client=0x%hx/%hu "
      "\n",
      s.temperature_fpga_10th_c, s.temperature_laser_10th_c, s.temperature_adc_10th_c, s.temperature_board_10th_c,
      // temperature_det_10th_c
      s.temperature_det_10th_c[0], s.temperature_det_10th_c[1], s.temperature_det_10th_c[2],
      s.temperature_det_10th_c[3],
      // temperature_other_10th_c
      s.temperature_other_10th_c[0], s.temperature_other_10th_c[1], s.temperature_other_10th_c[2],
      //
      s.heater_current_ma, s.motor_rpm_1000th, s.galvo_fpm_1000th, s.motor_rotation_total, s.galvo_round_total,
      // moisture_index
      s.moisture_index[0], s.moisture_index[1],
      // window_blockage_index
      s.window_blockage_index[0], s.window_blockage_index[1],
      // motor
      s.motor[0], s.motor[1], s.motor[2], s.motor[3], s.motor[4], s.motor[5],
      // galvo
      s.galvo[0], s.galvo[1], s.galvo[2], s.galvo[3], s.galvo[4], s.galvo[5],
      // laser
      s.laser[0], s.laser[1], s.laser[2], s.laser[3], s.laser[4], s.laser[5],
      // galvo mirror offset check in client sdk
      s.galvo_status_client, s.galvo_offset_angle_client);

  if (ret_s < 0) {  // error
    return ret_s;
  }

  if (static_cast<size_t>(ret_c + ret_s) >= buffer_size) {
    buffer[buffer_size - 1] = 0;
    return buffer_size - 1;
  }

  return ret_c + ret_s;
}

bool InnoDataPacketUtils::raw4_header_from_net(const char *buffer, size_t buffer_size, Raw4UdpHeader *header) {
  if (buffer_size < Raw4UdpHeader::kHeaderSize) {
    return false;
  }

  if (!buffer || !header) {
    return false;
  }

  header->idx = ntohl(*(reinterpret_cast<const uint32_t *>(buffer)));
  header->field_type = buffer[Raw4UdpHeader::kFieldTypeOffset];
  header->field_sequence_id = ntohl(*(reinterpret_cast<const uint32_t *>(buffer + Raw4UdpHeader::kFieldSeqIdOffset)));
  header->flag = buffer[Raw4UdpHeader::kFlagOffset];

  return true;
}

bool InnoDataPacketUtils::raw4_header_to_net(const Raw4UdpHeader &header, char *buffer, size_t buffer_size) {
  if (!buffer || buffer_size < Raw4UdpHeader::kHeaderSize) {
    return false;
  }

  *(reinterpret_cast<uint32_t *>(buffer)) = htonl(header.idx);
  buffer[Raw4UdpHeader::kFieldTypeOffset] = header.field_type;
  *(reinterpret_cast<uint32_t *>(buffer + Raw4UdpHeader::kFieldSeqIdOffset)) = htonl(header.field_sequence_id);
  buffer[Raw4UdpHeader::kFlagOffset] = header.flag;

  return true;
}

int InnoSummaryPackage::summary_data_package(const InnoDataPacket &pkt) {
  int result = 0;
  inno_log_trace("pkt.idx=%" PRI_SIZEU ", pkt.sub_idx=%d, last=%d", pkt.idx, pkt.sub_idx, pkt.is_last_sub_frame);
  // new frame
  if (current_frame_ != (int64_t)pkt.idx) {
    // last frame don't receive is_last_sub_frame flag
    if (!next_new_frame_ && expect_frame_ > 0) {
      inno_log_warning("Miss frame %" PRI_SIZED " 's last one packet.", current_frame_);
      miss_sub_frame_last_one_counter_++;
      miss_sub_frame_gap_counter_++;
      result = -1;
    }
    // check whether is expected frame ID
    if (expect_frame_ != (int64_t)pkt.idx && expect_frame_ > 0) {
      inno_log_warning("The expected frame is %" PRI_SIZED ", but the current frame is %" PRI_SIZEU,
                       expect_frame_, pkt.idx);
      if ((int64_t)pkt.idx > expect_frame_) {
        miss_frame_counter_ += (pkt.idx - expect_frame_);
        result = -2;
      }
    }
    current_frame_ = (int64_t)pkt.idx;
    expect_frame_ = current_frame_ + 1;
    frame_counter_++;
    next_new_frame_ = false;
  }

  current_sub_frame_ = pkt.sub_idx;
  // check whether is expect sub frame
  if (expect_sub_frame_ != (int32_t)current_sub_frame_ && result == 0 && expect_sub_frame_ > 0) {
    inno_log_warning("UDP sub frame miss. The current frame is %" PRI_SIZEU
                     ", expect_sub_frame = %d, but current_sub_frame = %u",
                     pkt.idx, expect_sub_frame_, current_sub_frame_);
    miss_sub_frame_gap_counter_++;
    if ((int32_t)current_sub_frame_ > expect_sub_frame_) {
      miss_sub_frame_except_last_one_counter_ += current_sub_frame_ - expect_sub_frame_;
      inno_log_warning("Miss sub frame id from %d to %u", expect_sub_frame_, current_sub_frame_ - 1);
      result = -1;
    }
  }
  expect_sub_frame_ = current_sub_frame_ + 1;
  if (pkt.is_last_sub_frame) {
    // next frame is new frame
    next_new_frame_ = true;
    expect_sub_frame_ = 0;
  }
  if (pkt.item_number == 0) {
    inno_log_info("Empty sub frame. Current_frame_ is %" PRI_SIZEU ", current_sub_frame = %u",
                  pkt.idx, current_sub_frame_);
    empty_sub_frame_counter_++;
  }

  return result;
}

void InnoSummaryPackage::inno_data_packet_receive_stats(const InnoDataPacket &pkt) {
  if (init_seq_num_ == false) {
    init_seq_num(pkt.sub_seq);
    init_seq_num_ = true;
  }
  // Check whether the new sequence number is the highest yet seen:
  total_receive_packet_++;
  uint64_t old_seq_num = (highest_seq_num_received_ & 0xFFFF);
  uint64_t seq_num_cycle = (highest_seq_num_received_ & 0xFFFFFFFFFFFF0000);
  uint64_t seq_num_difference = static_cast<uint64_t>(static_cast<int>(pkt.sub_seq) - static_cast<int>(old_seq_num));
  uint64_t new_seq_num = 0;

  if (is_newer_packet((uint16_t)old_seq_num, pkt.sub_seq)) {
    // This packet was not an old packet received out of order, so check it:

    if (seq_num_difference >= 0x8000) {
      // The sequence number wrapped around, so start a new cycle:
      seq_num_cycle += 0x10000;
    }

    new_seq_num = seq_num_cycle | pkt.sub_seq;

    if (new_seq_num > highest_seq_num_received_) {
      highest_seq_num_received_ = new_seq_num;
    }
  } else {
    // This packet was an old packet received out of order

    if (static_cast<int>(seq_num_difference) >= 0x8000) {
      // The sequence number wrapped around, so switch to an old cycle:
      seq_num_cycle = seq_num_cycle - 0x10000;
    }
    new_seq_num = seq_num_cycle | pkt.sub_seq;

    if (new_seq_num < base_seq_num_received_) {
      base_seq_num_received_ = new_seq_num;
    }
  }
  //  print packet loss rate
  print_inno_data_packet_loss_rate();
}

void InnoSummaryPackage::print_inno_data_packet_loss_rate(int print_interval_ms) {
  auto cur_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (cur_time - last_print_time_ > print_interval_ms) {
    uint64_t delta_received_now = total_receive_packet_ - last_recieve_total_packet_count_;
    uint64_t delta_expected_now = get_expect_receive_inno_packet() - last_recieve_expect_packet_count_;

    //  if sub_seq always increase, deltaExpectedNow should > 1
    if (delta_expected_now > 1) {
      double cur_loss_fraction =
        delta_expected_now == 0 ? 0.0 : 1.0 - static_cast<double>(delta_received_now) / delta_expected_now;

      double avage_loss_fraction =
        total_receive_packet_ == 0
          ? 0.0
          : 1.0 - static_cast<double>(total_receive_packet_) / get_expect_receive_inno_packet();
      // inno_log_info("delta_received_now: %" PRI_SIZEU " delta_expected_now: %" PRI_SIZEU
      //               " cur_loss_fraction:%f avage_loss_fraction:%f ",
      //               delta_received_now, delta_expected_now, cur_loss_fraction, avage_loss_fraction);
      last_recieve_total_packet_count_ = total_receive_packet_;
      last_recieve_expect_packet_count_ = get_expect_receive_inno_packet();
      last_print_time_ = cur_time;
    }
  }
}

void InnoSummaryPackage::init_seq_num(uint16_t seqNum) {
  highest_seq_num_received_ = 0x10000 | seqNum;
  base_seq_num_received_ = 0x10000 | seqNum;
  init_seq_num_ = 0;
}

bool InnoDataPacketUtils::check_status_packet_fault(const InnoStatusPacket &pkt) {
  if (pkt.in_faults.faults || pkt.in_faults.extended_faults || pkt.in_faults2.faults ||
      pkt.in_faults2.extended_faults) {
    return true;
  }
  return false;
}

bool InnoDataPacketUtils::check_status_packet_fault_id(const InnoStatusPacket &pkt, int fid) {
  bool ret = 0;
  // fid 0- 95 store in inno_faults
  if (fid < 96) {
    if (fid >= 64) {
      ret = pkt.in_faults.extended_faults & (1U << (fid - 64));  // fid 64 - 95
    } else {
      ret = pkt.in_faults.faults & (1LLU << fid);  // fid 0 - 63
    }
  } else if (fid >= 96 && fid < 192) {  // fid 96 - 191 store in inno_faults2
    if (fid >= 64 + 96) {               // fid 160 - 191
      ret = pkt.in_faults2.extended_faults & (1U << (fid - 64 - 96));
    } else {
      ret = pkt.in_faults2.faults & (1LLU << (fid - 96));  // fid 96 - 159
    }
  }
  return ret;
}

bool InnoDataPacketUtils::fix_imu_status(const InnoStatusPacket &pkt, InnoStatusPacket &out_pkt, bool is_wgs) {
  memcpy(&out_pkt, &pkt, sizeof(InnoStatusPacket));

  if (static_cast<InnoLidarType>(pkt.common.lidar_type) == INNO_LIDAR_TYPE_FALCON) {
    // 13 degree correction
    float fix_rad = 0.22689280276;
    float cos_value = cos(fix_rad);
    float sin_value = sin(fix_rad);

    float accel_x_unit = 0.0F;
    float accel_y_unit = 0.0F;
    float accel_z_unit = 0.0F;
    float gyro_x_unit = 0.0F;
    float gyro_y_unit = 0.0F;
    float gyro_z_unit = 0.0F;
    float accel_x_ = 0.0F;
    float accel_y_ = 0.0F;
    float accel_z_ = 0.0F;
    float gyro_x_ = 0.0F;
    float gyro_y_ = 0.0F;
    float gyro_z_ = 0.0F;

    // sync to lidar coordinate system
    if (is_wgs) {
      // for falcon-k24
      accel_x_unit = pkt.sensor_readings.accel_unit_z;
      accel_y_unit = -pkt.sensor_readings.accel_unit_y;
      accel_z_unit = pkt.sensor_readings.accel_unit_x;
      gyro_x_unit = pkt.sensor_readings.gyro_unit_z;
      gyro_y_unit = -pkt.sensor_readings.gyro_unit_y;
      gyro_z_unit = pkt.sensor_readings.gyro_unit_x;
      accel_x_ = static_cast<int16_t>(pkt.sensor_readings.accel_z);
      accel_y_ = -static_cast<int16_t>(pkt.sensor_readings.accel_y);
      accel_z_ = static_cast<int16_t>(pkt.sensor_readings.accel_x);
      gyro_x_ = static_cast<int16_t>(pkt.sensor_readings.gyro_z);
      gyro_y_ = -static_cast<int16_t>(pkt.sensor_readings.gyro_y);
      gyro_z_ = static_cast<int16_t>(pkt.sensor_readings.gyro_x);
    } else {
      // for falcon-k
      accel_x_unit = pkt.sensor_readings.accel_unit_z;
      accel_y_unit = -pkt.sensor_readings.accel_unit_x;
      accel_z_unit = -pkt.sensor_readings.accel_unit_y;
      gyro_x_unit = pkt.sensor_readings.gyro_unit_z;
      gyro_y_unit = -pkt.sensor_readings.gyro_unit_x;
      gyro_z_unit = -pkt.sensor_readings.gyro_unit_y;
      accel_x_ = static_cast<int16_t>(pkt.sensor_readings.accel_z);
      accel_y_ = -static_cast<int16_t>(pkt.sensor_readings.accel_x);
      accel_z_ = -static_cast<int16_t>(pkt.sensor_readings.accel_y);
      gyro_x_ = static_cast<int16_t>(pkt.sensor_readings.gyro_z);
      gyro_y_ = -static_cast<int16_t>(pkt.sensor_readings.gyro_x);
      gyro_z_ = -static_cast<int16_t>(pkt.sensor_readings.gyro_y);
    }

    // clockwise rotation
    float new_accel_x_unit = accel_x_unit * cos_value - accel_z_unit * sin_value;
    float new_accel_y_unit = accel_y_unit;
    float new_accel_z_unit = accel_x_unit * sin_value + accel_z_unit * cos_value;
    float new_gyro_x_unit = gyro_x_unit * cos_value - gyro_z_unit * sin_value;
    float new_gyro_y_unit = gyro_y_unit;
    float new_gyro_z_unit = gyro_x_unit * sin_value + gyro_z_unit * cos_value;


    out_pkt.sensor_readings.accel_unit_x = static_cast<int32_t>(new_accel_x_unit);
    out_pkt.sensor_readings.accel_unit_y = static_cast<int32_t>(new_accel_y_unit);
    out_pkt.sensor_readings.accel_unit_z = static_cast<int32_t>(new_accel_z_unit);
    out_pkt.sensor_readings.gyro_unit_x = static_cast<int32_t>(new_gyro_x_unit);
    out_pkt.sensor_readings.gyro_unit_y = static_cast<int32_t>(new_gyro_y_unit);
    out_pkt.sensor_readings.gyro_unit_z = static_cast<int32_t>(new_gyro_z_unit);

    float new_accel_x_ = accel_x_ * cos_value - accel_z_ * sin_value;
    float new_accel_y_ = accel_y_;
    float new_accel_z_ = accel_x_ * sin_value + accel_z_ * cos_value;
    float new_gyro_x_ = gyro_x_ * cos_value - gyro_z_ * sin_value;
    float new_gyro_y_ = gyro_y_;
    float new_gyro_z_ = gyro_x_ * sin_value + gyro_z_ * cos_value;

    out_pkt.sensor_readings.accel_x = static_cast<uint16_t>(new_accel_x_);
    out_pkt.sensor_readings.accel_y = static_cast<uint16_t>(new_accel_y_);
    out_pkt.sensor_readings.accel_z = static_cast<uint16_t>(new_accel_z_);
    out_pkt.sensor_readings.gyro_x = static_cast<uint16_t>(new_gyro_x_);
    out_pkt.sensor_readings.gyro_y = static_cast<uint16_t>(new_gyro_y_);
    out_pkt.sensor_readings.gyro_z = static_cast<uint16_t>(new_gyro_z_);

  } else {
    inno_log_warning("only for Falcon K/K24, unsupported lidar model: %u", pkt.common.lidar_type);
    return false;
  }
  return true;
}
