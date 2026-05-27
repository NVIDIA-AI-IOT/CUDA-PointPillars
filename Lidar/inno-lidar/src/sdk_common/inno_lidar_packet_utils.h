/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#ifndef SDK_COMMON_INNO_LIDAR_PACKET_UTILS_H_
#define SDK_COMMON_INNO_LIDAR_PACKET_UTILS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <string>
#include <chrono>
#if !defined(__MINGW64__) && defined(_WIN32)
#include "utils/getopt_windows.h"
#else
#include <getopt.h>
#endif

#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_other_packet.h"
#include "sdk_common/ring_id_converter_interface.h"
#include "utils/inno_lidar_log.h"

#define CHECK_XYZ_POINTCLOUD_DATA(X)                                                  \
(X == INNO_ITEM_TYPE_XYZ_POINTCLOUD || X == INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD || \
X == INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD || X == INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD || \
X == INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD)

#define CHECK_SPHERE_POINTCLOUD_DATA(X)                                                     \
(X == INNO_ITEM_TYPE_SPHERE_POINTCLOUD || X == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD || \
X == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD || X == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD || \
X == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD || X == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD)

#define CHECK_EN_XYZ_POINTCLOUD_DATA(X)                                                      \
(X == INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD || X == INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD || \
X == INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD || X == INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD)

#define CHECK_EN_SPHERE_POINTCLOUD_DATA(X) \
(X == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD || X == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD || \
X == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD)

#define CHECK_CO_SPHERE_POINTCLOUD_DATA(X) \
(X == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD || X == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD)

// FUNC is in type InnoDataPacketPointsIterCallback
#define ITERARATE_INNO_DATA_PACKET_CPOINTS(FUNC, ctx, packet, count)                                            \
  do {                                                                                                          \
    uint32_t unit_size;                                                                                         \
    uint32_t mr;                                                                                                \
    mr = InnoDataPacketUtils::get_return_times(InnoMultipleReturnMode((packet)->multi_return_mode));            \
    if (mr == 2) {                                                                                              \
      unit_size = sizeof(InnoBlock2);                                                                           \
    } else if (mr == 1) {                                                                                       \
      unit_size = sizeof(InnoBlock1);                                                                           \
    } else {                                                                                                    \
      inno_log_panic("return times of return mode %d is %d?", (packet)->multi_return_mode, mr);                 \
    }                                                                                                           \
    const InnoBlock *block = reinterpret_cast<const InnoBlock *>((packet)->payload);                            \
    for (size_t i = 0; i < (packet)->item_number;                                                               \
         i++, block = reinterpret_cast<const InnoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) { \
      InnoBlockFullAngles full_angles;                                                                          \
      InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header);                                  \
      for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {                                                    \
        for (uint32_t m = 0; m < mr; m++) {                                                                     \
          const InnoChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];                                 \
          FUNC(ctx, (*packet), (*block), pt, full_angles, ch, m);                                               \
          count++;                                                                                              \
        }                                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
  } while (0)

#define ITERARATE_INNO_DATA_PACKET_EN_CPOINTS(FUNC, ctx, packet, count)                                           \
  do {                                                                                                            \
    uint32_t unit_size;                                                                                           \
    uint32_t mr;                                                                                                  \
    mr = InnoDataPacketUtils::get_return_times(InnoMultipleReturnMode((packet)->multi_return_mode));              \
    if (mr == 2) {                                                                                                \
      unit_size = sizeof(InnoEnBlock2);                                                                           \
    } else if (mr == 1) {                                                                                         \
      unit_size = sizeof(InnoEnBlock1);                                                                           \
    } else {                                                                                                      \
      inno_log_panic("return times of return mode %d is %d?", (packet)->multi_return_mode, mr);                   \
    }                                                                                                             \
    const InnoEnBlock *block = reinterpret_cast<const InnoEnBlock *>((packet)->payload);                          \
    for (size_t i = 0; i < (packet)->item_number;                                                                 \
         i++, block = reinterpret_cast<const InnoEnBlock *>(reinterpret_cast<const char *>(block) + unit_size)) { \
      InnoBlockFullAngles full_angles;                                                                            \
      InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header,                                     \
                                                 static_cast<InnoItemType>((packet)->type));                      \
      for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {                                                      \
        for (uint32_t m = 0; m < mr; m++) {                                                                       \
          const InnoEnChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];                                 \
          FUNC(ctx, (*packet), (*block), pt, full_angles, ch, m);                                                 \
          count++;                                                                                                \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  } while (0)

#define ITERARATE_INNO_DATA_PACKET_CO_CPOINTS(FUNC, ctx, packet, count, table)                                    \
  do {                                                                                                            \
    uint32_t unit_size;                                                                                           \
    uint32_t mr;                                                                                                  \
    mr = InnoDataPacketUtils::get_return_times(InnoMultipleReturnMode((packet)->multi_return_mode));              \
    if (mr == 2) {                                                                                                \
      unit_size = sizeof(InnoCoBlock2);                                                                           \
    } else if (mr == 1) {                                                                                         \
      unit_size = sizeof(InnoCoBlock1);                                                                           \
    } else {                                                                                                      \
      inno_log_panic("return times of return mode %d is %d?", (packet)->multi_return_mode, mr);                   \
    }                                                                                                             \
    const InnoCoBlock *block = reinterpret_cast<const InnoCoBlock *>((packet)->payload);                          \
    for (size_t i = 0; i < (packet)->item_number;                                                                 \
         i++, block = reinterpret_cast<const InnoCoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) { \
      InnoCoBlockFullAngles full_angles;                                                                          \
      if (table) {                                                                                                \
        InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header,                                   \
                                                   static_cast<InnoItemType>((packet)->type), table);             \
      }                                                                                                           \
      for (uint32_t ch = 0; ch < kInnoCompactChannelNumber; ch++) {                                               \
        for (uint32_t m = 0; m < mr; m++) {                                                                       \
          if (table && !InnoDataPacketUtils::is_robinw_inside_fov_point(full_angles.angles[ch])) {                \
            continue;                                                                                             \
          }                                                                                                       \
          const InnoCoChannelPoint &pt = block->points[innocoblock_get_idx(ch, m)];                               \
          FUNC(ctx, (*packet), (*block), pt, full_angles, ch, m);                                                 \
          count++;                                                                                                \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  } while (0)
// FUNC is in type InnoDataPacketXYZPointsIterCallback
#define ITERARATE_INNO_DATA_PACKET_XYZ_POINTS(FUNC, ctx, packet)                                    \
  do {                                                                                              \
    const InnoXyzPoint *inno_xyz_point = reinterpret_cast<const InnoXyzPoint *>((packet)->payload); \
    for (size_t i = 0; i < (packet)->item_number; i++, inno_xyz_point++) {                          \
      FUNC(ctx, (*packet), *inno_xyz_point);                                                        \
    }                                                                                               \
  } while (0)

#define ITERARATE_INNO_DATA_PACKET_EN_XYZ_POINTS(FUNC, ctx, packet)                                        \
  do {                                                                                                     \
    const InnoEnXyzPoint *inno_en_xyz_point = reinterpret_cast<const InnoEnXyzPoint *>((packet)->payload); \
    for (size_t i = 0; i < (packet)->item_number; i++, inno_en_xyz_point++) {                              \
      FUNC(ctx, (*packet), *inno_en_xyz_point);                                                            \
    }                                                                                                      \
  } while (0)

// only used for robin
#define DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(type)                              \
  const uint8_t *channel_mapping;                                              \
  int tdc_channel_number;                                                      \
  if (type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD) {                       \
    channel_mapping = &InnoDataPacketUtils::robine_channel_mapping[0];         \
    tdc_channel_number = InnoDataPacketUtils::RobinETDCChannelNumber;          \
  } else if (type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD ||                \
             type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD) {               \
    channel_mapping = &InnoDataPacketUtils::robinw_channel_mapping[0];         \
    tdc_channel_number = InnoDataPacketUtils::RobinWTDCChannelNumber;          \
  } else {                                                                     \
    channel_mapping = &InnoDataPacketUtils::robinelite_channel_mapping[0];     \
    tdc_channel_number = 0;                                                    \
  }

extern "C" {
class InnoBlockAngles {
 public:
  int16_t h_angle;
  int16_t v_angle;
};

/**
 * @brief InnoBlockFullAngles
 */
class InnoBlockFullAngles {
 public:
  InnoBlockAngles angles[kInnoChannelNumber];
};

class InnoCoBlockFullAngles {
 public:
  InnoBlockAngles angles[kMaxReceiverInSet];
};
/**
 * @brief InnoDataPacketUtils
 */
class InnoDataPacketUtils {
  /**
   * @brief InnoDataPacketPointsIterCallback
   */
  typedef void (*InnoDataPacketPointsIterCallback)(void *ctx, const InnoDataPacket &pkt, const InnoBlock &block,
                                                   const InnoChannelPoint &pt, const InnoBlockFullAngles &angle,
                                                   const uint16_t ch, const uint16_t m);

  /**
   * @brief InnoDataPacketXyzPointsIterCallback
   */
  typedef void (*InnoDataPacketXyzPointsIterCallback)(void *ctx, const InnoDataPacket &pkt, const InnoXyzPoint &pt);

 private:
  static int init_;
  static int v_angle_offset_[INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD][kInnoChannelNumber];
  static const uint32_t kTableShift_ = 9;
  static const uint32_t kTableStep_ = 1 << kTableShift_;
  static const uint32_t kTableHalfStep_ = 1 << (kTableShift_ - 1);
  static const uint32_t kTableMask_ = kTableStep_ - 1;
  static const uint32_t kVTableSizeBits_ = 4;
  static const uint32_t kHTableSizeBits_ = 6;
  static const uint32_t kVTableSize_ = 1 << kVTableSizeBits_;
  static const uint32_t kHTableSize_ = 1 << kHTableSizeBits_;
  static const uint32_t kVTableEffeHalfSize_ = 6;
  static const uint32_t kHTableEffeHalfSize_ = 22;
  static const uint32_t kXZSize_ = 2;
  static int8_t nps_adjustment_[kVTableSize_][kHTableSize_][kInnoChannelNumber][kXZSize_];
  static const double kAdjustmentUnitInMeter_;
  static const double kAdjustmentUnitInMeterRobin_;
  static const uint32_t kHRobinTableEffeHalfSize_ = 27;
  static const uint32_t kXYZSize_ = 3;
  static const uint32_t kRobinWScanlines_ = 192;
  static const uint32_t kRobinEScanlines_ = 128;
  static const uint32_t kRobinWDistSize_ = 8;
  static const uint32_t kRobinDist_[kRobinWDistSize_-1];
  static const uint32_t kRobinEliteScanlines_ = 96;
  static constexpr double kUsInSecond = 1000000.0;
  static constexpr double k10UsInSecond = 100000.0;
  static int8_t vehicle_coordinate_;
  static int8_t robinw_nps_adjustment_[kRobinWScanlines_][kHTableSize_][kXYZSize_];
  static int8_t robinw_nps_adjustment_pin_[kRobinWDistSize_][kRobinWScanlines_][kHTableSize_][kXYZSize_];
  static int8_t robine_nps_adjustment_[kRobinEScanlines_][kHTableSize_][kXYZSize_];
  static int8_t robinelite_nps_adjustment_[kRobinEliteScanlines_][kHTableSize_][kXYZSize_];

 private:
  /**
   * @brief Ajust x z
   * @param angles Block angles
   * @param ch     Channel
   * @param x      X
   * @param z      Z
   */
  static void lookup_xz_adjustment_(const InnoBlockAngles &angles, uint32_t ch, double *x, double *z);

  /**
   * @brief Ajust x y z for Robin
   * @param angles Block angles
   * @param ch     Channel
   * @param x      X
   * @param Y      Y
   * @param z      Z
   */
  static void lookup_xyz_adjustment_(const InnoBlockAngles &angles, uint32_t scan_id, uint32_t radius_unit,
                                     uint32_t firing, double adj[], InnoItemType type);

 public:
  static const uint32_t RobinWTDCChannelNumber = 48;
  static const uint32_t RobinETDCChannelNumber = 32;
  static const uint8_t robinw_channel_mapping[48];
  // use first 32 entries for robine
  static const uint8_t robine_channel_mapping[48];
  static const uint8_t robinelite_channel_mapping[96];

 public:
  /**
   * @brief Initialize NPS adjustment table, angle offset table and etc.
   * @return Return 0 for success, others for error
   */
  static int init_f(void);
  static int init_f_robin(void);
  static int init_f_falcon(void);
  static void init_robine_nps_adjustment_();
  static void init_robinw_nps_adjustment_();
  static void init_robinw_nps_adjustment_pin_();

  static void init_robinelite_nps_adjustment_();
  static void set_vehicle_coordinate(int8_t value) {
    vehicle_coordinate_ = value;
  }

  /**
   * @brief Populate the points angles from the block header
   * @param full The angles to be populated
   * @param b    The block header
   */
  static inline void get_block_full_angles(InnoBlockFullAngles *full, const InnoBlockHeader &b,
                                           InnoItemType type = INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    full->angles[0].h_angle = b.h_angle;
    full->angles[0].v_angle = b.v_angle;
    full->angles[1].h_angle = b.h_angle + b.h_angle_diff_1;
    full->angles[1].v_angle = b.v_angle + b.v_angle_diff_1 + v_angle_offset_[type][1];
    full->angles[2].h_angle = b.h_angle + b.h_angle_diff_2;
    full->angles[2].v_angle = b.v_angle + b.v_angle_diff_2 + v_angle_offset_[type][2];
    full->angles[3].h_angle = b.h_angle + b.h_angle_diff_3;
    full->angles[3].v_angle = b.v_angle + b.v_angle_diff_3 + v_angle_offset_[type][3];
  }

  /**
   * @brief Populate the points angles from the block header
   * @param full The angles to be populated
   * @param b    The block header
   */
  static inline void get_block_full_angles(InnoBlockFullAngles *full, const InnoEnBlockHeader &b,
                                           InnoItemType type) {
    full->angles[0].h_angle = b.h_angle;
    full->angles[0].v_angle = b.v_angle;
    full->angles[1].h_angle = b.h_angle + b.h_angle_diff_1;
    full->angles[1].v_angle = b.v_angle + b.v_angle_diff_1 + v_angle_offset_[type][1];
    full->angles[2].h_angle = b.h_angle + b.h_angle_diff_2;
    full->angles[2].v_angle = b.v_angle + b.v_angle_diff_2 + v_angle_offset_[type][2];
    full->angles[3].h_angle = b.h_angle + b.h_angle_diff_3;
    full->angles[3].v_angle = b.v_angle + b.v_angle_diff_3 + v_angle_offset_[type][3];
  }

  static inline void get_block_full_angles(InnoCoBlockFullAngles *full, const InnoCoBlockHeader &b, InnoItemType type,
                                           const char *anglehv_table) {
    int polygon_mod = b.p_angle;
    int facet_num = b.facet;
    int set_num = b.scan_id;
    int h_offset_total = polygon_mod - kPolygonMinAngle;
    if (h_offset_total < 0) {
      h_offset_total = 0;
    }

    int h_idx = h_offset_total >> kEncoderTableShift;
    int h_offset = h_offset_total & kEncoderTableMask;
    int h_offset2 = kEncoderTableStep - h_offset;

    if (h_idx > signed(kPolygonTableSize - 2)) {
      h_idx = kPolygonTableSize - 2;
    }

    int max_set_number = kInnoRobinWMaxSetNumber;
    AngleHV *b1, *b2;
    if (type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
      using RELiteAnglehvTable =
        AngleHV(&)[kPolygonMaxFacets][kPolygonTableSize][kInnoRobinELiteMaxSetNumber][kMaxReceiverInSet];
      RELiteAnglehvTable table = reinterpret_cast<RELiteAnglehvTable>(*const_cast<char *>((anglehv_table)));
      max_set_number = kInnoRobinELiteMaxSetNumber;
      b1 = &table[facet_num][h_idx][set_num][0];
    } else {
      using RWAngleHvTable =
        AngleHV(&)[kPolygonMaxFacets][kPolygonTableSize][kInnoRobinWMaxSetNumber][kMaxReceiverInSet];
      RWAngleHvTable table = reinterpret_cast<RWAngleHvTable>(*const_cast<char *>((anglehv_table)));
      b1 = &table[facet_num][h_idx][set_num][0];
    }

    for (int r = 0; r < kMaxReceiverInSet; r++, b1++) {
      b2 = b1 + max_set_number * kMaxReceiverInSet;

      int32_t v;
      v = b1->h * h_offset2 + b2->h * h_offset;
      full->angles[r].h_angle = v >> kEncoderTableShift;

      v = b1->v * h_offset2 + b2->v * h_offset;
      full->angles[r].v_angle = v >> kEncoderTableShift;
    }
  }

  /**
   * @brief check if the point is inside the field of view (FOV)
   * only vilid inside the FOV
   * @param angle point angle
   * @return Return true if the point is valid, false otherwise
   */
  static inline bool is_robinw_inside_fov_point(InnoBlockAngles angle) {
    // FOV: -60 ~ 60, -55 ~ 15
    static int fov_top_left_angle = -60.0 * kInnoAngleUnitPerDegree;
    static int fov_top_right_angle = 60.0 * kInnoAngleUnitPerDegree;
    static int fov_top_low_angle = -55.0 * kInnoAngleUnitPerDegree;
    static int fov_top_high_angle = 15.0 * kInnoAngleUnitPerDegree;

    if (angle.h_angle < fov_top_left_angle || angle.h_angle > fov_top_right_angle ||
        angle.v_angle < fov_top_low_angle || angle.v_angle > fov_top_high_angle) {
      return false;
    } else {
      return true;
    }
  }

  /**
   * @brief Get the block size in bytes and number of returns in the pkt
   * @param pkt DataPacket
   * @param block_size_in_byte Return size (in bytes) of block in the pkt
   * @param number_return Return number of returns in the pkt
   * @return Void
   */
  static inline void get_block_size_and_number_return(const InnoDataPacket &pkt, uint32_t *block_size_in_byte,
                                                      uint32_t *number_return) {
    inno_log_verify(CHECK_SPHERE_POINTCLOUD_DATA(pkt.type), "invalid pkt type %u", pkt.type);
    if (CHECK_EN_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
          pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        *block_size_in_byte = sizeof(InnoEnBlock2);
        *number_return = 2;
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        *block_size_in_byte = sizeof(InnoEnBlock1);
        *number_return = 1;
      } else {
        inno_log_verify(false, "invalid return mode %u", pkt.multi_return_mode);
      }
    } else if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
          pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        *block_size_in_byte = sizeof(InnoCoBlock2);
        *number_return = 2;
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        *block_size_in_byte = sizeof(InnoCoBlock1);
        *number_return = 1;
      } else {
        inno_log_verify(false, "invalid return mode %u", pkt.multi_return_mode);
      }
    } else {
      if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
          pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        *block_size_in_byte = sizeof(InnoBlock2);
        *number_return = 2;
      } else if (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        *block_size_in_byte = sizeof(InnoBlock1);
        *number_return = 1;
      } else {
        inno_log_verify(false, "invalid return mode %u", pkt.multi_return_mode);
      }
    }
    return;
  }

  /**
   * @brief Calculate x/y/z coordinate based on angle and radius.
    *radius_meter = radius_unit * kMeterPerInnoDistanceUnit;
    double va = angles.v_angle * kRadPerInnoAngleUnit;
    double ha = angles.h_angle * kRadPerInnoAngleUnit;
    double t = *radius_meter * cos(va);
    *x_meter = *radius_meter * sin(va);
    *y_meter = t * sin(ha);
    *z_meter = t * cos(ha);
    Fine tune x_meter and z_meter in the end
   * @param angles Angles
   * @param radius_unit Radius in InnoDistanceUnit
   * @param channel Channel
   * @param result Store result.
   * @return Void
   */
  static void get_xyzr_meter(const InnoBlockAngles angles, const uint32_t radius_unit, const uint32_t channel,
                             InnoXyzrD *result, InnoItemType type = INNO_ITEM_TYPE_SPHERE_POINTCLOUD,
                             uint32_t firing = 1, bool long_distance_mode = false);

  /**
   * @brief convert an InnoChannelPoint in a block to
            an InnoXyzPoint
   * @param block Block header
   * @param cp Source channel point
   * @param angles Angles
   * @param channel Channel
   * @param pt Destination InnoXyzPoint
   * @return number of points in the data packet. -1 if invalid item type
   */
  static inline void get_xyz_point(const InnoBlockHeader &block, const InnoChannelPoint &cp,
                                   const InnoBlockAngles angles, const uint32_t channel, InnoXyzPoint *pt,
                                   bool long_distance_mode = false) {
    InnoXyzrD xyzr;
    get_xyzr_meter(angles, cp.radius, channel, &xyzr, INNO_ITEM_TYPE_SPHERE_POINTCLOUD, 1, long_distance_mode);
    if (vehicle_coordinate_ == 1) {
      pt->x = xyzr.z;
      pt->y = -xyzr.y;
      pt->z = xyzr.x;
    } else {
      pt->x = xyzr.x;
      pt->y = xyzr.y;
      pt->z = xyzr.z;
    }
    pt->radius = xyzr.radius;
    pt->ts_10us = block.ts_10us;
    pt->scan_idx = block.scan_idx;
    pt->scan_id = block.scan_id;
    pt->in_roi = block.in_roi;
    pt->facet = block.facet;
    pt->reserved_flags = block.reserved_flags;
    pt->refl = cp.refl;
    pt->type = cp.type;
    pt->elongation = cp.elongation;
    pt->channel = channel;
    pt->is_2nd_return = cp.is_2nd_return;
  }

  /**
     * @brief convert an InnoChannelPoint in a block to
              an InnoXyzPoint
     * @param block Block header
     * @param cp Source channel point
     * @param angles Angles
     * @param channel Channel
     * @param pt Destination InnoXyzPoint
     * @return number of points in the data packet. -1 if invalid item type
     */
  static inline void get_xyz_point(const InnoEnBlockHeader &block, const InnoEnChannelPoint &cp,
                                   const InnoBlockAngles angles, const uint32_t channel, InnoEnXyzPoint *pt,
                                   InnoItemType type) {
    InnoXyzrD xyzr;
    uint32_t scan_id = 0;
    if (type == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD) {
      scan_id = block.scan_id;
      get_xyzr_meter(angles, cp.radius, channel, &xyzr, type);
    } else if (type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD || type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD) {
      DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(type);
      int index = block.scan_id * 4 + channel;
      scan_id = channel_mapping[index] + block.facet * tdc_channel_number;
      get_xyzr_meter(angles, cp.radius, scan_id, &xyzr, type);
    }
    pt->x = xyzr.x;
    pt->y = xyzr.y;
    pt->z = xyzr.z;
    pt->radius = xyzr.radius;
    pt->ts_10us = block.ts_10us;
    pt->scan_idx = block.scan_idx;
    pt->scan_id = scan_id;
    pt->in_roi = block.in_roi;
    pt->facet = block.facet;
    pt->reflectance = cp.reflectance;
    pt->intensity = cp.intensity;
    pt->channel = channel;
    pt->is_2nd_return = cp.is_2nd_return;
    pt->firing = cp.firing;
  }

  /**
       * @brief convert an InnoCoChannelPoint in a block to
                an InnoEnXyzPoint
       * @param block Block header
       * @param cp Source channel point
       * @param angles Angles
       * @param channel Channel
       * @param pt Destination InnoXyzPoint
       * @return number of points in the data packet. -1 if invalid item type
       */
  static inline void get_xyz_point(const InnoCoBlockHeader &block, const InnoCoChannelPoint &cp,
                                   const InnoBlockAngles angles, const uint32_t channel, InnoEnXyzPoint *pt,
                                   InnoItemType type) {
    InnoXyzrD xyzr;
    uint32_t scan_id = 0;
    DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(type);
    int index = block.scan_id * kMaxReceiverInSet + channel;
    scan_id = channel_mapping[index] + block.facet * tdc_channel_number;
    get_xyzr_meter(angles, cp.radius, scan_id, &xyzr, type, cp.firing);
    if (vehicle_coordinate_ == 1) {
      pt->x = xyzr.z;
      pt->y = -xyzr.y;
      pt->z = xyzr.x;
    } else {
      pt->x = xyzr.x;
      pt->y = xyzr.y;
      pt->z = xyzr.z;
    }
    pt->radius = xyzr.radius;
    pt->ts_10us = block.ts_10us;
    pt->scan_idx = block.scan_idx;
    pt->scan_id = scan_id;
    pt->in_roi = block.in_roi;
    pt->facet = block.facet;
    pt->reflectance = cp.refl;
    pt->intensity = cp.refl;
    pt->channel = channel;
    pt->is_2nd_return = cp.is_2nd_return;
    pt->firing = cp.firing;
  }

  /**
   * @brief Enumerate each point in the data packet and make callback,
            only apply to INNO_ITEM_TYPE_SPHERE_POINTCLOUD
   * @param pkt Data packet
   * @param callback Callback
   * @param ctx Callback context
   * @return number of points in the data packet.
   */
  static inline ssize_t iterate_cpoints(const InnoDataPacket &pkt, InnoDataPacketPointsIterCallback callback,
                                        void *ctx) {
    inno_log_verify(pkt.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD, "invalid pkt type %u", pkt.type);
    size_t pcount = 0;
    ITERARATE_INNO_DATA_PACKET_CPOINTS(callback, ctx, &pkt, pcount);
    return pcount;
  }

  /**
   * @brief Enumerate each point in the data packet and make callback,
            only apply to XYZ_POINTCLOUD
   * @param pkt Data packet
   * @param callback Callback
   * @param ctx Callback context
   * @return number of points in the data packet.
   */
  static inline ssize_t iterate_xyz_points(const InnoDataPacket &pkt, InnoDataPacketXyzPointsIterCallback callback,
                                           void *ctx) {
    inno_log_verify(CHECK_XYZ_POINTCLOUD_DATA(pkt.type), "invalid pkt type %u", pkt.type);
    ITERARATE_INNO_DATA_PACKET_XYZ_POINTS(callback, ctx, &pkt);
    return pkt.item_number;
  }

  /**
   * @brief Return the data packet size according to the item type, count
            and return mode.
   * @param type Data packet type
   * @param item_count Item count
   * @param mode Multi-return mode
   * @return size of data packet in bytes. Return 0 if the type is invalid.
   */
  static inline size_t get_data_packet_size(InnoItemType type, uint32_t item_count, InnoMultipleReturnMode mode) {
    size_t unit_size;

    switch (type) {
    case INNO_ITEM_TYPE_SPHERE_POINTCLOUD:
      if (mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        unit_size = sizeof(InnoBlock1);
      } else if (mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        unit_size = sizeof(InnoBlock2);
      } else {
        return 0;
      }
      break;
    case INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD:
    case INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD:
      if (mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        unit_size = sizeof(InnoCoBlock1);
      } else if (mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        unit_size = sizeof(InnoCoBlock2);
      } else {
        return 0;
      }
      break;
    case INNO_ITEM_TYPE_XYZ_POINTCLOUD:
      unit_size = sizeof(InnoXyzPoint);
      break;
    case INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD:
    case INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD:
    case INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD:
      if (mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        unit_size = sizeof(InnoEnBlock1);
      } else if (mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        unit_size = sizeof(InnoEnBlock2);
      } else {
        return 0;
      }
      break;
    case INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD:
    case INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD:
      unit_size = sizeof(InnoEnXyzPoint);
      break;
    default:
      inno_log_verify(false, "bad type = %d", type);
    }
    return sizeof(InnoDataPacket) + item_count * unit_size;
  }

  /**
   * @brief Get true return times of a return mode, we assert the input mode is valid
   * If new return mode added, we may need to refine this function
   * @param mode InnoMultipleReturnMode
   * @return return times
   */
  static inline int get_return_times(const InnoMultipleReturnMode mode) {
    return mode >= INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ? 2 : 1;
  }

  /**
   * @brief Convert INNO_ITEM_TYPE_SPHERE_POINTCLOUD points the source
            data packet to INNO_ITEM_TYPE_XYZ_POINTCLOUD
   * @param src Source data packet, type must be
            INNO_ITEM_TYPE_SPHERE_POINTCLOUD
   * @return converted InnoDataPacket allocated by malloc.
             return NULL if malloc failed or invalid packet
   */
  static InnoDataPacket *convert_to_xyz_pointcloud_malloced(const InnoDataPacket &src,
                                                            RingIdConverterInterface *ring_id_converter = NULL,
                                                            const char *hvangle_table = NULL);
  /**
   * @brief Convert INNO_ITEM_TYPE_SPHERE_POINTCLOUD points the source
            data packet to INNO_ITEM_TYPE_XYZ_POINTCLOUD and add them
            to the destination data packet.
   * @param src Source data packet
   * @param dest Destination data packet
   * @param dest_size Max size of the destination data packet
   * @param crc_disable  
   * @return false if the pkt is invalid, true otherwise
   */
  static bool convert_to_xyz_pointcloud(const InnoDataPacket &src, InnoDataPacket *dest, size_t dest_size,
                                        bool crc_disable, RingIdConverterInterface *ring_id_converter = NULL,
                                        const char *hvangle_table = NULL);

  /**
   * @brief Sanity check the integrity of a InnoDataPacketGet.
   * @param pkt DataPacket
   * @param size Size of pkt if it is received from network or
   *        read from file
   * @return false if the pkt is invalid, true otherwise
   */
  static bool check_data_packet(const InnoDataPacket &pkt, size_t size);

  /**
   * @brief Sanity check the integrity of a InnoDataPacketGet.
   * @param pkt StatusPacket
   * @param size Size of pkt if it is received from network or
   *        read from file, 0 means don't check size
   * @return false if the pkt is invalid, true otherwise
   */
  static bool check_status_packet(const InnoStatusPacket &pkt, size_t size);

  /**
   * @brief check if occur fault
   * @param pkt StatusPacket
   * @return false if not occur fault, true otherwise
   */
  static bool check_status_packet_fault(const InnoStatusPacket &pkt);

  /**
   * @brief Correct IMU status data based on IMU position
   * @param pkt InnoStatusPacket
   * @param out_pkt Destination status packet
   * @param is_wgs is_wgs IMU physical direction, falcon-k: false falcon-k24: true
   * @return false if not occur fault, true otherwise
   */
  static bool fix_imu_status(const InnoStatusPacket &pkt, InnoStatusPacket &out_pkt, bool is_wgs);

  /**
   * @brief check if occur fault specified by fault id
   * @param pkt StatusPacket
   * @param fid fault id
   * @return false if not occur fault, true otherwise
   */
  static bool check_status_packet_fault_id(const InnoStatusPacket &pkt, int fid);

  /**
   * @brief InnoStatusPacket formatted output.
   * @param pkt StatusPacket
   * @param buffer      Buffer to store status packet information
   * @param buffer_size Buffer size
   * @return Upon successful return, these functions return the
   *         number of characters printed (excluding the null byte
   *         used to end output to strings).
   *         If an output error is encountered, a negative value is returned.
   */
  static int printf_status_packet(const InnoStatusPacket &pkt, char *buffer, size_t buffer_size);

  /**
   * @brief init Raw4UdpHeader by net buffer.
   * @param buffer      Buffer received from network
   * @param buffer_size Buffer size
   * @param header      Raw4UdpHeader to be filled
   * @return true: successful, false: failed.
   */
  static bool raw4_header_from_net(const char *buffer, size_t buffer_size, Raw4UdpHeader *header);

  /**
   * @brief fill net buffer by init Raw4UdpHeader.
   * @param header       Raw4UdpHeader
   * @param buffer       Buffer to store Raw4UdpHeader information
   * @param buffer_size  Buffer size
   * @return true: successful, false: failed.
   */
  static bool raw4_header_to_net(const Raw4UdpHeader &header, char *buffer, size_t buffer_size);

  /**
   * @brief Get the number of points in the pkt, have to iterate each
   *        block for SPHERE_POINTCLOUD pkt.
   * @param pkt DataPacket
   * @return number of points in the pkt
   */
  static inline uint32_t get_points_count(const InnoDataPacket &pkt) {
#define ADD_FN(ctx, p, b, pt, full_angles, ch, m)                                                                      \
  do {                                                                                                                 \
    if (pt.radius) {                                                                                                   \
      item_count++;                                                                                                    \
    }                                                                                                                  \
  } while (0)
    if (CHECK_XYZ_POINTCLOUD_DATA(pkt.type)) {
      return pkt.item_number;
    } else if (pkt.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      uint32_t item_count = 0;
      uint32_t dummy_count = 0;
      ITERARATE_INNO_DATA_PACKET_CPOINTS(ADD_FN, NULL, &pkt, dummy_count);
      return item_count;
    } else if (CHECK_EN_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      uint32_t item_count = 0;
      uint32_t dummy_count = 0;
      ITERARATE_INNO_DATA_PACKET_EN_CPOINTS(ADD_FN, NULL, &pkt, dummy_count);
      return item_count;
    } else if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      uint32_t item_count = 0;
      uint32_t dummy_count = 0;
      ITERARATE_INNO_DATA_PACKET_CO_CPOINTS(ADD_FN, NULL, &pkt, dummy_count, nullptr);
      return item_count;
    } else {
      inno_log_verify(false, "invalid type %u", pkt.type);
    }
  }

  /**
   * @brief Get the number of points in the pkt which has 2nd return
   * @param pkt DataPacket
   * @return Return number of points in the pkt which has 2nd return
   */
  static inline uint64_t get_points_count_2nd_return(const InnoDataPacket &pkt) {
    if (CHECK_XYZ_POINTCLOUD_DATA(pkt.type)) {
      uint64_t cnt = 0;
      if (pkt.type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
      const InnoXyzPoint *xyz_points = reinterpret_cast<const InnoXyzPoint *>(pkt.payload);
      for (auto i = 0; i < pkt.item_number; ++i) {
        cnt += xyz_points[i].is_2nd_return;
      }
      } else {
      const InnoEnXyzPoint *xyz_points = reinterpret_cast<const InnoEnXyzPoint *>(pkt.payload);
      for (auto i = 0; i < pkt.item_number; ++i) {
        cnt += xyz_points[i].is_2nd_return;
      }
      }

      return cnt;
    } else {
      inno_log_verify(false, "invalid type %u", pkt.type);
    }
  }

  /**
   * @brief Get max number of points in the pkt. For
   *        SPHERE_POINTCLOUD pkt it is calculated
   *        based on number of blocks and number of returns.
   * @param pkt DataPacket
   * @return max number of points in the pkt
   */
  static inline uint32_t get_max_points_count(const InnoDataPacket &pkt) {
    if (CHECK_XYZ_POINTCLOUD_DATA(pkt.type)) {
      return pkt.item_number;
    } else if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      return pkt.item_number * kInnoCompactChannelNumber *
             (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE ? 1 : 2);
    } else if (CHECK_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      return pkt.item_number * kInnoChannelNumber * (pkt.multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE ? 1 : 2);
    } else {
      inno_log_verify(false, "invalid type %u", pkt.type);
    }
  }
};

/**
 * @brief InnoPacketReader
 */
class InnoPacketReader {
 public:
  /**
   * @brief Get packet crc32
   * @param header InnoCommonHeader
   * @return Return crc32
   */
  static uint32_t calculate_packet_crc32(const InnoCommonHeader *header);

  /**
   * @brief Set packet crc32
   * @param header InnoCommonHeader
   */
  static void set_packet_crc32(InnoCommonHeader *header);

  /**
   * @brief Verify packet crc32
   * @param header InnoCommonHeader
   * @return Return true if crc32 is correct, otherwise return false
   */
  static bool verify_packet_crc32(const InnoCommonHeader *header);

  /**
   * @brief Get http buffer crc32
   * @param buffer    Buffer received from network
   * @param length    Buffer length
   * @param append    True to calculate crc32 with separator
   * @return Return crc32
   */
  static uint32_t calculate_http_crc32(const char *buffer, uint32_t length, bool append = false);

  /**
   * @brief Verify http buffer crc32
   * @param buffer   Buffer received from network
   * @param url      Http url
   * @return  Return 0 if crc32 is correct, others for error
   */
  static int verify_http_crc32(const char *buffer, const char *url);
};
};

/**
 * @brief InnoSummaryPackage
 */
class InnoSummaryPackage {
 public:
  /*
  * @brief Collect statistics about Miss frames and Miss sub frames
           based on data stream.
    @param pkt the stream of innoDataPacket
    @return 0 means success
            -1 means miss sub frame
            -2 means miss frame
  */
  InnoSummaryPackage() {
    current_frame_ = 0;
    expect_frame_ = 0;
    current_sub_frame_ = 0;
    expect_sub_frame_ = -1;
    next_new_frame_ = false;
    frame_counter_ = 0;
    miss_frame_counter_ = 0;
    miss_sub_frame_gap_counter_ = 0;
    miss_sub_frame_last_one_counter_ = 0;
    miss_sub_frame_except_last_one_counter_ = 0;
    empty_sub_frame_counter_ = 0;
    last_print_time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
  }
  ~InnoSummaryPackage() {
  }
  /**
   * @brief Summary frame count, miss frame count, miss sub frame count and etc. from data package
   * @param pkt DataPacket
   * @return Return 0 if no frame missing, otherwise return error code
   */
  int summary_data_package(const InnoDataPacket &pkt);

  /**
   * @brief Get count of missed frames
   * @return Return count of missed frames
   */
  uint64_t get_miss_frame_count(void) {
    return miss_frame_counter_;
  }

  /**
   * @brief Get count of gaps those have missed sub frames
   * @return Return count of gaps those have missed sub frames
   */
  uint64_t get_miss_sub_frame_gap_count(void) {
    return miss_sub_frame_gap_counter_;
  }

  /**
   * @brief Get total receive inno data packet
   * @return Return count of missed sub frames
   */
  uint64_t get_total_receive_inno_packet(void) {
    return total_receive_packet_;
  }

  /**
   * @brief Get expect receive inno data packet
   * @return Return count of missed sub frames
   */
  uint64_t get_expect_receive_inno_packet(void) {
    return highest_seq_num_received_ - base_seq_num_received_ + 1;
  }
  /**
   * @brief Get frame count
   * @return Return frame count
   */
  uint64_t get_frame_count(void) {
    if (frame_counter_ > 0) {
      return frame_counter_;
    } else {
      return 0;
    }
  }

  /**
   * @brief Get count of sub frames whose items count is 0
   * @return Return count of sub frames whose items count is 0
   */
  uint64_t get_empty_sub_frame_count(void) {
    return empty_sub_frame_counter_;
  }

  /**
   * @brief Get count of frames whose is_last_sub_frame flag is missing
   * @param Return count of frames whose is_last_sub_frame flag is missing
   */
  uint64_t get_miss_sub_frame_last_one_count(void) {
    return miss_sub_frame_last_one_counter_;
  }

  /**
   * @brief Get the count of missed sub frames
   * @return Return the count of missed sub frames
   */
  uint64_t get_empty_sub_frame_except_last_one_count(void) {
    return miss_sub_frame_except_last_one_counter_;
  }

  /**
   * @brief check if the packet is the newer one
   * @param pre_seq previous packet sequence number
   * @param seq current packet sequence number
   * @return true if the packet is the newer one, otherwise return false
   */
  bool is_newer_packet(uint16_t pre_seq, uint16_t seq) {
    // a 'less-than' on 16-bit sequence numbers
    int diff = seq - pre_seq;

    if (diff > 0) {
      return (diff < 0x8000);
    } else if (diff < 0) {
      return (diff < -0x8000);
    } else {  // diff == 0
      return false;
    }
  }

  void inno_data_packet_receive_stats(const InnoDataPacket &pkt);
  void init_seq_num(uint16_t seqNum);
  void print_inno_data_packet_loss_rate(int print_interval_ms = 30000);

 private:
  const uint64_t kPrintLossRateInterval_ = 30000;  //  30s
  int64_t current_frame_ = 0;
  int64_t expect_frame_ = 0;
  uint32_t current_sub_frame_ = 0;
  int32_t expect_sub_frame_ = 0;
  bool next_new_frame_;
  uint64_t frame_counter_ = 0;
  uint64_t miss_frame_counter_ = 0;
  uint64_t miss_sub_frame_gap_counter_ = 0;
  uint64_t miss_sub_frame_last_one_counter_ = 0;
  uint64_t miss_sub_frame_except_last_one_counter_ = 0;
  uint64_t empty_sub_frame_counter_ = 0;

  uint64_t highest_seq_num_received_ = 0;
  uint64_t base_seq_num_received_ = 0;
  uint64_t total_receive_packet_ = 0;
  uint64_t expect_receive_packet_ = 0;
  uint64_t last_recieve_total_packet_count_ = 0;
  uint64_t last_recieve_expect_packet_count_ = 0;
  uint64_t last_print_time_ = 0;
  bool init_seq_num_ = false;
};

#endif  // SDK_COMMON_INNO_LIDAR_PACKET_UTILS_H_
