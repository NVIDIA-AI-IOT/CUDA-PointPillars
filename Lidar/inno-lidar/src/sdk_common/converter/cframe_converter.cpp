/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_common/converter/cframe_converter.h"

#include <stdio.h>
#include <stdlib.h>

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace innovusion {

CframeConverter::CframeConverter() {
  current_cframe_id_ = -1;
  current_cframe_ = &cframe0_;
  radius_shift_ = 0;
  angle_shift_ = 0;
  for (uint32_t i = 0; i < 10; i++) {
    if ((cpoint_distance_unit_per_meter_c << i) == kInnoDistanceUnitPerMeter200) {
      radius_shift_ = i;
    }
    if ((cpoint_distance_unit_per_meter_c << i) == kInnoDistanceUnitPerMeter400) {
      en_radius_shift_ = i;
    }
    if ((cpoint_angle_unit_per_PI_c << i) == kInnoAngleUnitPerPiRad) {
      angle_shift_ = i;
    }
  }
}

CframeConverter::~CframeConverter() {
}

inno_cframe_header *CframeConverter::close_current_frame() {
  if (current_cframe_id_ != -1 &&
      current_cframe_ &&
      current_cframe_->item_number > 0) {
    inno_cframe_header *ret = current_cframe_;
    current_cframe_id_ = -1;
    return ret;
  } else {
    return NULL;
  }
}

inno_cframe_header *CframeConverter::add_data_packet(const InnoDataPacket *pkt,
                                                     int interval, const char *table) {
  inno_log_verify(pkt, "pkt");
  if (!CHECK_SPHERE_POINTCLOUD_DATA(pkt->type) && !CHECK_XYZ_POINTCLOUD_DATA(pkt->type)) {
    return NULL;
  }
  inno_cframe_header *ret = NULL;
  if (ssize_t(pkt->idx) != current_cframe_id_) {
    if (current_cframe_id_ < 0) {
      // do nothing
    } else {
      // close the previous frame
      current_cframe_->ts_us_end = pkt->common.ts_start_us;
      if (interval == 0 || current_cframe_id_ % interval == 0) {
        ret = current_cframe_;
      }
    }
    current_cframe_id_ = pkt->idx;
    start_new_current_cframe_(pkt);
  }
  if (interval == 0 || current_cframe_id_ % interval == 0) {
    update_current_cframe_v2_(pkt,table);
  }
  return ret;
}

void CframeConverter::start_new_current_cframe_(const InnoDataPacket *pkt) {
  current_cframe_ = current_cframe_ == &cframe1_ ?
                    &cframe0_ :
                    &cframe1_;
  memset(current_cframe_, 0, sizeof(*current_cframe_));

  current_cframe_->version = cframe_version_c;
  current_cframe_->flags = 0;
  current_cframe_->checksum = 0;
  current_cframe_->idx = pkt->idx;
  current_cframe_->sub_idx = 0;
  current_cframe_->sub_seq = 0;
  current_cframe_->ts_us_start = pkt->common.ts_start_us;
  current_cframe_->ts_us_end = pkt->common.ts_start_us;  // need to update
  if (CHECK_SPHERE_POINTCLOUD_DATA(pkt->type)) {
    current_cframe_->type = INNO_CFRAME_CPOINT;
  } else if (CHECK_XYZ_POINTCLOUD_DATA(pkt->type)) {
    current_cframe_->type = INNO_CFRAME_POINT;
  } else {
    inno_log_verify(false, "invalid type %d", pkt->type);
  }
  current_cframe_->topic = 0;
  current_cframe_->item_number = 0;  // need to update
  current_cframe_->conf_level = 255;
  current_cframe_->source_id = 0;
  if(pkt->type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD ||
     pkt->type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD) {
    current_cframe_->lidar_type = INNO_LIDAR_TYPE_ROBINW;
  } else if (pkt->type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD) {
    current_cframe_->lidar_type = INNO_LIDAR_TYPE_ROBINE;
  }
}

inline void CframeConverter::add_cpoint_to_current_cframe_(
    void *ctx,
    const InnoDataPacket &pkt,
    const InnoBlock &block,
    const InnoChannelPoint &pt,
    const InnoBlockFullAngles &full_angles,
    const uint16_t ch,
    const uint16_t m) {
  size_t pcount = current_cframe_->item_number;
  inno_cpoint &cpoint = current_cframe_->cpoints[pcount];

  if (pt.radius > 0 && pcount < kMaxNumberInCframe) {
    // xxx todo: hard coded
    cpoint.radius = pt.radius >> radius_shift_;
    cpoint.h_angle = full_angles.angles[ch].h_angle >> angle_shift_;
    cpoint.v_angle = full_angles.angles[ch].v_angle >> angle_shift_;
    cpoint.ts_100us = block.header.ts_10us / 10 +
                      (pkt.common.ts_start_us -
                       current_cframe_->ts_us_start) / 100;
    cpoint.scan_id = block.header.scan_id;
    cpoint.scan_idx = block.header.scan_idx;
    cpoint.flags = ch;
    if (block.header.in_roi == 0x3) {
      cpoint.flags |= 0x8;
    }
    if (m == 1) {
      cpoint.flags |= 0x4;
    }
    cpoint.ref = pt.refl;
    current_cframe_->item_number += 1;
  }
  return;
}

inline void CframeConverter::add_cpoint_to_current_cframe_(void *ctx, const InnoDataPacket &pkt,
                                                           const InnoEnBlock &block, const InnoEnChannelPoint &pt,
                                                           const InnoBlockFullAngles &full_angles, const uint16_t ch,
                                                           const uint16_t m) {
  size_t pcount = current_cframe_->item_number;
  inno_cpoint &cpoint = current_cframe_->cpoints[pcount];

  if (pt.radius > 0 && pcount < kMaxNumberInCframe) {
    // xxx todo: hard coded
    cpoint.radius = pt.radius >> en_radius_shift_;
    cpoint.h_angle = full_angles.angles[ch].h_angle >> angle_shift_;
    int tmp = full_angles.angles[ch].v_angle >> angle_shift_;
    // robinW V_FOV (-58°~15°), if v_angle < -45, add 90° to make sure cpoint.h_angle not overflow
    // decoder need check v_angle, if  32° <= v_angle < =45° need -90°
    if (pkt.type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD && tmp < -2048) {
      tmp = tmp + 4096;  // 4096 means 90°
    }
    cpoint.v_angle = tmp;

    cpoint.ts_100us = block.header.ts_10us / 10 + (pkt.common.ts_start_us - current_cframe_->ts_us_start) / 100;
    cpoint.scan_id = block.header.scan_id;
    cpoint.scan_idx = block.header.scan_idx;
    cpoint.flags = ch;
    if (block.header.in_roi == 0x3) {
      cpoint.flags |= 0x8;
    }
    if (m == 1) {
      cpoint.flags |= 0x4;
    }
    cpoint.ref = pkt.use_reflectance ? pt.reflectance : pt.intensity;
    current_cframe_->item_number += 1;
  }
  return;
}

inline void CframeConverter::add_cpoint_to_current_cframe_(void *ctx, const InnoDataPacket &pkt,
                                                           const InnoCoBlock &block, const InnoCoChannelPoint &pt,
                                                           const InnoCoBlockFullAngles &full_angles, const uint16_t ch,
                                                           const uint16_t m) {
  size_t pcount = current_cframe_->item_number;
  inno_cpoint &cpoint = current_cframe_->cpoints[pcount];

  if (pt.radius > 0 &&
      pcount < kMaxNumberInCframe && InnoDataPacketUtils::is_robinw_inside_fov_point(full_angles.angles[ch])) {
    // xxx todo: hard coded
    cpoint.radius = pt.radius >> en_radius_shift_;
    cpoint.h_angle = full_angles.angles[ch].h_angle >> angle_shift_;
    int tmp = full_angles.angles[ch].v_angle >> angle_shift_;
    // robinW V_FOV (-58°~15°), if v_angle < -45, add 90° to make sure cpoint.h_angle not overflow
    // decoder need check v_angle, if  32° <= v_angle < =45° need -90°
    if (tmp < -2048) {
      tmp = tmp + 4096;  // 4096 means 90°
    }
    cpoint.v_angle = tmp;

    cpoint.ts_100us = block.header.ts_10us / 10 + (pkt.common.ts_start_us - current_cframe_->ts_us_start) / 100;
    cpoint.scan_id = block.header.scan_id;
    cpoint.scan_idx = block.header.scan_idx;
    cpoint.flags = ch;
    if (block.header.in_roi == 0x3) {
      cpoint.flags |= 0x8;
    }
    if (m == 1) {
      cpoint.flags |= 0x4;
    }
    cpoint.ref = pt.refl;
    current_cframe_->item_number += 1;
  }
}

inline void CframeConverter::add_xyz_point_to_current_cframe_(
    void *ctx,
    const InnoDataPacket &pkt,
    const InnoXyzPoint &pt) {
  size_t pcount = current_cframe_->item_number;
  if (pt.radius > 0 && pcount < kMaxNumberInCframe) {
    inno_point &point = current_cframe_->points[pcount];
    point.x = pt.x;
    point.y = pt.y;
    point.z = pt.z;
    point.radius = pt.radius;
    point.ts_100us = pt.ts_10us / 10 +
                     (pkt.common.ts_start_us -
                      current_cframe_->ts_us_start) / 100;
    point.ref = pt.refl;
    point.flags = pt.channel;
    if (pt.in_roi == 3) {
      point.flags |= 0x8;
    }
    point.scan_id = pt.scan_id;
    point.scan_idx = pt.scan_idx;
    point.reserved = 0;

    current_cframe_->item_number += 1;
  }
}

inline void CframeConverter::add_xyz_point_to_current_cframe_(void *ctx, const InnoDataPacket &pkt,
                                                              const InnoEnXyzPoint &pt) {
  size_t pcount = current_cframe_->item_number;
  if (pt.radius > 0 && pcount < kMaxNumberInCframe) {
    inno_point &point = current_cframe_->points[pcount];
    point.x = pt.x;
    point.y = pt.y;
    point.z = pt.z;
    point.radius = pt.radius;
    point.ts_100us = pt.ts_10us / 10 + (pkt.common.ts_start_us - current_cframe_->ts_us_start) / 100;
    point.ref = pkt.use_reflectance ? pt.reflectance : pt.intensity;
    point.flags = pt.channel;
    if (pt.in_roi == 3) {
      point.flags |= 0x8;
    }
    point.scan_id = pt.scan_id;
    point.scan_idx = pt.scan_idx;
    point.reserved = 0;

    current_cframe_->item_number += 1;
  }
}

void CframeConverter::update_current_cframe_v2_(
    const InnoDataPacket *pkt,const char *table) {
  // sanity check
  if (!InnoDataPacketUtils::check_data_packet(*pkt, 0)) {
    inno_log_error("pkt sanity check failed");
    return;
  }

  current_cframe_->ts_us_end = pkt->common.ts_start_us;  // need to update
  if (pkt->confidence_level < current_cframe_->conf_level) {
    current_cframe_->conf_level = pkt->confidence_level;
  }
  // use macro way is as fast as the faster one
  if (CHECK_EN_SPHERE_POINTCLOUD_DATA(pkt->type)) {
    size_t count = 0;
    ITERARATE_INNO_DATA_PACKET_EN_CPOINTS(add_cpoint_to_current_cframe_, NULL, pkt, count);
  } else if (CHECK_EN_XYZ_POINTCLOUD_DATA(pkt->type)) {
    ITERARATE_INNO_DATA_PACKET_EN_XYZ_POINTS(add_xyz_point_to_current_cframe_, NULL, pkt);
  } else if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    size_t count = 0;
    ITERARATE_INNO_DATA_PACKET_CPOINTS(add_cpoint_to_current_cframe_, NULL, pkt, count);
  } else if (pkt->type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
             pkt->type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
    size_t count = 0;
    ITERARATE_INNO_DATA_PACKET_CO_CPOINTS(add_cpoint_to_current_cframe_, NULL, pkt, count, table);
  } else if (pkt->type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
    ITERARATE_INNO_DATA_PACKET_XYZ_POINTS(add_xyz_point_to_current_cframe_, NULL, pkt);
  } else {
    inno_log_verify(false, "invalid type %d", pkt->type);
  }
}

void CframeConverter::update_current_cframe_(const InnoDataPacket *pkt) {
  if (pkt->type != INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    inno_log_warning("ignore type %u", pkt->type);
    return;
  }
  current_cframe_->ts_us_end = pkt->common.ts_start_us;  // need to update

  // sanity check
  if (!InnoDataPacketUtils::check_data_packet(*pkt, 0)) {
    inno_log_error("pkt sanity check failed");
    return;
  }

  uint32_t unit_size;
  uint32_t mr;
  InnoDataPacketUtils::get_block_size_and_number_return(*pkt,
                                                        &unit_size,
                                                        &mr);
  char *p_addr = reinterpret_cast<char*>(const_cast<InnoDataPacket*>(pkt));
  const InnoBlock *block =
      reinterpret_cast<const InnoBlock *>(p_addr + sizeof(InnoDataPacket));
  for (size_t i = 0;
       i < pkt->item_number;
       i++, block = reinterpret_cast<const InnoBlock *>
                  (reinterpret_cast<const char *>(block) + unit_size)) {
    InnoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header);

    for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {
      for (uint32_t m = 0; m < mr; m++) {
        const InnoChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];
        add_cpoint_to_current_cframe_(NULL, *pkt, *block, pt,
                                      full_angles, ch, m);
      }
    }
  }
}

}  // namespace innovusion
