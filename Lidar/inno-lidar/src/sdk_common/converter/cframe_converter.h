/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#ifndef CONVERTER_CFRAME_CONVERTER_H_
#define CONVERTER_CFRAME_CONVERTER_H_

#include <stdlib.h>

#include "sdk_common/converter/cframe_legacy.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"

namespace innovusion {

class CframeConverter {
 private:
  static const size_t kMaxNumberInCframe = 600 * 1000;

 public:
  CframeConverter();
  ~CframeConverter();

 public:
  inno_cframe_header *add_data_packet(const InnoDataPacket *pkt,
                                      int interval, const char *table = nullptr);
  inno_cframe_header *close_current_frame();

 private:
  void start_new_current_cframe_(const InnoDataPacket *pkt);
  void update_current_cframe_(const InnoDataPacket *pkt);
  void update_current_cframe_v2_(const InnoDataPacket *pkt, const char *table = nullptr);
  void add_cpoint_to_current_cframe_(void *ctx,
                                     const InnoDataPacket &pkt,
                                     const InnoBlock &block,
                                     const InnoChannelPoint &pt,
                                     const InnoBlockFullAngles &,
                                     const uint16_t ch,
                                     const uint16_t m);

  void add_cpoint_to_current_cframe_(void *ctx, const InnoDataPacket &pkt, const InnoEnBlock &block,
                                     const InnoEnChannelPoint &pt, const InnoBlockFullAngles &, const uint16_t ch,
                                     const uint16_t m);

  void add_cpoint_to_current_cframe_(void *ctx, const InnoDataPacket &pkt, const InnoCoBlock &block,
                                     const InnoCoChannelPoint &pt, const InnoCoBlockFullAngles &full_angles,
                                     const uint16_t ch, const uint16_t m);
  void add_xyz_point_to_current_cframe_(
      void *ctx,
      const InnoDataPacket &pkt,
      const InnoXyzPoint &pt);

  void add_xyz_point_to_current_cframe_(void *ctx, const InnoDataPacket &pkt, const InnoEnXyzPoint &pt);

 private:
  uint32_t radius_shift_;
  uint32_t en_radius_shift_;  // for inno en point
  uint32_t angle_shift_;

  ssize_t current_cframe_id_;
  inno_cframe_header *current_cframe_;
  union {
    inno_cframe_header cframe0_;
    char c0a_[sizeof(inno_cframe_header) +
             sizeof(inno_cpoint) * kMaxNumberInCframe];
    char c0b_[sizeof(inno_cframe_header) +
             sizeof(inno_point) * kMaxNumberInCframe];
  };
  union {
    inno_cframe_header cframe1_;
    char c1a_[sizeof(inno_cframe_header) +
              sizeof(inno_cpoint) * kMaxNumberInCframe];
    char c1b_[sizeof(inno_cframe_header) +
              sizeof(inno_point) * kMaxNumberInCframe];
  };
};

}  // namespace innovusion

#endif  // CONVERTER_CFRAME_CONVERTER_H_
