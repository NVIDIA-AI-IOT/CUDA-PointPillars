/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/ring_id_converter/ring_id_converter.h"
#include <algorithm>
#include <cfloat>
#include <sstream>
#include <vector>
#include "sdk_common/inno_lidar_packet.h"
#include "utils/inno_lidar_log.h"
#include "utils/utils.h"
#include "sdk_client/lidar_client.h"
#include "ring_id_converter.h"


namespace innovusion {

RingIdConverter::RingIdConverter() {
}

RingIdConverter::~RingIdConverter() {
}

int RingIdConverter::update_ring_id_table(InnoDataPacket *pkt) {
  if (pkt) {
    // setup table from ring id table packet received from server
    int direct = pkt->scanner_direction;
    for (uint16_t sc = 0; sc < 64; sc++) {
      ring_id_table_[direct][sc][0] =
        reinterpret_cast<InnoRingIdTable *>(pkt->payload)->table[sc * kInnoChannelNumber + 0];
      ring_id_table_[direct][sc][1] =
        reinterpret_cast<InnoRingIdTable *>(pkt->payload)->table[sc * kInnoChannelNumber + 1];
      ring_id_table_[direct][sc][2] = reinterpret_cast<InnoRingIdTable *>(pkt->payload)->table[sc * kInnoChannelNumber + 2];
      ring_id_table_[direct][sc][3] =
        reinterpret_cast<InnoRingIdTable *>(pkt->payload)->table[sc * kInnoChannelNumber + 3];
    }
  } else {
    memset(ring_id_table_, 0,
           sizeof(uint16_t) * INNO_FRAME_DIRECTION_MAX * RingIdConsts::kMaxScanLinePerFrame * kInnoChannelNumber);
  }
  return 0;
}

}  // namespace innovusion
