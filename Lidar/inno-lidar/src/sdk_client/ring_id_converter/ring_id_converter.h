/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_RING_ID_CONVERTER_H_
#define SDK_CLIENT_RING_ID_CONVERTER_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "sdk_client/ring_id_converter/consts.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/ring_id_converter_interface.h"
#include "utils/utils.h"

namespace innovusion {
class InnoLidarClient;
/**
 * @brief RingIdConverter
 */
class RingIdConverter : public RingIdConverterInterface {
 public:
  RingIdConverter();
  ~RingIdConverter();
  /**
   * @brief Update ring id table
   * @return Return 0 for success, others for error
   */
  int update_ring_id_table(InnoDataPacket *pkt);

  /**
   * @brief Get ring id
   * @param mode            InnoLidarMode
   * @param scan_direction  Galvo scan direction
   * @param scan_id         Scan line id
   * @param ch              Channel id
   * @return Return ring id
   */
  inline uint16_t get_ring_id(InnoLidarMode mode, uint32_t scan_direction, uint32_t scan_id, uint32_t ch) override {
    return ring_id_table_[scan_direction][scan_id][ch];
  }

 private:
  uint16_t ring_id_table_[INNO_FRAME_DIRECTION_MAX]\
                         [RingIdConsts::kMaxScanLinePerFrame]\
                         [kInnoChannelNumber];
};
}  // namespace innovusion
#endif  // SDK_CLIENT_RING_ID_CONVERTER_H_
