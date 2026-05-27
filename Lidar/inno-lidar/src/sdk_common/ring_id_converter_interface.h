/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */



#ifndef SDK_COMMON_RING_ID_CONVERTER_INTERFACE_H_
#define SDK_COMMON_RING_ID_CONVERTER_INTERFACE_H_

#include <sys/types.h>

/**
 * @brief RingIdConverterInterface
 */
class RingIdConverterInterface {
 public:
  virtual ~RingIdConverterInterface() {}
  /**
   * @brief Get ring id
   * @param mode            InnoLidarMode
   * @param scan_direction  Galvo scan direction
   * @param scan_id         Scan line id
   * @param ch              Channel id
   * @return Return ring id
   */
  virtual uint16_t get_ring_id(InnoLidarMode mode,
                               uint32_t scan_direction,
                               uint32_t scan_id, uint32_t ch) = 0;
};

#endif  // SDK_COMMON_RING_ID_CONVERTER_INTERFACE_H_
