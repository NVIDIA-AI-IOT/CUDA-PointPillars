/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_RING_ID_CONVERTER_CONSTS_H_
#define SDK_CLIENT_RING_ID_CONVERTER_CONSTS_H_

namespace innovusion {
class RingIdConsts {
 public:
  static const size_t kGalvoPeridUnitsPerSecond = 15000;
  static const size_t kPolygonFacet = 5;
  static const size_t kMaxScanLinePerFrame = 64;
  static const size_t kMaxRPM = kMaxScanLinePerFrame * 10 * 60 / kPolygonFacet;  // only for 10 fps
};

}  // namespace innovusion

#endif // SDK_CLIENT_RING_ID_CONVERTER_CONSTS_H_
