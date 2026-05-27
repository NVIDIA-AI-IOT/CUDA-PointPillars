/************************************************************************************************
Copyright (C) 2023 Hesai Technology Co., Ltd.
Copyright (C) 2023 Original Authors
All rights reserved.

All code in this repository is released under the terms of the following Modified BSD License. 
Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and 
  the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
  the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************************************************************/

#ifndef HSLIDARP40_H
#define HSLIDARP40_H
#include <udp_protocol_header.h>
#include <cstdint>
#include <cstdio>
namespace hesai
{
namespace lidar
{
#define P40_BLOCKNUM 10
#define P40_LASERNUM 40
#define PKT_SIZE_40P (1262)
#define PKT_SIZE_AC (1256)
#define P40_DISTANCEUNIT 0.004

#pragma pack(push, 1)

struct HS_LIDAR_BODY_CHN_UNIT_L40 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_AZIMUTH_L40 {
  // 0xFFEE 2bytes
  uint16_t m_u16Sob;
  uint16_t m_u16Azimuth;

  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_TAIL_SEQ_NUM_L40 {
  uint32_t m_u32SeqNum;

  inline uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
};

struct HS_LIDAR_TAIL_L40 {
  ReservedInfo1 m_reservedInfo1;
  uint8_t m_u8Reserved0[2];
  uint8_t m_u8Shutdown;
  uint16_t m_u16ErrorCode;
  uint16_t m_u16MotorSpeed;
  uint32_t m_u32Timestamp;
  uint8_t m_u8ReturnMode;
  uint8_t m_u8FactoryInfo;
  uint8_t m_u8UTC[6];

  inline uint16_t GetMotorSpeed() const { return little_to_native(m_u16MotorSpeed); }
  inline uint8_t GetStatusID() const { return m_reservedInfo1.GetID(); }
  inline uint16_t GetStatusData() const { return m_reservedInfo1.GetData(); }
  inline uint16_t GetErrorCode() const { return little_to_native(m_u16ErrorCode); }
  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }

  uint64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
    if (last_utc_time.last_utc[0] == m_u8UTC[0] 
        && last_utc_time.last_utc[1] == m_u8UTC[1] 
        && last_utc_time.last_utc[2] == m_u8UTC[2] 
        && last_utc_time.last_utc[3] == m_u8UTC[3] 
        && last_utc_time.last_utc[4] == m_u8UTC[4] 
        && last_utc_time.last_utc[5] == m_u8UTC[5]) {
      return last_utc_time.last_time + GetTimestamp();
    }
    last_utc_time.last_utc[0] = m_u8UTC[0];
    last_utc_time.last_utc[1] = m_u8UTC[1];
    last_utc_time.last_utc[2] = m_u8UTC[2];
    last_utc_time.last_utc[3] = m_u8UTC[3];
    last_utc_time.last_utc[4] = m_u8UTC[4];
    last_utc_time.last_utc[5] = m_u8UTC[5];

    struct tm t = {0};
    t.tm_year = m_u8UTC[0];
    if (t.tm_year < 70) {
        t.tm_year += 100;
    }
    t.tm_mon = m_u8UTC[1] - 1;
    t.tm_mday = m_u8UTC[2] + 1;
    t.tm_hour = m_u8UTC[3];
    t.tm_min = m_u8UTC[4];
    t.tm_sec = m_u8UTC[5];
    t.tm_isdst = 0;
#ifdef _MSC_VER
    TIME_ZONE_INFORMATION tzi;
    GetTimeZoneInformation(&tzi);
    long int timezone =  tzi.Bias * 60;
#endif
    last_utc_time.last_time = (mktime(&t) - timezone - 86400) * 1000000;
    return last_utc_time.last_time + GetTimestamp();
  }
};
#pragma pack(pop)

namespace P40 {
  static constexpr int32_t P40_BLOCK_NS_OFFSET1 = -28580;
  static constexpr int32_t P40_BLOCK_NS_OFFSET2 = -55560;
}

}  // namespace lidar
}  // namespace hesai
#endif  // INTENSITYRETEST_HSLIDARP64_H
