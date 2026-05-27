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
#ifndef HS_LIDAR_QT_V1_H
#define HS_LIDAR_QT_V1_H

#include <udp_protocol_header.h>
namespace hesai
{
namespace lidar
{
#define PANDARQT_MAX_BLOCK_NUM 4
#define PANDARQT_MAX_LASER_NUM 64
#pragma pack(push, 1)

struct HS_LIDAR_BODY_AZIMUTH_QT_V1 {
  uint16_t m_u16Azimuth;

  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_BODY_CHN_NNIT_QT_V1 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  uint8_t m_u8Confidence;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
  inline uint8_t GetConfidenceLevel() const { return m_u8Confidence; }
};

struct HS_LIDAR_TAIL_QT_V1 {
  // shutdown flag, bit 0
  static const uint8_t kShutdown = 0x01;

  // return mode
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kDualReturn = 0x39;

  ReservedInfo1 m_reservedInfo1;
  ReservedInfo2 m_reservedInfo2;
  uint8_t m_u8Shutdown;
  ReservedInfo3 m_reservedInfo3;
  uint16_t m_u16MotorSpeed;
  uint32_t m_u32Timestamp;
  uint8_t m_u8ReturnMode;
  uint8_t m_u8FactoryInfo;
  uint8_t m_u8UTC[6];
  uint32_t m_u32SeqNum;

  inline uint8_t GetStsID0() const { return m_reservedInfo1.GetID(); }
  inline uint16_t GetData0() const { return m_reservedInfo1.GetData(); }

  inline uint8_t GetStsID1() const { return m_reservedInfo2.GetID(); }
  inline uint16_t GetData1() const { return m_reservedInfo2.GetData(); }

  inline uint8_t HasShutdown() const { return m_u8Shutdown & kShutdown; }

  inline uint8_t GetStsID2() const { return m_reservedInfo3.GetID(); }
  inline uint16_t GetData2() const { return m_reservedInfo3.GetData(); }

  inline uint16_t GetMotorSpeed() const { return little_to_native(m_u16MotorSpeed); }

  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }

  inline uint8_t GetReturnMode() const { return m_u8ReturnMode; }
  inline bool IsLastReturn() const { return m_u8ReturnMode == kLastReturn; }
  inline bool IsStrongestReturn() const { return m_u8ReturnMode == kStrongestReturn; }
  inline bool IsDualReturn() const { return m_u8ReturnMode == kDualReturn; }

  inline uint8_t GetFactoryInfo() const { return m_u8FactoryInfo; }

  inline uint8_t GetUTCData(uint8_t index) const {
    return m_u8UTC[index < sizeof(m_u8UTC) ? index : 0];
  }

  inline uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }

  static uint16_t GetSeqNumSize() { return sizeof(m_u32SeqNum); }
  uint64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
    if (m_u8UTC[0] != 0) {
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
		else {
      uint32_t utc_time_big = *(uint32_t*)(&m_u8UTC[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}

  }
};

struct HS_LIDAR_HEADER_QT_V1 {
  static const uint8_t kSequenceNum = 0x01;
  static const uint8_t kIMU = 0x02;
  static const uint8_t kDistUnit = 0x04;
  static const uint8_t kFirstBlockLastReturn = 0x01;
  static const uint8_t kFirstBlockStrongestReturn = 0x02;

  uint8_t m_u8LaserNum;
  uint8_t m_u8BlockNum;
  uint8_t m_u8EchoCount;
  uint8_t m_u8DistUnit;
  uint8_t m_u8EchoNum;
  uint8_t m_u8Status;

  inline uint8_t GetLaserNum() const { return m_u8LaserNum; }
  inline uint8_t GetBlockNum() const { return m_u8BlockNum; }
  inline float GetDistUnit() const { return m_u8DistUnit / 1000.f; }
  inline uint8_t GetEchoCount() const { return m_u8EchoCount; }
  inline bool IsFirstBlockLastReturn() const {
    return m_u8EchoCount == kFirstBlockLastReturn;
  }
  inline bool IsFirstBlockStrongestReturn() const {
    return m_u8EchoCount == kFirstBlockStrongestReturn;
  }
  inline uint8_t GetEchoNum() const { return m_u8EchoNum; }
  inline bool HasSeqNum() const { return m_u8Status & kSequenceNum; }
  inline bool HasIMU() const { return m_u8Status & kIMU; }

  inline uint16_t GetPacketSize() const {
    uint16_t u16TailDelta =
        (HasSeqNum() ? 0 : HS_LIDAR_TAIL_QT_V1::GetSeqNumSize());

    return sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_QT_V1) +
           sizeof(HS_LIDAR_TAIL_QT_V1) +
           (sizeof(HS_LIDAR_BODY_AZIMUTH_QT_V1) +
            sizeof(HS_LIDAR_BODY_CHN_NNIT_QT_V1) * GetLaserNum()) *
               GetBlockNum() -
           u16TailDelta;
  }
};
#pragma pack(pop)

namespace PandarQt {
  static constexpr int32_t PandarQT_BLOCK_NS_OFFSET1 = 25710;
  const int32_t PandarQT_BLOCK_NS_OFFSET2[PANDARQT_MAX_BLOCK_NUM] = { 0, 166670, 333330, 500000};
}

}  // namespace lidar
}  // namespace hesai
#endif
