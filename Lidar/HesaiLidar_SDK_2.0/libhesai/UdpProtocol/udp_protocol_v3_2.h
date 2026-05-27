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
#ifndef HS_LIDAR_QT_V2_H
#define HS_LIDAR_QT_V2_H

#include <udp_protocol_header.h>
namespace hesai
{
namespace lidar
{

#pragma pack(push, 1)
  namespace QT128
  {
    static constexpr int kMaxChannelQT128 = 128;
    static constexpr int Qt128LoopNum = 4;
    struct FiretimesQt128 {
      float firetimes[Qt128LoopNum][kMaxChannelQT128];
    };
    struct PandarQTChannelConfig {
      uint16_t sob;
      uint8_t major_version;
      uint8_t min_version;
      uint8_t laser_num;
      uint8_t m_u8BlockNum;
      uint8_t loopNum;
      int channelConfigTable[Qt128LoopNum][kMaxChannelQT128];
      std::string m_sHashValue;
      bool isObtained;
    };
  }

struct HS_LIDAR_BODY_AZIMUTH_QT_V2 {
  uint16_t m_u16Azimuth;
  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_BODY_CHN_UNIT_QT_V2 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  uint8_t reserved;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_CHN_UNIT_NO_CONF_QT_V2 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_CRC_QT_V2 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_FUNCTION_SAFETY {
  uint8_t m_u8Version;
  uint8_t m_u8Status;
  uint8_t m_u8FaultInfo;
  uint16_t m_u16FaultCode;
  uint8_t m_u8Reserved[8];
  HS_LIDAR_BODY_CRC_QT_V2 m_crc;
  inline uint16_t GetFaultCode() const { return little_to_native(m_u16FaultCode); }
  inline uint8_t GetLidarState() const {
    return (m_u8Status >> 5) & 0x07;
  }
};

struct HS_LIDAR_TAIL_QT_V2 {
  // shutdown flag, bit 0
  static const uint8_t kShutdown = 0x01;

  // return mode
  static const uint8_t kFirstReturn = 0x33;
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kLastAndStrongestReturn = 0x39;
  static const uint8_t kFirstAndLastReturn = 0x3b;
  static const uint8_t kFirstAndStrongestReturn = 0x3c;
  static const uint8_t kStrongestAndSecondReturn = 0x3e;
  static const uint8_t kSecondReturn = 0x34;
  static const uint8_t kFirstAndSecondReturn = 0x3a;

  ReservedInfo1 m_reservedInfo1;
  uint16_t m_reservedInfo2;
  uint8_t m_u8ModeFlag;
  ReservedInfo3 m_reservedInfo3;
  uint16_t m_u16AzimuthFlag;
  uint8_t m_u8WorkingMode;
  uint8_t m_u8ReturnMode;
  uint16_t m_u16MotorSpeed;
  uint8_t m_u8UTC[6];
  uint32_t m_u32Timestamp;
  uint8_t m_u8FactoryInfo;

  inline uint8_t GetStsID1() const { return m_reservedInfo1.GetID(); }
  inline uint16_t GetData1() const { return m_reservedInfo1.GetData(); }
  inline uint8_t GetStsID3() const { return m_reservedInfo3.GetID(); }
  inline uint16_t GetData3() const { return m_reservedInfo3.GetData(); }
  inline uint8_t GetModeFlag() const { return m_u8ModeFlag; }
  inline uint16_t GetMotorSpeed() const { return little_to_native(m_u16MotorSpeed); }
  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }
  inline uint8_t GetReturnMode() const { return m_u8ReturnMode; }
  inline bool IsLastReturn() const { return m_u8ReturnMode == kLastReturn; }
  inline bool IsStrongestReturn() const { return m_u8ReturnMode == kStrongestReturn; }
  inline bool IsLastAndStrongestReturn() const {
    return m_u8ReturnMode == kLastAndStrongestReturn;
  }
  inline uint8_t GetFactoryInfo() const { return m_u8FactoryInfo; }
  inline uint8_t GetUTCData(uint8_t index) const {
    return m_u8UTC[index < sizeof(m_u8UTC) ? index : 0];
  }
  inline bool getFiretimeMode(int blockIndex) const {
    return (m_u8ModeFlag >> blockIndex) & 0x01;
  }

  inline uint64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
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

struct HS_LIDAR_TAIL_SEQ_NUM_QT_V2 {
  uint32_t m_u32SeqNum;

  inline uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
  static uint32_t GetSeqNumSize() { return sizeof(m_u32SeqNum); }
};

struct HS_LIDAR_TAIL_CRC_QT_V2 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_CYBER_SECURITY_QT_V2 {
  uint8_t m_u8Signature[32];

  inline uint8_t GetSignatureData(uint8_t index) const {
    return m_u8Signature[index < sizeof(m_u8Signature) ? index : 0];
  }
};

struct HS_LIDAR_HEADER_QT_V2 {
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

  inline uint32_t unitSize() const {
    return (sizeof(HS_LIDAR_BODY_CHN_UNIT_NO_CONF_QT_V2) + (hasConfidence(m_u8Status) ? sizeof(uint8_t) : 0));
  }

  inline uint16_t GetPacketSize() const {
    return sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_QT_V2) +
           (sizeof(HS_LIDAR_BODY_AZIMUTH_QT_V2) +
           unitSize() * GetLaserNum()) * GetBlockNum() +
           sizeof(HS_LIDAR_BODY_CRC_QT_V2) +
           (hasFunctionSafety(m_u8Status) ? sizeof(HS_LIDAR_FUNCTION_SAFETY) : 0) +
           sizeof(HS_LIDAR_TAIL_QT_V2) +
           (hasSeqNum(m_u8Status) ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_QT_V2) : 0) +
           sizeof(HS_LIDAR_TAIL_CRC_QT_V2) +
           (hasSignature(m_u8Status) ? sizeof(HS_LIDAR_CYBER_SECURITY_QT_V2) : 0);
  }
};

#pragma pack(pop)

namespace QT128C2X {
  static constexpr int32_t QT128C2X_BLOCK_NS_OFFSET1 = 9000;
  static constexpr int32_t PandarQT_BLOCK_NS_OFFSET2 = 111110;
}
}  // namespace lidar
}  // namespace hesai
#endif
