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
#ifndef HS_LIDAR_ME_V4_H
#define HS_LIDAR_ME_V4_H

#include <udp_protocol_header.h>
#include <ctime>
namespace hesai
{
namespace lidar
{
  namespace pandarN {
    static constexpr int kMaxChannelPandarN = 128;
    #define STR_PANDARN   "PanadarN"
    #define STR_OT128     "OT128"
    #define STR_OTHER     "OTHER"
    struct FiretimeSectionValues {
        struct SectionValue {
          int firetime[2];
        };
        SectionValue section_values[8];
    };

    struct FiretimesPandarN {
        double section_distance = 0;
        FiretimeSectionValues firetime_section_values[kMaxChannelPandarN];
    };
  }

#pragma pack(push, 1)
struct HS_LIDAR_BODY_AZIMUTH_ME_V4 {
  uint16_t m_u16Azimuth;

  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_BODY_CHN_UNIT_NO_CONF_ME_V4 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_CHN_UNIT_ME_V4 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  uint8_t reserved[3];

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_CRC_ME_V4 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_CYBER_SECURITY_ME_V4 {
  uint8_t m_u8Signature[32];

  inline uint8_t GetSignatureData(uint8_t index) const {
    return m_u8Signature[index < sizeof(m_u8Signature) ? index : 0];
  }
};

struct HS_LIDAR_FUNC_SAFETY_ME_V4 {
  static const uint8_t kLidarStateShiftBits = 5;
  static const uint8_t kLidarStateMask = 0x07;
  static const uint8_t kRollingCntShiftBits = 0;
  static const uint8_t kRollingCntMask = 0x07;
  static const uint8_t kFaultTypeCurrent = 0x08;
  static const uint8_t kFaultTypeHistory = 0x10;
  static const uint8_t kFaultNumShiftBits = 4;
  static const uint8_t kFaultNumMask = 0x0F;
  static const uint8_t kFaultIDShiftBits = 0;
  static const uint8_t kFaultIDMask = 0x0F;

  uint8_t m_u8Version;
  uint8_t m_u8Status;
  uint8_t m_u8FaultInfo;
  uint16_t m_u16FaultCode;
  uint8_t m_u8Reserved[8];
  uint32_t m_u32Crc;

  inline uint8_t GetVersion() const { return m_u8Version; }
  inline uint8_t GetLidarState() const {
    return (m_u8Status >> kLidarStateShiftBits) & kLidarStateMask;
  }
  inline bool IsHistoryFault() const { return m_u8Status & kFaultTypeHistory; }
  inline bool IsCurrentFault() const { return m_u8Status & kFaultTypeCurrent; }
  inline uint8_t GetRollingCnt() const {
    return (m_u8Status >> kRollingCntShiftBits) & kRollingCntMask;
  }
  inline uint8_t GetFaultNum() const {
    return (m_u8FaultInfo >> kFaultNumShiftBits) & kFaultNumMask;
  }
  inline uint8_t GetFaultID() const {
    return (m_u8FaultInfo >> kFaultIDShiftBits) & kFaultIDMask;
  }
  inline uint16_t GetFaultCode() const { return little_to_native(m_u16FaultCode); }
  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_TAIL_ME_V4 {
  // return mode
  static const uint8_t kFirstReturn = 0x33;
  static const uint8_t kSecondReturn = 0x34;
  static const uint8_t kThirdReturn = 0x35;
  static const uint8_t kFourthReturn = 0x36;
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kDualReturn = 0x39;
  static const uint8_t kFirstLastReturn = 0x3b;
  static const uint8_t kStongestFirstReturn = 0x3c;

  ReservedInfo1 m_reservedInfo1;
  ReservedInfo2 m_reservedInfo2;
  ReservedInfo3 m_reservedInfo3;
  uint16_t m_u16AzimuthFlag;
  uint8_t m_u8RunningMode;
  uint8_t m_u8ReturnMode;
  uint16_t m_u16MotorSpeed;
  uint8_t m_u8UTC[6];
  uint32_t m_u32Timestamp;
  uint8_t m_u8FactoryInfo;

  inline uint8_t GetStsID0() const { return m_reservedInfo1.GetID(); }
  inline uint16_t GetData0() const { return m_reservedInfo1.GetData(); }
  inline uint8_t GetStsID1() const { return m_reservedInfo2.GetID(); }
  inline uint16_t GetData1() const { return m_reservedInfo2.GetData(); }
  inline uint8_t GetStsID2() const { return m_reservedInfo3.GetID(); }
  inline uint16_t GetData2() const { return m_reservedInfo3.GetData(); }

  inline uint8_t GetReturnMode() const { return m_u8ReturnMode; }
  inline uint16_t GetMotorSpeed() const { return little_to_native(m_u16MotorSpeed); }
  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }
  inline uint8_t getOperationMode() const { return m_u8RunningMode; }
  inline uint8_t getAngleState(int blockIndex) const {
      if (blockIndex >= 8) return 0;
      return (m_u16AzimuthFlag >> (2 * (7 - blockIndex))) & 0x03;
  }

  inline bool IsFirstReturn() const { return m_u8ReturnMode == kFirstReturn; }
  inline bool IsSecondReturn() const { return m_u8ReturnMode == kSecondReturn; }
  inline bool IsThirdReturn() const { return m_u8ReturnMode == kThirdReturn; }
  inline bool IsFourthReturn() const { return m_u8ReturnMode == kFourthReturn; }
  inline bool IsLastReturn() const { return m_u8ReturnMode == kLastReturn; }
  inline bool IsStrongestReturn() const { return m_u8ReturnMode == kStrongestReturn; }
  inline bool IsDualReturn() const { return m_u8ReturnMode == kDualReturn; }
  inline bool IsFirstLastReturn() const { return m_u8ReturnMode == kFirstLastReturn; }
  inline bool IsStongestFirstReturn() const { return m_u8ReturnMode == kStongestFirstReturn; }
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
        return 0;
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
      return last_utc_time.last_time + GetTimestamp() ;
		}
		else {
      uint32_t utc_time_big = *(uint32_t*)(&m_u8UTC[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}
  }

  inline uint8_t GetFactoryInfo() const { return m_u8FactoryInfo; }
};

struct HS_LIDAR_TAIL_SEQ_NUM_ME_V4 {
  uint32_t m_u32SeqNum;

  inline uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
};

struct HS_LIDAR_TAIL_IMU_ME_V4 {
  int16_t m_i16IMUTemperature;
  uint16_t m_u16IMUAccelUnit;
  uint16_t m_u16IMUAngVelUnit;
  uint32_t m_u32IMUTimeStamp;
  int16_t m_i16IMUXAccel;
  int16_t m_i16IMUYAccel;
  int16_t m_i16IMUZAccel;
  int16_t m_i16IMUXAngVel;
  int16_t m_i16IMUYAngVel;
  int16_t m_i16IMUZAngVel;

  inline double GetIMUTemperature() const {
    return little_to_native(m_i16IMUTemperature) / 100.0;
  }
  inline double GetIMUAccelUnit() const {
    return little_to_native(m_u16IMUAccelUnit) / 1000.0 / 1000.0;
  }
  inline double GetIMUAngVelUnit() const {
    return little_to_native(m_u16IMUAngVelUnit) / 100.0 / 1000.0;
  }
  inline uint64_t GetIMUTimestamp() const {
    return little_to_native(m_u32IMUTimeStamp) * 25;
  }
  inline double GetIMUXAccel() const {
    return little_to_native(m_i16IMUXAccel) * GetIMUAccelUnit();
  }
  inline double GetIMUYAccel() const {
    return little_to_native(m_i16IMUYAccel) * GetIMUAccelUnit();
  }
  inline double GetIMUZAccel() const {
    return little_to_native(m_i16IMUZAccel) * GetIMUAccelUnit();
  }
  inline double GetIMUXAngVel() const {
    return little_to_native(m_i16IMUXAngVel) * GetIMUAngVelUnit();
  }
  inline double GetIMUYAngVel() const {
    return little_to_native(m_i16IMUYAngVel) * GetIMUAngVelUnit();
  }
  inline double GetIMUZAngVel() const {
    return little_to_native(m_i16IMUZAngVel) * GetIMUAngVelUnit();
  }
};

struct HS_LIDAR_TAIL_CRC_ME_V4 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_HEADER_ME_V4 {
  static const uint8_t kSequenceNum = 0x01;
  static const uint8_t kIMU = 0x02;
  static const uint8_t kFunctionSafety = 0x04;
  static const uint8_t kCyberSecurity = 0x08;
  static const uint8_t kConfidenceLevel = 0x10;
  static const uint8_t kDistUnit = 0x04;

  static const uint8_t kSingleReturn = 0x00;
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
  inline uint8_t GetEchoNum() const { return m_u8EchoNum; }
  inline bool IsFirstBlockLastReturn() const {
    return m_u8EchoCount == kFirstBlockLastReturn;
  }
  inline bool IsFirstBlockStrongestReturn() const {
    return m_u8EchoCount == kFirstBlockStrongestReturn;
  }

  inline uint32_t unitSize() const {
    return (sizeof(HS_LIDAR_BODY_CHN_UNIT_NO_CONF_ME_V4) + (hasConfidence(m_u8Status) ? sizeof(uint8_t) : 0) + 
            (hasWeightFactor(m_u8Status) ? sizeof(uint8_t) : 0) + (hasEnvLight(m_u8Status) ? sizeof(uint8_t) : 0));
  }

  inline uint32_t packetSize() const {
    return (sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_ME_V4) + (sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + 
            unitSize() * GetLaserNum()) * GetBlockNum() + sizeof(HS_LIDAR_BODY_CRC_ME_V4) + 
            (hasFunctionSafety(m_u8Status) ? sizeof(HS_LIDAR_FUNC_SAFETY_ME_V4) : 0) + 
            sizeof(HS_LIDAR_TAIL_ME_V4) + 
            (hasSeqNum(m_u8Status) ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ME_V4) : 0) + 
            (hasImu(m_u8Status) ? sizeof(HS_LIDAR_TAIL_IMU_ME_V4) : 0) + 
            sizeof(HS_LIDAR_TAIL_CRC_ME_V4) + 
            (hasSignature(m_u8Status) ? sizeof(HS_LIDAR_CYBER_SECURITY_ME_V4) : 0));
  }
};
#pragma pack(pop)

namespace PandarN {
  static constexpr int32_t PandarN_BLOCK_NS_OFFSET1 = 3148;
  static constexpr int32_t PandarN_BLOCK_NS_OFFSET2 = -27778;
  static constexpr int32_t Pandar40S_BLOCK_NS_OFFSET2 = -55556;
  static constexpr int32_t OT128_BLOCK_NS_OFFSET2 = -27778;
  static constexpr int32_t OTHER_BLOCK_NS_OFFSET2 = -111000;
}

}  // namespace lidar
}  // namespace hesai
#endif
