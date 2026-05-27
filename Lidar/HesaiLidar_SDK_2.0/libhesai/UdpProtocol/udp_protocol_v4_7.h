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
#ifndef HS_LIDAR_ST_V7_H
#define HS_LIDAR_ST_V7_H

#include <udp_protocol_header.h>
#include "plat_utils.h"

namespace hesai
{
namespace lidar
{
#pragma pack(push, 1)
  namespace ATX {
    static constexpr int ATX_MAX_CHANNEL_NUM = 200;
    struct ATXFiretimesHeader {
      uint8_t delimiter[2];
      uint8_t version[2];
      uint8_t reserved1;
      uint8_t reserved2;
    };
    struct ATXFiretimesFloat {
      double even_firetime_correction_[ATX_MAX_CHANNEL_NUM];
      double odd_firetime_correction_[ATX_MAX_CHANNEL_NUM];
    };

    struct ATXFiretimes {
      ATXFiretimesHeader header;
      uint8_t channel_number;
      uint16_t angle_division;
      uint16_t raw_even_firetime_correction_[ATX_MAX_CHANNEL_NUM];
      uint16_t raw_odd_firetime_correction_[ATX_MAX_CHANNEL_NUM];
      ATXFiretimesFloat floatCorr;
      uint8_t SHA_value[32];
      bool setToFloatUseAngleDivision() {
        for (int i = 0; i < ATX_MAX_CHANNEL_NUM; i++) {
          floatCorr.even_firetime_correction_[i] = raw_even_firetime_correction_[i] * 1.0;
          floatCorr.odd_firetime_correction_[i] = raw_odd_firetime_correction_[i] * 1.0;
        } 
        return true;
      }
    };

    struct ATXCorrectionsHeader {
      uint8_t delimiter[2];
      uint8_t version[2];
      uint8_t reserved1;
      uint8_t reserved2;
    };

    struct ATXCorrectionFloat {
      uint8_t min_version;
      float azimuth[ATX_MAX_CHANNEL_NUM];
      float azimuth_even[ATX_MAX_CHANNEL_NUM];
      float azimuth_odd[ATX_MAX_CHANNEL_NUM];
      float elevation[ATX_MAX_CHANNEL_NUM];
      float elevation_adjust[ATX_MAX_CHANNEL_NUM];
      static constexpr float kBegElevationAdjust = 20.0;
      static constexpr float kStepElevationAdjust = 2.0;
      static constexpr uint32_t kLenElevationAdjust = 70;
      static constexpr double kEndElevationAdjust = kBegElevationAdjust + kStepElevationAdjust * double(kLenElevationAdjust - 1);
      float ElevationAdjust(float azimuth_angle)
      {
        while(azimuth_angle < 0) {
          azimuth_angle += kCircle;
        }
        while(azimuth_angle >= kCircle) {
          azimuth_angle -= kCircle;
        }
        if (azimuth_angle < kBegElevationAdjust || azimuth_angle > kEndElevationAdjust)
        {
          return 0;
        }
        int index = int((azimuth_angle - kBegElevationAdjust) / kStepElevationAdjust);
        if (index == kLenElevationAdjust - 1){
          return elevation_adjust[index];
        }
        float left_percent = (azimuth_angle - kBegElevationAdjust - index * kStepElevationAdjust) / kStepElevationAdjust;
        return elevation_adjust[index] * (1 - left_percent) + elevation_adjust[index + 1] * left_percent;
      }
    };

    struct ATXCorrections {
      ATXCorrectionsHeader header;
      uint8_t channel_number;
      uint16_t angle_division;
      int16_t raw_azimuths[ATX_MAX_CHANNEL_NUM];
      int16_t raw_azimuths_even[ATX_MAX_CHANNEL_NUM];
      int16_t raw_azimuths_odd[ATX_MAX_CHANNEL_NUM];
      int16_t raw_elevations[ATX_MAX_CHANNEL_NUM];
      int16_t raw_elevations_adjust[ATX_MAX_CHANNEL_NUM];
      ATXCorrectionFloat floatCorr;
      uint8_t SHA_value[32];
      bool display[ATX_MAX_CHANNEL_NUM];
      ATXCorrections() {
        for (int i = 0; i < ATX_MAX_CHANNEL_NUM; i++) {
          display[i] = true;
        }
      }
      bool setToFloatUseAngleDivision() {
        if (angle_division == 0) return false;
        for (int i = 0; i < ATX_MAX_CHANNEL_NUM; i++) {
          floatCorr.azimuth[i] = 1.f * raw_azimuths[i] / angle_division;
          floatCorr.azimuth_even[i] = 1.f * raw_azimuths_even[i] / angle_division;
          floatCorr.azimuth_odd[i] = 1.f * raw_azimuths_odd[i] / angle_division;
          floatCorr.elevation[i] = 1.f * raw_elevations[i] / angle_division;
        }
        for (uint32_t i = 0; i < floatCorr.kLenElevationAdjust; i++) {
          floatCorr.elevation_adjust[i] = 1.f * raw_elevations_adjust[i] / angle_division;
        }
        floatCorr.min_version = header.version[1];
        return true;
      }
    };
  }
struct HS_LIDAR_BODY_AZIMUTH_ST_V7 {
  uint16_t m_u16Azimuth;

  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7 {
  uint8_t m_u8FineAzimuth;

  inline uint8_t GetFineAzimuth() const { return m_u8FineAzimuth; }
};

struct HS_LIDAR_BODY_CHN_NNIT_ST_V7 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  uint8_t m_u8Confidence;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
  inline uint8_t GetConfidenceLevel() const { return m_u8Confidence; }
};


struct HS_LIDAR_BODY_CRC_ST_V7 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_TAIL_ST_V7 {
  static const uint8_t kShutdown = 0x01;
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kDualReturn = 0x39;

  ReservedInfo1 m_reservedInfo1;
  uint8_t m_u8UReserved0[7];
  uint8_t m_u8FrameID;
  uint8_t m_u8UReserved1[7];
  int16_t m_i16MotorSpeed;
  uint32_t m_u32Timestamp;
  uint8_t m_u8ReturnMode;
  uint8_t m_u8FactoryInfo;
  uint8_t m_u8UTC[6];

  inline uint8_t GetStsID0() const { return m_reservedInfo1.GetID(); }
  inline uint16_t GetData0() const { return m_reservedInfo1.GetData(); }

  inline uint8_t GetFrameID() const { return m_u8FrameID; }
  inline int16_t GetMotorSpeed() const { return little_to_native(m_i16MotorSpeed); }
  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }
  inline uint8_t GetReturnMode() const { return m_u8ReturnMode; }
  inline bool IsLastReturn() const { return m_u8ReturnMode == kLastReturn; }
  inline bool IsStrongestReturn() const { return m_u8ReturnMode == kStrongestReturn; }
  inline bool IsDualReturn() const { return m_u8ReturnMode == kDualReturn; }

  inline uint8_t GetFactoryInfo() const { return m_u8FactoryInfo; }
  inline uint8_t GetUTCData(uint8_t index) const {
    return m_u8UTC[index < sizeof(m_u8UTC) ? index : 0];
  }
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
};

struct HS_LIDAR_TAIL_SEQ_NUM_ST_V7 {
  uint32_t m_u32SeqNum;
  inline uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
  static uint32_t GetSeqNumSize() { return sizeof(m_u32SeqNum); }
};

struct HS_LIDAR_TAIL_CRC_ST_V7 {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_CYBER_SECURITY_ST_V7 {
  uint8_t m_u8Signature[32];

  inline uint8_t GetSignatureData(uint8_t index) const {
    return m_u8Signature[index < sizeof(m_u8Signature) ? index : 0];
  }
};

struct HS_LIDAR_E2E_HEADER_ST_V7 {
  uint16_t m_u16Length;
  uint16_t m_u16Counter;
  uint32_t m_u32DataID;
  uint32_t m_u32CRC;

  inline uint16_t GetLength() const { return big_to_native(m_u16Length); }
  inline uint16_t GetCounter() const { return big_to_native(m_u16Counter); }
  inline uint32_t GetDataID() const { return big_to_native(m_u32DataID); }
  inline uint32_t GetCRC() const { return big_to_native(m_u32CRC); }

  inline uint16_t GetE2ECrcSize() const {
    return GetLength() - sizeof(HS_LIDAR_CYBER_SECURITY_ST_V7) - sizeof(uint32_t);
  }
};

struct HS_LIDAR_HEADER_ST_V7 {
  static const uint8_t kSequenceNum = 0x01;
  static const uint8_t kIMU = 0x02;
  static const uint8_t kFunctionSafety = 0x04;
  static const uint8_t kCyberSecurity = 0x08;
  static const uint8_t kConfidenceLevel = 0x10;

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
  inline bool HasFuncSafety() const { return m_u8Status & kFunctionSafety; }
  inline bool HasCyberSecurity() const { return m_u8Status & kCyberSecurity; }
  inline bool HasConfidenceLevel() const { return m_u8Status & kConfidenceLevel; }

  inline uint16_t GetPacketSize() const {
    return sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_ST_V7) +
           (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) +
            sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
            sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * GetLaserNum()) *
               GetBlockNum() +
           sizeof(HS_LIDAR_BODY_CRC_ST_V7) + sizeof(HS_LIDAR_TAIL_ST_V7) +
           (HasSeqNum() ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ST_V7) : 0) +
           sizeof(HS_LIDAR_E2E_HEADER_ST_V7) +
           (HasCyberSecurity() ? sizeof(HS_LIDAR_CYBER_SECURITY_ST_V7) : 0);
  }

  inline uint16_t GetE2ECrcSize() const {
    return sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_ST_V7) +
           (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) +
            sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
            sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * GetLaserNum()) *
               GetBlockNum() +
           sizeof(HS_LIDAR_BODY_CRC_ST_V7) + sizeof(HS_LIDAR_TAIL_ST_V7) +
           (HasSeqNum() ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ST_V7) : 0) +
           sizeof(HS_LIDAR_E2E_HEADER_ST_V7) - sizeof(uint32_t);
  }
};

struct FaultMessageVersion4_7 {
 public:
  uint8_t version_info;
  uint8_t utc_time[6];
  uint32_t time_stamp;
  uint8_t operate_state;
  uint8_t fault_state;
  uint8_t total_fault_code_num;
  uint8_t fault_code_id;
  uint32_t fault_code;
  uint8_t time_division_multiplexing[15];
  uint8_t fault_indicate[9];
  uint8_t software_version[3];
  uint8_t reversed[17];
  HS_LIDAR_E2E_HEADER_ST_V7 E2Eheader;
  uint8_t cycber_security[24];
  inline uint32_t GetTimestamp() const { return little_to_native(time_stamp); }
  uint64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
    if (utc_time[0] != 0) {
      if (last_utc_time.last_utc[0] == utc_time[0] 
          && last_utc_time.last_utc[1] == utc_time[1] 
          && last_utc_time.last_utc[2] == utc_time[2] 
          && last_utc_time.last_utc[3] == utc_time[3] 
          && last_utc_time.last_utc[4] == utc_time[4] 
          && last_utc_time.last_utc[5] == utc_time[5]) {
        return last_utc_time.last_time + GetTimestamp();
      }
      last_utc_time.last_utc[0] = utc_time[0];
      last_utc_time.last_utc[1] = utc_time[1];
      last_utc_time.last_utc[2] = utc_time[2];
      last_utc_time.last_utc[3] = utc_time[3];
      last_utc_time.last_utc[4] = utc_time[4];
      last_utc_time.last_utc[5] = utc_time[5];

			struct tm t = {0};
			t.tm_year = utc_time[0];
			if (t.tm_year < 70) {
        return 0;
      }
			t.tm_mon = utc_time[1] - 1;
			t.tm_mday = utc_time[2] + 1;
			t.tm_hour = utc_time[3];
			t.tm_min = utc_time[4];
			t.tm_sec = utc_time[5];
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
      uint32_t utc_time_big = *(uint32_t*)(&utc_time[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}
  }
  void ParserFaultMessage(FaultMessageInfo &fault_message_info, LastUtcTime &last_utc_time) {
    fault_message_info.fault_parse_version = 0x0407;
    fault_message_info.version = version_info;
    memcpy(fault_message_info.utc_time, utc_time, sizeof(utc_time));
    fault_message_info.timestamp = GetTimestamp();
    fault_message_info.total_time = static_cast<double>(GetMicroLidarTimeU64(last_utc_time)) / 1000000.0;
    fault_message_info.operate_state = operate_state;
    fault_message_info.fault_state = fault_state;
    fault_message_info.total_faultcode_num = total_fault_code_num;
    fault_message_info.faultcode_id = fault_code_id;
    fault_message_info.faultcode = fault_code;
    fault_message_info.union_info.fault4_7.tdm_data_indicate = time_division_multiplexing[0];
    memcpy(fault_message_info.union_info.fault4_7.time_division_multiplexing, time_division_multiplexing + 1, sizeof(time_division_multiplexing) - 1);
    fault_message_info.union_info.fault4_7.internal_fault_id = fault_indicate[0];
    memcpy(fault_message_info.union_info.fault4_7.fault_indicate, fault_indicate + 1, sizeof(fault_indicate) - 1);
    fault_message_info.union_info.fault4_7.customer_id = software_version[0];
    fault_message_info.union_info.fault4_7.software_version = software_version[1];
    fault_message_info.union_info.fault4_7.iteration_version = software_version[2];
    memcpy(fault_message_info.union_info.fault4_7.reversed, reversed, sizeof(reversed));
  }
} ;
#pragma pack(pop)
}  // namespace lidar
}  // namespace hesai
#endif