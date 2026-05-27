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
#ifndef HS_LIDAR_JT16_H
#define HS_LIDAR_JT16_H

#include <udp_protocol_header.h>
#include <ctime>
namespace hesai
{
namespace lidar
{
  namespace JT {
    static constexpr int kMaxChannelJt16 = 16;
  }
#pragma pack(push, 1)
struct HS_LIDAR_PRE_HEADER_JT {
  static const uint16_t kDelimiter = 0xffee;

  uint16_t m_u16Delimiter;
  uint8_t m_u8VersionMajor;
  uint8_t m_u8VersionMinor;
  uint8_t m_u8TimeMultiVersion;
  uint8_t m_u8DataType;

  inline bool IsValidDelimiter() const {
    return little_to_native(m_u16Delimiter) == kDelimiter;
  }
  inline uint8_t GetTimeMultiVersion() const { return m_u8TimeMultiVersion; }
  inline uint8_t GetDataType() const { return m_u8DataType; }
};

struct HS_LIDAR_HEADER_JT {
  uint8_t m_u8UTC[6];
  uint32_t m_u32Timestamp;

  int64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
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
      int64_t unix_second = ((utc_time_big >> 24) & 0xff) |
              ((utc_time_big >> 8) & 0xff00) |
              ((utc_time_big << 8) & 0xff0000) |
              ((utc_time_big << 24));
      return unix_second * 1000000 + GetTimestamp();
		}
  }

  inline uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }
  inline uint8_t GetUTCData(uint8_t index) const {
    return m_u8UTC[index < sizeof(m_u8UTC) ? index : 0];
  }
};

struct HS_LIDAR_TAIL_JT {
  uint32_t m_u32Crc;

  inline uint32_t GetCrc() const { return little_to_native(m_u32Crc); }
};

struct HS_LIDAR_BODY_AZIMUTH_JT {
  uint16_t m_u16Azimuth;

  inline uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }
};

struct HS_LIDAR_BODY_CHN_UNIT_JT {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;

  inline uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  inline uint8_t GetReflectivity() const { return m_u8Reflectivity; }
};

struct HS_LIDAR_BODY_POINT_JT {
  uint32_t m_u32DirtyDegree;
  uint8_t m_u8LidarState;
  uint8_t m_u8ReservedId;
  uint16_t m_u16ReservedInfo;
  uint16_t m_u16SequenceNum;


  inline uint8_t GetDirtyDegree(uint8_t index) const { 
    return static_cast<uint8_t>((m_u32DirtyDegree >> (index * 2)) & 0x03);
  }
  inline uint8_t GetLidarState() const { return m_u8LidarState; }
  inline uint8_t GetReservedId() const { return m_u8ReservedId; }
  inline uint16_t GetReservedInfo() const { return little_to_native(m_u16ReservedInfo); }
  inline uint16_t GetSequenceNum() const { return little_to_native(m_u16SequenceNum); }
};

struct HS_LIDAR_BODY_IMU_JT {
  static constexpr double AccelUint = 1 / 8192.0;
  static constexpr double AngVelUint = 1 / 32.8;
  int16_t m_i16IMUXAccel;
  int16_t m_i16IMUYAccel;
  int16_t m_i16IMUZAccel;
  int16_t m_i16IMUXAngVel;
  int16_t m_i16IMUYAngVel;
  int16_t m_i16IMUZAngVel;
  int16_t m_i16SequenceNum;

  inline double GetIMUXAccel() const {
    return little_to_native(m_i16IMUXAccel) * AccelUint;
  }
  inline double GetIMUYAccel() const {
    return little_to_native(m_i16IMUYAccel) * AccelUint;
  }
  inline double GetIMUZAccel() const {
    return little_to_native(m_i16IMUZAccel) * AccelUint;
  }
  inline double GetIMUXAngVel() const {
    return little_to_native(m_i16IMUXAngVel) * AngVelUint;
  }
  inline double GetIMUYAngVel() const {
    return little_to_native(m_i16IMUYAngVel) * AngVelUint;
  }
  inline double GetIMUZAngVel() const {
    return little_to_native(m_i16IMUZAngVel) * AngVelUint;
  }
  inline uint16_t GetSequenceNum() const { return little_to_native(m_i16SequenceNum); }
};
#pragma pack(pop)
}  // namespace lidar
}  // namespace hesai
#endif
