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
#ifndef HS_LIDAR_FT_V2_H
#define HS_LIDAR_FT_V2_H

#include <udp_protocol_header.h>
namespace hesai
{
namespace lidar
{
#pragma pack(push, 1)
  namespace FT {
    static constexpr int FT2_CORRECTION_LEN = 512;
    static constexpr int CHANNEL_MAX = 128;
    static constexpr int COLUMN_MAX = 384;
    static constexpr int HASH_BYTES_LENGTH = 64;
    struct PandarFTCorrectionsHeader {
      uint8_t pilot[2];
      uint8_t version[2];
      uint8_t reversed[2];
    };
    struct PandarFTCorrections {
      PandarFTCorrectionsHeader header;
      uint16_t column_number;
      uint16_t channel_number;
      uint8_t resolution;
      int16_t angles[CHANNEL_MAX * COLUMN_MAX * 2];
      int32_t angles_32[CHANNEL_MAX * COLUMN_MAX * 2];
      float elevations[FT2_CORRECTION_LEN * FT2_CORRECTION_LEN];
      float azimuths[FT2_CORRECTION_LEN * FT2_CORRECTION_LEN];
      uint8_t hashValue[32];
      bool display[FT2_CORRECTION_LEN * FT2_CORRECTION_LEN];
      PandarFTCorrections() {
        for (int i = 0; i < FT2_CORRECTION_LEN * FT2_CORRECTION_LEN; i++) {
          display[i] = true;
        }
      }
      void clear() {
        memset(&header, 0, sizeof(PandarFTCorrectionsHeader));
        memset(elevations, 0, sizeof(float) * FT2_CORRECTION_LEN * FT2_CORRECTION_LEN);
        memset(azimuths, 0, sizeof(float) * FT2_CORRECTION_LEN * FT2_CORRECTION_LEN);
      }
    };
  }

struct HS_LIDAR_BODY_CHN_UNIT_FT_V2 {
  uint16_t distance;
  uint8_t reflectivity;
  uint8_t background;
  uint8_t confidence;

  uint16_t GetDistance() const { return little_to_native(distance); }
  uint8_t GetReflectivity() const { return reflectivity; }
  uint8_t GetConfidenceLevel() const { return confidence; }
};

struct HS_LIDAR_TAIL_FT_V2 {
  // shutdown flag, bit 0
  static const uint8_t kShutdown = 0x01;

  // return mode
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kDualReturn = 0x39;

  ReservedInfo1 m_reservedInfo1;
  ReservedInfo2 m_reservedInfo2;
  uint8_t status_mode;
  uint16_t column_id;
  uint8_t frame_flag;
  uint8_t high_temperature_shutdown_flag;
  uint8_t return_mode;
  uint16_t frame_duration;
  uint8_t utc[6];
  uint32_t timestamp;
  uint8_t factory;
  uint32_t sequence_num;

  uint8_t GetStsID0() const { return m_reservedInfo1.GetID(); }
  uint16_t GetData0() const { return m_reservedInfo1.GetData(); }

  uint8_t GetStsID1() const { return m_reservedInfo2.GetID(); }
  uint16_t GetData1() const { return m_reservedInfo2.GetData(); }


  uint16_t GetMotorSpeed() const { return little_to_native(frame_duration); }

  uint32_t GetTimestamp() const { return little_to_native(timestamp); }

  uint8_t GetReturnMode() const { return return_mode; }

  uint8_t GetUTCData(uint8_t index) const {
    return utc[index < sizeof(utc) ? index : 0];
  }

  uint32_t GetSeqNum() const { return little_to_native(sequence_num); }

  uint64_t GetMicroLidarTimeU64(LastUtcTime &last_utc_time) const {
    if (utc[0] != 0) {
      if (last_utc_time.last_utc[0] == utc[0]
          && last_utc_time.last_utc[1] == utc[1]
          && last_utc_time.last_utc[2] == utc[2]
          && last_utc_time.last_utc[3] == utc[3]
          && last_utc_time.last_utc[4] == utc[4]
          && last_utc_time.last_utc[5] == utc[5]) {
        return last_utc_time.last_time + GetTimestamp();
      }
      last_utc_time.last_utc[0] = utc[0];
      last_utc_time.last_utc[1] = utc[1];
      last_utc_time.last_utc[2] = utc[2];
      last_utc_time.last_utc[3] = utc[3];
      last_utc_time.last_utc[4] = utc[4];
      last_utc_time.last_utc[5] = utc[5];

			struct tm t = {0};
			t.tm_year = utc[0];
			if (t.tm_year < 70) {
        t.tm_year += 100;
      }
			t.tm_mon = utc[1] - 1;
			t.tm_mday = utc[2] + 1;
			t.tm_hour = utc[3];
			t.tm_min = utc[4];
			t.tm_sec = utc[5];
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
      uint32_t utc_time_big = *(uint32_t*)(&utc[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}

  }
};

struct HS_LIDAR_HEADER_FT_V2 {
  static const uint8_t kSequenceNum = 0x01;
  static const uint8_t kIMU = 0x02;
  static const uint8_t kDistUnit = 0x04;
  static const uint8_t kFirstBlockLastReturn = 0x01;
  static const uint8_t kFirstBlockStrongestReturn = 0x02;
  uint16_t total_column;
  uint16_t total_row;
  uint8_t column_resolution;
  uint8_t row_resolution;
  uint8_t echo;
  uint8_t dist_unit;
  uint8_t block_index;
  uint16_t channel_num;
  uint8_t reserved[8];

  uint16_t GetChannelNum() const { return channel_num; }
  float GetDistUnit() const { return dist_unit / 1000.f; }
  uint8_t GetEchoCount() const { return echo; }
  bool IsFirstBlockLastReturn() const {
    return echo == kFirstBlockLastReturn;
  }
  bool IsFirstBlockStrongestReturn() const {
    return echo == kFirstBlockStrongestReturn;
  }
};
#pragma pack(pop)
}  // namespace lidar
}  // namespace hesai
#endif
