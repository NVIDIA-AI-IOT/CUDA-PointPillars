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
#ifndef FAULT_MESSAGE_H
#define FAULT_MESSAGE_H

#define LENS_AZIMUTH_AREA_NUM (12)
#define LENS_ELEVATION_AREA_NUM (8)
namespace hesai
{
namespace lidar
{
#define faultmessagePrint printf
#pragma pack(push, 1)
enum LensDirtyState {
  kUndefineData = -1,
  kLensNormal = 0,
  kPassable = 1,
  kUnPassable = 3,
};

struct MultiplexingInfo {
  uint8_t blockage_flag;
  uint16_t average_energy;
};

struct FaultMessageInfo4_3 {
  uint8_t fault_code_type;
  uint8_t rolling_counter;
  uint8_t tdm_data_indicate;
  uint8_t time_division_multiplexing[27];
  uint16_t software_id;
  uint16_t software_version;
  uint16_t hardware_version;
  uint16_t bt_version;
  uint8_t heating_state;
  uint8_t high_temperture_shutdown_state;
  uint8_t reversed[3];
  void Print() const {
    faultmessagePrint("faultcode_type: %d\n", fault_code_type);
    faultmessagePrint("rolling_counter: %u\n", rolling_counter);
    faultmessagePrint("tdm_data_indicate: %d\n", tdm_data_indicate);
    faultmessagePrint("tdm_data:");
    for (int i = 0; i < 27; i++) {
      faultmessagePrint(" 0x%02x", time_division_multiplexing[i]);
    }
    faultmessagePrint("\n");
    faultmessagePrint("software_id: %04x, software_version: %04x, hardware_version: %04x, bt_version: %04x\n", 
            software_id, software_version, hardware_version, bt_version);
    faultmessagePrint("heating_state: %d\n", heating_state);
    faultmessagePrint("lidar_high_temp_state: %d\n", high_temperture_shutdown_state);
  }
};

struct FaultMessageInfo4_7 {
  uint8_t tdm_data_indicate;
  uint8_t time_division_multiplexing[14];
  uint8_t internal_fault_id;
  uint8_t fault_indicate[8];
  uint8_t customer_id;
  uint8_t software_version;
  uint8_t iteration_version;
  uint8_t reversed[17];
  void Print() const {
    faultmessagePrint("tdm_data_indicate: %d\n", tdm_data_indicate);
    faultmessagePrint("tdm_data:");
    for (int i = 0; i < 14; i++) {
      faultmessagePrint(" 0x%02x", time_division_multiplexing[i]);
    }
    faultmessagePrint("\n");
    faultmessagePrint("internal_fault_id: %d\n", internal_fault_id);
    faultmessagePrint("tdm_data:");
    for (int i = 0; i < 8; i++) {
      faultmessagePrint(" 0x%02x", fault_indicate[i]);
    }
    faultmessagePrint("\n");
    faultmessagePrint("customer_id: %02x, software_version: %02x, iteration_version: %02x\n", 
            customer_id, software_version, iteration_version);
  }
};

struct FaultMessageInfo4_9 {
  uint8_t tdm_data_indicate;
  uint8_t time_division_multiplexing[27];
  uint8_t ptp_status;
  float mask_temperature;
  float md_temperature;
  uint8_t heating_state;
  uint8_t internal_fault_id;
  uint8_t fault_indicate[9];
  uint8_t reversed[9];
  void Print() const {
    faultmessagePrint("tdm_data_indicate: %d\n", tdm_data_indicate);
    faultmessagePrint("tdm_data:");
    for (int i = 0; i < 27; i++) {
      faultmessagePrint(" 0x%02x", time_division_multiplexing[i]);
    }
    faultmessagePrint("\n");
    faultmessagePrint("ptp_status: %d\n", ptp_status);
    faultmessagePrint("mask_temperature: %f℃\n", mask_temperature);
    faultmessagePrint("md_temperature: %f℃\n", md_temperature);
    faultmessagePrint("winheat_status: %d\n", heating_state);
    faultmessagePrint("internal_fault_ID: 0x%02x\n", internal_fault_id);
    faultmessagePrint("fault_indicate:");
    for (int i = 0; i < 9; i++) {
      faultmessagePrint(" 0x%02x", fault_indicate[i]);
    }
    faultmessagePrint("\n");
  }
};

struct FaultMessageInfo4_10 {
  uint8_t fault_code_type;
  uint8_t rolling_counter;
  uint32_t blockage_fault_flag;
  uint8_t data_indicator;
  int16_t internal_temperature;
  MultiplexingInfo time_division_multiplexing[8];
  uint8_t blockage_intensity;
  uint8_t reserved0[2];
  uint8_t reserved1[6];
  uint32_t crc;
  uint64_t serial_number;
  uint8_t reserved3[24];
  void Print() const {
    faultmessagePrint("fault_code_type: %u\n", fault_code_type);
    faultmessagePrint("rolling_counter: %u\n", rolling_counter);
    faultmessagePrint("blockage_fault_flag: %u\n", blockage_fault_flag);
    faultmessagePrint("data_indicator: %u\n", data_indicator);
    faultmessagePrint("internal_temperature: %d\n", internal_temperature);
    faultmessagePrint("blockage_intensity: %u\n", blockage_intensity);
    for (int i = 0; i < 9; i++) {
      faultmessagePrint("%d: blockage_flag[%u],average_energy[%u]\n", i, time_division_multiplexing[i].blockage_flag, time_division_multiplexing[i].average_energy);
    }
    faultmessagePrint("serial_number: %lu\n", serial_number);
  }
};

union FaultMessageUnionInfo {
  FaultMessageInfo4_3 fault4_3;
  FaultMessageInfo4_7 fault4_7;
  FaultMessageInfo4_9 fault4_9;
  FaultMessageInfo4_10 fault4_10;
};

struct FaultMessageInfo {
  uint16_t fault_parse_version;
  uint8_t version;
  uint8_t utc_time[6];
  uint32_t timestamp_sec;
  uint32_t timestamp;
  double total_time;
  uint8_t operate_state;
  uint8_t fault_state;
  uint8_t total_faultcode_num;
  uint8_t faultcode_id;
  uint32_t faultcode;
  FaultMessageUnionInfo union_info;
  inline uint8_t getVersion() { return version; }
  inline uint8_t getLidarState() { return operate_state; }
  inline uint8_t getFaultState() { return fault_state; }
  inline uint8_t getFaultNum() { return total_faultcode_num; }
  inline uint32_t getFaultCode() { return faultcode; }
  void Print() const {
    faultmessagePrint("version: %u\n", version);
    faultmessagePrint("utc_time: %u.%u.%u %u:%u:%u.%u\n", utc_time[0], utc_time[1], 
            utc_time[2], utc_time[3], utc_time[4], utc_time[5], timestamp);
    faultmessagePrint("timestamp: %u.%u\n", timestamp_sec, timestamp);
    faultmessagePrint("total_time: %lf\n", total_time);
    faultmessagePrint("operate_state: %u\n", operate_state);
    faultmessagePrint("fault_state: %d\n", fault_state);
    faultmessagePrint("total_faultcode_num: %d, faultcode_id: %d, faultcode: 0x%08x\n", 
            total_faultcode_num, faultcode_id, faultcode);
    switch (fault_parse_version) {
      case 0x0403:
        union_info.fault4_3.Print();
        break;
      case 0x0407:
        union_info.fault4_7.Print();
        break;
      case 0x0409:
        union_info.fault4_9.Print();
        break;
      case 0x040A:
        union_info.fault4_10.Print();
        break;
      default:
        break;
    }
  }
};
#pragma pack(pop)

}
}
#endif