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

/*
 * File:       general_parser.h
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Declare GeneralParser class
*/

#ifndef GENERAL_PARSER_H_
#define GENERAL_PARSER_H_
#define CIRCLE (36000 * 256)
#define DEFAULT_MAX_LASER_NUM (256)
#define CIRCLE_ANGLE (36000)
#define RETURN_MODE_MULTI (0x39)
#define RETURN_MODE_MULTI_TRIPLE (0x3D)
#define SOMEIP_OFFSET (21)
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef _MSC_VER
#include <semaphore.h>
#endif
#include <list>
#include <cmath>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "logger.h"
#include "lidar_types.h"
#include "plat_utils.h"
#include "fault_message.h"
namespace hesai
{
namespace lidar
{  

#define DEFINE_MEMBER_CHECKER(member)                                                                                  \
  template <typename T, typename V = bool>                                                                             \
  struct has_##member : std::false_type                                                                                \
  {                                                                                                                    \
  };                                                                                                                   \
  template <typename T>                                                                                                \
  struct has_##member<                                                                                                 \
      T, typename std::enable_if<!std::is_same<decltype(std::declval<T>().member), void>::value, bool>::type>          \
      : std::true_type                                                                                                 \
  {                                                                                                                    \
  };
#define PANDAR_HAS_MEMBER(C, member) has_##member<C>::value

#define DEFINE_SET_GET(member, Type)                                                                                   \
  template <typename T_Point>                                                                                          \
  inline typename std::enable_if<!PANDAR_HAS_MEMBER(T_Point, member)>::type set_##member(T_Point& point, const Type& value) \
  {                                                                                                                    \
  }                                                                                                                    \
  template <typename T_Point>                                                                                          \
  inline typename std::enable_if<PANDAR_HAS_MEMBER(T_Point, member)>::type set_##member(T_Point& point, const Type& value) \
  {                                                                                                                    \
      point.member = value;                                                                                            \
  }                                                                                                                    \
  template <typename T_Point>                                                                                          \
  inline typename std::enable_if<!PANDAR_HAS_MEMBER(T_Point, member)>::type get_##member(T_Point& point, Type& value)  \
  {                                                                                                                    \
  }                                                                                                                    \
  template <typename T_Point>                                                                                          \
  inline typename std::enable_if<PANDAR_HAS_MEMBER(T_Point, member)>::type get_##member(T_Point& point, Type& value)  \
  {                                                                                                                    \
      value = point.member;                                                                                            \
  }  

DEFINE_MEMBER_CHECKER(x)
DEFINE_MEMBER_CHECKER(y)
DEFINE_MEMBER_CHECKER(z)
DEFINE_MEMBER_CHECKER(intensity)
DEFINE_MEMBER_CHECKER(ring)
DEFINE_MEMBER_CHECKER(timestamp)
DEFINE_MEMBER_CHECKER(confidence)
DEFINE_MEMBER_CHECKER(timeSecond)
DEFINE_MEMBER_CHECKER(timeNanosecond)
DEFINE_MEMBER_CHECKER(weightFactor)
DEFINE_MEMBER_CHECKER(envLight)

DEFINE_SET_GET(x, float)  
DEFINE_SET_GET(y, float)  
DEFINE_SET_GET(z, float)  
DEFINE_SET_GET(intensity, uint8_t)  
DEFINE_SET_GET(ring, uint16_t)  
DEFINE_SET_GET(timestamp, double)  
DEFINE_SET_GET(timeSecond, uint64_t)  
DEFINE_SET_GET(timeNanosecond, uint32_t)
DEFINE_SET_GET(confidence, uint8_t)  
DEFINE_SET_GET(weightFactor, uint8_t)
DEFINE_SET_GET(envLight, uint8_t)



inline float deg2Rad(float deg)
{
    return (float)(deg * 0.01745329251994329575);
}

inline float rad2Deg(float rad)
{
    return (float)(rad * 57.29577951308232087721);
}

enum DistanceCorrectionType {
  OpticalCenter,
  GeometricCenter,
};

struct PacketSeqnumLossMessage {
  uint64_t last_total_package_count;
  uint32_t last_seqnum;
  uint32_t loss_count;
  uint32_t start_time;
  uint32_t total_loss_count;
  uint64_t total_packet_count;
  uint32_t max_sequence;
  bool is_packet_loss;
  bool is_init;
  PacketSeqnumLossMessage() {
    last_total_package_count = 0;
    last_seqnum = 0;
    loss_count = 0;
    start_time = 0;
    total_loss_count = 0;
    total_packet_count = 0;
    max_sequence = 0xFFFFFFFF;
    is_packet_loss = false;
    is_init = false;
  }
};

struct PacketTimeLossMessage {
  uint64_t last_timestamp;
  uint32_t timeloss_count;
  uint32_t timeloss_start_time;
  uint32_t total_timeloss_count;
  uint64_t last_total_package_count;
  bool is_init;
  PacketTimeLossMessage() {
    last_timestamp = 0;
    timeloss_count = 0;
    timeloss_start_time = 0;
    total_timeloss_count = 0;
    last_total_package_count = 0;
    is_init = false;
  }
};

enum StructType {
  CORRECTION_STRUCT = 1,
  FIRETIME_STRUCT = 2
};

struct CorrectionData {
  float elevation[DEFAULT_MAX_LASER_NUM];
  float azimuth[DEFAULT_MAX_LASER_NUM];
  bool display[DEFAULT_MAX_LASER_NUM];
  std::string hash;
  CorrectionData() {
    memset(elevation, 0, sizeof(float) * DEFAULT_MAX_LASER_NUM);
    memset(azimuth, 0, sizeof(float) * DEFAULT_MAX_LASER_NUM);
    for (int i = 0; i < DEFAULT_MAX_LASER_NUM; ++i) {  
        display[i] = true;
    } 
    hash = "";
  }
};

struct LastUtcTime {
  uint64_t last_time = 0;
  int16_t last_utc[6];
  LastUtcTime() {
    last_time = 0;
    last_utc[0] = -1;
    last_utc[1] = -1;
    last_utc[2] = -1;
    last_utc[3] = -1;
    last_utc[4] = -1;
    last_utc[5] = -1;
  }
};

// class GeneralParser
// the GenneralParser class is a base class for parsering packets and computing points
// you can parser the udp or pcap packets using the DecodePacket fuction
// you can compute xyzi of points using the ComputeXYZI fuction, which uses cpu to compute
template <typename T_Point>
class GeneralParser {
 public:
  GeneralParser();
  virtual ~GeneralParser();

  // load correction file, which is necessary for DecodePacket.
  virtual void LoadCorrectionFile(const std::string& correction_path);
  virtual int LoadCorrectionString(const char *correction_string, int len);
  // load firetimes file
  virtual void LoadFiretimesFile(const std::string& firetimes_path);
  virtual int LoadFiretimesString(const char *firetimes_string, int len);
  // load channel config file
  virtual int LoadChannelConfigFile(const std::string channel_config_path);  
  // get the pointer to the struct of the parsed correction file or firetimes file
  virtual void* getStruct(const int type);
  // get display 
  virtual int getDisplay(bool **);
  // get/set correction/firetimes file loading flag
  bool isSetCorrectionSucc() { return get_correction_file_; }
  bool isSetFiretimeSucc() { return get_firetime_file_; }

  // covert a origin udp packet to decoded data, and pass the decoded data to a frame struct to reduce memory copy
  virtual int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index = -1); 
  // xyzi of points after computed is puted in frame  
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index);
  // parse the detailed content of the fault message
  virtual int ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info);
  // determine whether to frame based on azimuth
  bool IsNeedFrameSplit(uint16_t azimuth, FrameDecodeParam &param);
  // In ComputeXYZI, thread-safely count the number of packets that have been parsed
  void FrameNumAdd();
  // get/clear the number of parsed packets
  uint32_t GetComputePacketNum() { return compute_packet_num; }
  void SetComputePacketNumToZero() { compute_packet_num = 0; }
  // compute lidar firetime correciton
  double GetFiretimesCorrection(int laserId, double speed);
  // compute optical center correction
  void GetDistanceCorrection(LidarOpticalCenter optical_center, int &azimuth, int &elevation, float &distance, DistanceCorrectionType type);
  // compute coordinate transformation
  void TransformPoint(float& x, float& y, float& z, const TransformParam& transform);
  // ensure the angle is between [0-360) with 1/25600 accuracy
  void CircleRevise(int &angle);

  // set frame azimuth
  virtual void SetFrameAzimuth(float frame_start_azimuth);
  // Statistical packet loss data
  void CalPktLoss(uint32_t PacketSeqnum, FrameDecodeParam param);
  void CalPktTimeLoss(uint64_t PacketTimestamp, FrameDecodeParam param);
  // crc
  void CRCInit();
  uint32_t CRCCalc(const uint8_t *bytes, int len, int zeros_num);
  // remake 
  void setRemakeDefaultConfig(LidarDecodedFrame<T_Point> &frame);
  void DoRemake(int azi, int elev, const RemakeConfig& rq, int& point_idx);
  // update right memory space
  virtual void setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame);
  // used for updating cuda correction and firetime by itself
  uint32_t* getCorrectionLoadSequenceNum() { return &correction_load_sequence_num_; }
  uint32_t* getFiretimeLoadSequenceNum() { return &firetime_load_sequence_num_; }
  bool* getCorrectionLoadFlag() { return &get_correction_file_; }
  bool* getFiretimeLoadFlag() { return &get_firetime_file_; }
  void loadCorrectionSuccess() { get_correction_file_ = true; correction_load_sequence_num_++; }
  void loadFiretimeSuccess() { get_firetime_file_ = true; firetime_load_sequence_num_++; }



  PacketSeqnumLossMessage seqnum_loss_message_;
  PacketTimeLossMessage time_loss_message_;

 protected:
  float cos_all_angle_[CIRCLE];
  float sin_all_angle_[CIRCLE];
  CorrectionData correction;
  float firetime_correction_[DEFAULT_MAX_LASER_NUM];
  bool get_correction_file_ = false;
  bool get_firetime_file_ = false;
  uint32_t correction_load_sequence_num_ = 0;
  uint32_t firetime_load_sequence_num_ = 0;
  int32_t last_azimuth_;
  int32_t last_last_azimuth_;
  std::atomic<uint32_t> compute_packet_num;
  int rotation_flag;
  std::string lidar_type_;
  uint16_t frame_start_azimuth_uint16_;
  LidarOpticalCenter optical_center;
  RemakeConfig default_remake_config;
  uint32_t m_CRCTable[256];
  bool crc_initialized;
  LastUtcTime last_utc_time;
};
}
}
#include "general_parser.cc"

#endif  // GENERAL_PARSER_H_
