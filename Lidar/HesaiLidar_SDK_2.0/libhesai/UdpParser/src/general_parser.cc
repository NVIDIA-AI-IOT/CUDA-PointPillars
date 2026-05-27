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
 * File:       general_parser.cc
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Implemente GeneralParser class
*/

#include "general_parser.h"
#include <iomanip> 

using namespace hesai::lidar;
template <typename T_Point>
GeneralParser<T_Point>::GeneralParser() {
  this->default_remake_config.min_azi = 0.f;
  this->default_remake_config.max_azi = 360.f;
  this->default_remake_config.ring_azi_resolution = 0.2f;
  this->default_remake_config.max_azi_scan = 1800;   // (max_azi - min_azi) / ring_azi_resolution
  this->default_remake_config.min_elev = -25.f;
  this->default_remake_config.max_elev = 15.f;
  this->default_remake_config.ring_elev_resolution = 0.2f;
  this->default_remake_config.max_elev_scan = 200;   // (max_elev - min_elev) / ring_elev_resolution
  last_azimuth_ = 0;
  last_last_azimuth_ = 0;
  crc_initialized = false;
  for (int i = 0; i < CIRCLE; ++i) {
    sin_all_angle_[i] = std::sin(i * 2 * M_PI / CIRCLE);
    cos_all_angle_[i] = std::cos(i * 2 * M_PI / CIRCLE);
  }
}


template <typename T_Point>
int GeneralParser<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index) {
  (void)frame;
  (void)packet_index;
  return 0;
}
template <typename T_Point>
void GeneralParser<T_Point>::FrameNumAdd() {
  compute_packet_num++;
}
template <typename T_Point>
int GeneralParser<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index) {
  (void)frame;
  (void)udpPacket;
  return -1;
} 
template <typename T_Point>
int GeneralParser<T_Point>::ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info) {
  (void)udp_packet;
  (void)fault_message_info;
  return -1;
}

template <typename T_Point>
GeneralParser<T_Point>::~GeneralParser() { LogInfo("release general parser"); }

template <typename T_Point>
void GeneralParser<T_Point>::SetFrameAzimuth(float frame_start_azimuth) {
  // Determine frame_start_azimuth [0,360)
  if (frame_start_azimuth < 0.0f || frame_start_azimuth >= 360.0f) {
    frame_start_azimuth = 0.0f;
  }
  frame_start_azimuth_uint16_ = frame_start_azimuth * kResolutionInt;
}
template <typename T_Point>
void GeneralParser<T_Point>::LoadCorrectionFile(const std::string& correction_path) {
  try {
    std::ifstream fin(correction_path, std::ios::in);
    if (fin.is_open()) {
      int length = 0;
      fin.seekg(0, std::ios::end);
      length = static_cast<int>(fin.tellg());
      fin.seekg(0, std::ios::beg);
      char *buffer = new char[length];
      fin.read(buffer, length);
      fin.close();
      int ret = LoadCorrectionString(buffer, length);
      delete[] buffer;
      if (ret != 0) {
        LogError("Parse local correction file Error");
      } else {
        LogInfo("Parser correction file success!");
      }
    } else {
      LogError("Open correction file failed");
      return;
    }
  } catch (const std::exception& e) {
    LogFatal("error loading correction file: %s", e.what());
    return;
  }
}
template <typename T_Point>
int GeneralParser<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  try {
    std::string correction_content_str(correction_string, len);
    std::istringstream ifs(correction_content_str);
    std::string line;

    // skip first line "Laser id,Elevation,Azimuth" or "eeff"
    std::getline(ifs, line);  
    float elevation_list[DEFAULT_MAX_LASER_NUM], azimuth_list[DEFAULT_MAX_LASER_NUM];
    std::vector<std::string> vfirstLine;
    split_string(vfirstLine, line, ',');
    if (vfirstLine[0] == "EEFF" || vfirstLine[0] == "eeff") {
      // skip second line
      std::getline(ifs, line);  
    }

    int lineCount = 0;
    while (std::getline(ifs, line)) {
      std::vector<std::string> vLineSplit;
      split_string(vLineSplit, line, ',');
      // skip error line or hash value line
      if (vLineSplit.size() != 3) {  
        continue;
      } else {
        lineCount++;
      }
      float elevation, azimuth;
      int laserId = 0;
      laserId = std::stoi(vLineSplit[0]);
      elevation = std::stof(vLineSplit[1]);
      azimuth = std::stof(vLineSplit[2]);

      if (laserId > DEFAULT_MAX_LASER_NUM || laserId <= 0) {
        throw std::invalid_argument("laser id is wrong in correction file. laser Id: "   
                                      + std::to_string(laserId) + ", line: " + std::to_string(lineCount));
      }
      if (laserId != lineCount) {
        LogWarning("laser id is wrong in correction file. laser Id: %d, line: %d.  continue", laserId, lineCount);
        lineCount--;
        continue;
      }
      elevation_list[laserId - 1] = elevation;
      azimuth_list[laserId - 1] = azimuth;
    }

    for (int i = 0; i < lineCount; ++i) {
      this->correction.elevation[i] = elevation_list[i];
      this->correction.azimuth[i] = azimuth_list[i];
    }
    this->loadCorrectionSuccess();
  } catch (const std::exception &e) {
    LogFatal("load correction error: %s", e.what());
    this->get_correction_file_ = false;
    return -1;
  }
  return 0;
}

template <typename T_Point>
void GeneralParser<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  try {
    std::ifstream inFile(firetimes_path, std::ios::in);
    if (inFile.is_open()) {
      std::string lineStr;
      //skip first line
      std::getline(inFile, lineStr); 
      int lineCount = 0;
      while (getline(inFile, lineStr)) {
        std::vector<std::string> vLineSplit;
        split_string(vLineSplit, lineStr, ',');
        // skip error line or hash value line
        if (vLineSplit.size() != 2) {  
          continue;
        } else {
          lineCount++;
        }
        float deltTime = 0.f;
        int laserId = 0;
        laserId = std::stoi(vLineSplit[0]);
        deltTime = std::stof(vLineSplit[1]);
        if (laserId > DEFAULT_MAX_LASER_NUM || laserId <= 0) {
          throw std::invalid_argument("laser id is wrong in firetimes file. laser Id: "   
                                      + std::to_string(laserId) + ", line: " + std::to_string(lineCount));
        }
        firetime_correction_[laserId - 1] = deltTime;
      }
      this->loadFiretimeSuccess();
      LogInfo("Open firetime file success!");
      inFile.close();
      return;
    } else {
      throw std::invalid_argument("Open firetime file failed");
    }
  } catch (const std::exception &e) {
    LogFatal("load firetime error: %s", e.what());
    this->get_firetime_file_ = false;
    return;
  }
}

template <typename T_Point>
int GeneralParser<T_Point>::LoadChannelConfigFile(const std::string channel_config_path) {
  (void)channel_config_path;
  LogWarning("don't support channel config file");
  return -1;
}

template <typename T_Point>
void* GeneralParser<T_Point>::getStruct(const int type) { 
  if (type == CORRECTION_STRUCT)
    return (void*)&correction;
  else if (type == FIRETIME_STRUCT)
    return (void*)firetime_correction_;
  return nullptr;
}

template <typename T_Point>
int GeneralParser<T_Point>::getDisplay(bool **display) {
  *display = correction.display;
  return DEFAULT_MAX_LASER_NUM;
}

template <typename T_Point>
void GeneralParser<T_Point>::CircleRevise(int &angle) {
  while (angle < 0) {
    angle += CIRCLE;
  }
  while (angle >= CIRCLE) {
    angle -= CIRCLE;
  }
}

template <typename T_Point>
int GeneralParser<T_Point>::LoadFiretimesString(const char *firetimes_string, int len) {
  (void)firetimes_string;
  LogWarning("don't load firetimes string");
  return 0;
}

template <typename T_Point>
double GeneralParser<T_Point>::GetFiretimesCorrection(int laserId, double speed) {
  return this->firetime_correction_[laserId] * speed * 6E-6;
}

template<typename T_Point>
bool GeneralParser<T_Point>::IsNeedFrameSplit(uint16_t azimuth, FrameDecodeParam &param) {
  // The first two packet dont have the information of last_azimuth_  and last_last_azimuth, so do not need split frame
  // The initial value of last_azimuth_ is -1
  // Determine the rotation direction and division
  
  int32_t division = 0;
  // If last_last_azimuth_ != -1，the packet is the third, so we can determine whether the current packet requires framing
  if (this->last_last_azimuth_ != -1) 
  {
    // Get the division
    int32_t division1 = abs(this->last_azimuth_ - this->last_last_azimuth_);
    int32_t division2 = abs(this->last_azimuth_ - azimuth);
    division = division1 > division2 ? division2 : division1 ;
    // Prevent two consecutive packets from having the same angle when causing an error in framing
    if ( division == 0) return false;
    // In the three consecutive angle values, if the angle values appear by the division of the decreasing situation,it must be reversed
    // The same is true for FOV
    if( this->last_last_azimuth_ - this->last_azimuth_ == division || this->last_azimuth_ - azimuth == division)
    {
      param.UpdateRotation(-1);
    } else {
      param.UpdateRotation(1);
    }
  } else {
    // The first  and second packet do not need split frame
    return false;
  }
  if (param.rotation_flag > 0) {
    if (this->last_azimuth_- azimuth > division)
    {
      if (frame_start_azimuth_uint16_ > this->last_azimuth_ || frame_start_azimuth_uint16_ <= azimuth) {
        return true;
      } 
      return false;
    }  
    if (this->last_azimuth_ < azimuth && this->last_azimuth_ < frame_start_azimuth_uint16_ 
        && azimuth >= frame_start_azimuth_uint16_) {
      return true;
    }
    return false;
  } else {
    if (azimuth - this->last_azimuth_ > division)
    {
      if (frame_start_azimuth_uint16_ <= this->last_azimuth_ || frame_start_azimuth_uint16_ > azimuth) {
        return true;
      } 
      return false;
    }  
    if (this->last_azimuth_ > azimuth && this->last_azimuth_ > frame_start_azimuth_uint16_ 
        && azimuth <= frame_start_azimuth_uint16_) {
      return true;
    }
    return false;
  }
}

template <typename T_Point>
void GeneralParser<T_Point>::GetDistanceCorrection(LidarOpticalCenter optical_center, int &azimuth, int &elevation, float &distance, DistanceCorrectionType type) {
  if (distance <= 0.09) return;
  CircleRevise(azimuth);
  CircleRevise(elevation);
  float tx = this->cos_all_angle_[elevation] * this->sin_all_angle_[azimuth];
  float ty = this->cos_all_angle_[elevation] * this->cos_all_angle_[azimuth];
  float tz = this->sin_all_angle_[elevation];
  float d  = distance;
  if (type == GeometricCenter) {
    float B = 2 * tx * optical_center.x + 2 * ty * optical_center.y + 2 * tz * optical_center.z;
    float C = optical_center.x * optical_center.x + optical_center.y * optical_center.y + optical_center.z * optical_center.z - d * d;
    float d_opitcal = std::sqrt(B * B / 4 - C) - B / 2;
    float x = d_opitcal * tx + optical_center.x;
    float y = d_opitcal * ty + optical_center.y;
    float z = d_opitcal * tz + optical_center.z;
    azimuth = static_cast<int>(std::atan2(x, y) * kHalfCircleFloat * kFineResolutionFloat / M_PI);
    elevation = static_cast<int>(std::asin(z / d) * kHalfCircleFloat * kFineResolutionFloat / M_PI);
    distance = d;
  } else if (type == OpticalCenter) {
    float x = d * tx + optical_center.x;
    float y = d * ty + optical_center.y;
    float z = d * tz + optical_center.z;
    float d_geometric_center = std::sqrt(x * x + y * y + z * z);
    azimuth = static_cast<int>(std::atan2(x, y) * kHalfCircleFloat * kFineResolutionFloat / M_PI);
    elevation = static_cast<int>(std::asin(z / d_geometric_center) * kHalfCircleFloat * kFineResolutionFloat / M_PI);
    distance = d_geometric_center;
  } else {
    // It should never have been executed here.
  }
  CircleRevise(azimuth);
  CircleRevise(elevation);
}

template <typename T_Point>
void GeneralParser<T_Point>::TransformPoint(float& x, float& y, float& z, const TransformParam& transform)
{
  if (transform.use_flag == false) return;

  float cosa = std::cos(transform.roll);
  float sina = std::sin(transform.roll);
  float cosb = std::cos(transform.pitch);
  float sinb = std::sin(transform.pitch);
  float cosc = std::cos(transform.yaw);
  float sinc = std::sin(transform.yaw);

  float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
              (sina * sinc + cosa * sinb * cosc) * z + transform.x;
  float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
              (cosa * sinb * sinc - sina * cosc) * z + transform.y;
  float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;
  x = x_;
  y = y_;
  z = z_; 
}

template <typename T_Point>
void GeneralParser<T_Point>::CalPktLoss(uint32_t PacketSeqnum, FrameDecodeParam param) {
  if (param.enable_packet_loss_tool_ == false) {
    return;
  }
  this->seqnum_loss_message_.is_packet_loss = false;
  if (this->seqnum_loss_message_.is_init == false) {
    this->seqnum_loss_message_.loss_count = 0;
    this->seqnum_loss_message_.total_loss_count = 0;
    this->seqnum_loss_message_.start_time = GetMicroTickCount();
    this->seqnum_loss_message_.last_total_package_count = 0;
    this->seqnum_loss_message_.total_packet_count = 0;
    this->seqnum_loss_message_.last_seqnum = PacketSeqnum;
    this->seqnum_loss_message_.is_init = true;
    return;
  }
  uint32_t diff = 0;
  if (PacketSeqnum >= this->seqnum_loss_message_.last_seqnum) {
    diff = PacketSeqnum - this->seqnum_loss_message_.last_seqnum;
  } else {
    diff = (this->seqnum_loss_message_.max_sequence - this->seqnum_loss_message_.last_seqnum) + PacketSeqnum + 1;
  }
  this->seqnum_loss_message_.total_packet_count += diff;
  
  if (diff > 1) {
    this->seqnum_loss_message_.loss_count += (diff - 1);
    this->seqnum_loss_message_.total_loss_count += (diff - 1);
    this->seqnum_loss_message_.is_packet_loss = true;
  }
  // print log every 1s
  if (this->seqnum_loss_message_.loss_count != 0 && GetMicroTickCount() - this->seqnum_loss_message_.start_time >= 1 * 1000 * 1000) {
    LogWarning("pkt loss freq: %u/%u", this->seqnum_loss_message_.loss_count,
            this->seqnum_loss_message_.total_packet_count - this->seqnum_loss_message_.last_total_package_count);
    this->seqnum_loss_message_.loss_count = 0;
    this->seqnum_loss_message_.start_time = GetMicroTickCount();
    this->seqnum_loss_message_.last_total_package_count = this->seqnum_loss_message_.total_packet_count;
  }
  this->seqnum_loss_message_.last_seqnum = PacketSeqnum;
}

template <typename T_Point>
void GeneralParser<T_Point>::CalPktTimeLoss(uint64_t PacketTimestamp, FrameDecodeParam param) {
  if(param.enable_packet_timeloss_tool_ == false){
    return;
  } 
  if(param.packet_timeloss_tool_continue_ == false && this->time_loss_message_.total_timeloss_count != 0){    
    return;
  }
  if (this->time_loss_message_.is_init == false) {
    this->time_loss_message_.timeloss_count = 0;
    this->time_loss_message_.total_timeloss_count = 0;
    this->time_loss_message_.timeloss_start_time = GetMicroTickCount();
    this->time_loss_message_.last_timestamp = PacketTimestamp;
    this->time_loss_message_.last_total_package_count = this->seqnum_loss_message_.total_packet_count;
    this->time_loss_message_.is_init = true;
    return;
  }
  // packet time loss reset
  else if(this->seqnum_loss_message_.is_packet_loss) {
    LogWarning("pkt time loss freq: %u/%u", this->time_loss_message_.timeloss_count, this->seqnum_loss_message_.total_packet_count - this->time_loss_message_.last_total_package_count);
    this->time_loss_message_.timeloss_count = 0;
    this->time_loss_message_.timeloss_start_time = GetMicroTickCount();
    this->time_loss_message_.last_timestamp = PacketTimestamp;
    this->time_loss_message_.last_total_package_count = this->seqnum_loss_message_.total_packet_count;
    return;
  }
  if (PacketTimestamp <= this->time_loss_message_.last_timestamp) {
    this->time_loss_message_.timeloss_count++;
    this->time_loss_message_.total_timeloss_count++;
  }
  // print log every 1s
  if (this->time_loss_message_.timeloss_count != 0 && GetMicroTickCount() - this->time_loss_message_.timeloss_start_time >= 1 * 1000 * 1000) {
    LogWarning("pkt time loss freq: %u/%u", this->time_loss_message_.timeloss_count,
           this->seqnum_loss_message_.total_packet_count - this->time_loss_message_.last_total_package_count);
    this->time_loss_message_.timeloss_count = 0;
    this->time_loss_message_.timeloss_start_time = GetMicroTickCount();
    this->time_loss_message_.last_total_package_count = this->seqnum_loss_message_.total_packet_count;
  }
  this->time_loss_message_.last_timestamp = PacketTimestamp;    
}

template <typename T_Point>
uint32_t GeneralParser<T_Point>::CRCCalc(const uint8_t *bytes, int len, int zeros_num) {
  CRCInit();
  uint32_t i_crc = 0xffffffff;
  for (int i = 0; i < len; i++)
      i_crc = (i_crc << 8) ^ m_CRCTable[((i_crc >> 24) ^ bytes[i]) & 0xff];
  for (int i = 0; i < zeros_num; i++)
      i_crc = (i_crc << 8) ^ m_CRCTable[((i_crc >> 24) ^ 0) & 0xff];
  return i_crc;
}

template <typename T_Point>
void GeneralParser<T_Point>::CRCInit() {
  if (crc_initialized) {
      return;
  }
  crc_initialized = true;

  uint32_t i, j;
  for (i = 0; i < 256; i++) {
      uint32_t k = 0;
      for (j = (i << 24) | 0x800000; j != 0x80000000; j <<= 1)
          k = (k << 1) ^ (((k ^ j) & 0x80000000) ? 0x04c11db7 : 0);
      m_CRCTable[i] = k;
  }
}

template <typename T_Point>
void GeneralParser<T_Point>::setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame) {
  frame.resetMalloc(5000, 1024);
}

template <typename T_Point>
void GeneralParser<T_Point>::setRemakeDefaultConfig(LidarDecodedFrame<T_Point> &frame) {
  auto &rq = frame.fParam.remake_config;
  if (rq.flag == false) return;
  if (rq.min_azi < 0) rq.min_azi = default_remake_config.min_azi;
  if (rq.max_azi < 0) rq.max_azi = default_remake_config.max_azi;
  if (rq.ring_azi_resolution < 0) rq.ring_azi_resolution = default_remake_config.ring_azi_resolution;
  if (rq.max_azi_scan < 0) rq.max_azi_scan = default_remake_config.max_azi_scan;
  if (rq.max_azi_scan > frame.maxPackerPerFrame) {
    LogError("LidarDecodedFrame(%u) space is too small(%d), please expand the packet space per frame", frame.maxPackerPerFrame, rq.max_azi_scan);
    rq.max_azi_scan = frame.maxPackerPerFrame;
  }
  if (rq.min_elev < 0) rq.min_elev = default_remake_config.min_elev;
  if (rq.max_elev < 0) rq.max_elev = default_remake_config.max_elev;
  if (rq.ring_elev_resolution < 0) rq.ring_elev_resolution = default_remake_config.ring_elev_resolution;
  if (rq.max_elev_scan < 0) rq.max_elev_scan = default_remake_config.max_elev_scan;
  if (rq.max_elev_scan > frame.maxPointPerPacket) {
    LogError("LidarDecodedFrame(%u) space is too small(%d), please expand the packet space per frame", frame.maxPointPerPacket, rq.max_elev_scan);
    rq.max_elev_scan = frame.maxPointPerPacket;
  }
}

template <typename T_Point>
void GeneralParser<T_Point>::DoRemake(int azi, int elev, const RemakeConfig& rq, int& point_idx) {
  if (rq.flag == false) return;
  float azi_ = azi / kAllFineResolutionFloat;
  float elev_ = elev / kAllFineResolutionFloat;
  elev_ = elev_ > 180.0 ? elev_ - 360.0 : elev_;
  point_idx = -1;
  int new_azi_iscan = (azi_ - rq.min_azi) / rq.ring_azi_resolution;
  int new_elev_iscan = (elev_ - rq.min_elev) / rq.ring_elev_resolution;
  if (new_azi_iscan >= 0 && new_azi_iscan < rq.max_azi_scan && new_elev_iscan >= 0 && new_elev_iscan < rq.max_elev_scan) {
    point_idx = new_azi_iscan * rq.max_elev_scan + new_elev_iscan;
  }
}


