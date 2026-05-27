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
 * File:       udp4_3_parser.cc
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Implemente Udp4_3Parser class
*/

#include "udp4_3_parser.h"

using namespace hesai::lidar;
template<typename T_Point>
Udp4_3Parser<T_Point>::Udp4_3Parser() {
  use_angle_ = true;
  this->default_remake_config.min_azi = 27.0f;
  this->default_remake_config.max_azi = 153.0f;
  this->default_remake_config.ring_azi_resolution = 0.1f;
  this->default_remake_config.max_azi_scan = 1260;   // (max_azi - min_azi) / ring_azi_resolution
  this->default_remake_config.min_elev = -13.f;
  this->default_remake_config.max_elev = 13.f;
  this->default_remake_config.ring_elev_resolution = 0.2f;
  this->default_remake_config.max_elev_scan = 130;   // (max_elev - min_elev) / ring_elev_resolution
  LogInfo("init 4_3 parser");
}

template<typename T_Point>
Udp4_3Parser<T_Point>::~Udp4_3Parser() { LogInfo("release 4_3 Parser"); }

template <typename T_Point>
void Udp4_3Parser<T_Point>::setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame) {
  frame.resetMalloc(1800, 256);
}

template<typename T_Point>
void Udp4_3Parser<T_Point>::LoadCorrectionFile(const std::string& lidar_correction_file) {
  try {
    LogInfo("load correction file from local correction.csv now!");
    std::ifstream fin(lidar_correction_file, std::ios::binary);
    if (fin.is_open()) {
      LogDebug("Open correction file success");
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
        LogError("Parse local Correction file Error");
      } else {
        LogInfo("Parse local Correction file Success!!!");
      }
    } else {
      LogError("Open correction file failed");
      return;
    }
  } catch (const std::exception& e) {
    LogFatal("error loading correction file: %s", e.what());
  }
}

template<typename T_Point>
int Udp4_3Parser<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  try {
    if (len < 32) {
      throw std::invalid_argument("correction string length is too short");
    }
    const char *p = correction_string;
    AT::ATCorrections_Header header = *(AT::ATCorrections_Header *)p;
    if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
      switch (header.min_version) {
        case 5: {
          memcpy((void *)&AT_corrections.header, p, sizeof(AT::ATCorrections_Header));
          p += sizeof(AT::ATCorrections_Header);
          AT_corrections.channel_number = *((uint8_t *)p);
          p++;
          AT_corrections.mirror_number = *((uint8_t *)p);
          p++;
          AT_corrections.frame_number = *((uint8_t *)p);
          p++;
          if (AT_corrections.channel_number > AT::AT_MAX_CHANNEL_NUM || AT_corrections.mirror_number > AT::AT_MAX_MIRROR_NUM) {
            throw std::invalid_argument("correction error, channel_number or mirror number out of range");
          }
          if (static_cast<size_t>(len) < 16 + sizeof(uint32_t) * AT_corrections.mirror_number * 2 + sizeof(int32_t) * AT_corrections.channel_number * 2 + 
              sizeof(int8_t) * 128 * 180 * 2 + 32) {
            throw std::invalid_argument("correction string length is too short");
          }
          memcpy((void *)&AT_corrections.frame_config_byte, p,
                 sizeof(uint8_t) * 8);
          p += sizeof(uint8_t) * 8;
          AT_corrections.angle_division = *((uint8_t *)p);
          p++;
          memcpy((void *)&AT_corrections.start_frame, p,
                 sizeof(uint32_t) * AT_corrections.mirror_number);
          p += sizeof(uint32_t) * AT_corrections.mirror_number;
          memcpy((void *)&AT_corrections.end_frame, p,
                 sizeof(uint32_t) * AT_corrections.mirror_number);
          p += sizeof(uint32_t) * AT_corrections.mirror_number;
          memcpy((void *)&AT_corrections.azimuths, p,
                 sizeof(int32_t) * AT_corrections.channel_number);
          p += sizeof(int32_t) * AT_corrections.channel_number;
          memcpy((void *)&AT_corrections.elevations, p,
                 sizeof(int32_t) * AT_corrections.channel_number);
          p += sizeof(int32_t) * AT_corrections.channel_number;
          int adjust_length = 128 * 180;
          memcpy((void *)&AT_corrections.azimuth_adjust, p,
                 sizeof(int8_t) * adjust_length);
          p += sizeof(int8_t) * adjust_length;
          memcpy((void *)&AT_corrections.elevation_adjust, p,
                 sizeof(int8_t) * adjust_length);
          p += sizeof(int8_t) * adjust_length;
          memcpy((void *)&AT_corrections.SHA_value, p,
                 sizeof(uint8_t) * 32);
          p += sizeof(uint8_t) * 32;
        } break;
        default:
          throw std::invalid_argument("min_version is wrong!");
          break;
      }
    } else {
        throw std::invalid_argument("file delimiter is error");
    }
  } catch (const std::exception &e) {
    this->get_correction_file_ = false;
    LogFatal("load correction error: %s", e.what());
    return -1;
  }
  if (AT_corrections.setToFloatUseAngleDivision()) {
    this->loadCorrectionSuccess();
    return 0;
  }
  return -1;
}

template<typename T_Point>
void Udp4_3Parser<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  LogWarning("don't support firetimes file");
}

template <typename T_Point>
void* Udp4_3Parser<T_Point>::getStruct(const int type) {
  if (type == CORRECTION_STRUCT)
    return (void*)&(AT_corrections);
  else if (type == FIRETIME_STRUCT)
    return nullptr;
  return nullptr;
}

template <typename T_Point>
int Udp4_3Parser<T_Point>::getDisplay(bool **display) {
  *display = AT_corrections.display;
  return AT::AT_MAX_CHANNEL_NUM;
}

template<typename T_Point>
bool Udp4_3Parser<T_Point>::IsNeedFrameSplit(uint16_t azimuth) {
  if (abs(azimuth - this->last_azimuth_) > AT::AT_AZIMUTH_TOLERANCE &&
      this->last_azimuth_ != 0) {
    return true;
  }
  return false;
}

template<typename T_Point>
int Udp4_3Parser<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index) {
  if ((packet_index + 1) * frame.per_points_num >= frame.maxPackerPerFrame * frame.maxPointPerPacket) {
    LogFatal("total points exceeded memory storage limit, packet_index: %d", packet_index + 1);
    GeneralParser<T_Point>::FrameNumAdd();
    return -1;
  }
  int point_index = packet_index * frame.per_points_num;
  int point_num = 0;
  auto& packetData = frame.packetData[packet_index];
  for (int i = point_index; i < point_index + static_cast<int>(frame.valid_points[packet_index]); i++) {
    auto& pointData = frame.pointData[i];
    float raw_azimuth = pointData.azimuth;
    float raw_elevation = 0;
    int field = pointData.data.dAT.field;
    float distance = static_cast<float>(pointData.distance * frame.distance_unit);
    if (this->get_correction_file_) {
      {
        raw_elevation = AT_corrections.floatCorr.f_elevations[pointData.channel_index] +
                        AT_corrections.getElevationAdjust(pointData.channel_index, raw_azimuth);
        raw_azimuth = (raw_azimuth + kCircle - AT_corrections.floatCorr.start_frame[field] / kAllFineResolutionFloat) * 2 -
                          AT_corrections.floatCorr.f_azimuths[pointData.channel_index] +
                          AT_corrections.getAzimuthAdjust(pointData.channel_index, raw_azimuth);
      }
    }
    int elevation = floatToInt(raw_elevation * kAllFineResolutionInt);
    int azimuth = floatToInt(raw_azimuth * kAllFineResolutionInt);
    this->CircleRevise(azimuth);
    this->CircleRevise(elevation);
    if (frame.fParam.config.fov_start != -1 && frame.fParam.config.fov_end != -1) {
      int fov_transfer = azimuth / kAllFineResolutionInt;
      if (fov_transfer < frame.fParam.config.fov_start || fov_transfer > frame.fParam.config.fov_end) { //不在fov范围continue
        continue;
      }
    }
    float xyDistance = distance * this->cos_all_angle_[(elevation)];
    float x = xyDistance * this->sin_all_angle_[(azimuth)];
    float y = xyDistance * this->cos_all_angle_[(azimuth)];
    float z = distance * this->sin_all_angle_[(elevation)];
    this->TransformPoint(x, y, z, frame.fParam.transform);


    int point_index_rerank = point_index + point_num; 
    GeneralParser<T_Point>::DoRemake(azimuth, elevation, frame.fParam.remake_config, point_index_rerank); 
    if(point_index_rerank >= 0) { 
      auto& ptinfo = frame.points[point_index_rerank]; 
      set_x(ptinfo, x); 
      set_y(ptinfo, y); 
      set_z(ptinfo, z); 
      set_ring(ptinfo, pointData.channel_index); 
      set_intensity(ptinfo, pointData.reflectivity);  
      set_timestamp(ptinfo, double(packetData.t.sensor_timestamp) / kMicrosecondToSecond);
      set_confidence(ptinfo, pointData.data.dAT.confidence);

      point_num++;
    }
  }
  frame.valid_points[packet_index] = point_num;
  GeneralParser<T_Point>::FrameNumAdd();
  return 0;
}

template<typename T_Point>
int Udp4_3Parser<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index) {
  if(udpPacket.is_timeout == true) {
    for(int i = 0; i < 3; i++) {
      int angle = AT_corrections.floatCorr.start_frame[i] / kFineResolutionInt + AT::AT_MARGINAL_ANGLE - this->last_azimuth_;
      if (abs(angle) < AT::AT_ACCEPTANCE_ANGLE) {
        use_angle_ = false;
        frame.scan_complete = true;
        return 0;
      }
    }
  }
  uint32_t packet_index_use = (packet_index >= 0 ? packet_index : frame.packet_num);
  if (udpPacket.buffer[0] != 0xEE || udpPacket.buffer[1] != 0xFF ||
      udpPacket.buffer[2] != 4 || udpPacket.buffer[3] != 3) {
    LogDebug("Invalid point cloud");
    return -1;
  }
  const HS_LIDAR_HEADER_ST_V3 *pHeader =
      reinterpret_cast<const HS_LIDAR_HEADER_ST_V3 *>(
          &(udpPacket.buffer[0]) + sizeof(HS_LIDAR_PRE_HEADER));

  const HS_LIDAR_TAIL_ST_V3 *pTail =
      reinterpret_cast<const HS_LIDAR_TAIL_ST_V3 *>(
          (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V3) +
          (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3) +
           sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3) +
           sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V3) * pHeader->GetLaserNum()) *
              pHeader->GetBlockNum() +
          sizeof(HS_LIDAR_BODY_CRC_ST_V3));
  if (pHeader->HasSeqNum()) {
    const HS_LIDAR_TAIL_SEQ_NUM_ST_V3 *pTailSeqNum =
        reinterpret_cast<const HS_LIDAR_TAIL_SEQ_NUM_ST_V3 *>(
            (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V3) +
            (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3) +
             sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3) +
             sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V3) * pHeader->GetLaserNum()) *
                pHeader->GetBlockNum() +
            sizeof(HS_LIDAR_BODY_CRC_ST_V3) + sizeof(HS_LIDAR_TAIL_ST_V3));
    this->CalPktLoss(pTailSeqNum->GetSeqNum(), frame.fParam);
  } 
  this->CalPktTimeLoss(pTail->GetMicroLidarTimeU64(this->last_utc_time), frame.fParam);    

  frame.scan_complete = false;
  frame.block_num = pHeader->GetBlockNum();
  frame.laser_num = pHeader->GetLaserNum();
  frame.per_points_num = pHeader->GetBlockNum() * pHeader->GetLaserNum();
  frame.distance_unit = pHeader->GetDistUnit();
  if (frame.per_points_num > frame.maxPointPerPacket) {
    LogFatal("per_points_num(%u) out of %d", frame.per_points_num, frame.maxPointPerPacket);
    return -1;
  }
  if (frame.laser_num > AT::AT_MAX_CHANNEL_NUM) {
    LogFatal("laser_num(%u) out of %d", frame.laser_num, AT::AT_MAX_CHANNEL_NUM);
    return -1;
  }

  if (frame.fParam.use_timestamp_type == 0) {
    frame.packetData[packet_index_use].t.sensor_timestamp = pTail->GetMicroLidarTimeU64(this->last_utc_time);
  } else {
    frame.packetData[packet_index_use].t.sensor_timestamp = udpPacket.recv_timestamp;
  }   
  if (frame.frame_start_timestamp == 0) frame.frame_start_timestamp = double(frame.packetData[packet_index_use].t.sensor_timestamp) / kMicrosecondToSecond;
  frame.frame_end_timestamp = double(frame.packetData[packet_index_use].t.sensor_timestamp) / kMicrosecondToSecond;
  if (frame.fParam.pcap_time_synchronization) frame.host_timestamp = GetMicroTickCountU64();
  frame.work_mode = pTail->m_u8Shutdown;
  frame.return_mode = pTail->GetReturnMode();

  const HS_LIDAR_BODY_AZIMUTH_ST_V3 *pAzimuth =
      reinterpret_cast<const HS_LIDAR_BODY_AZIMUTH_ST_V3 *>(
          (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V3));
  uint16_t u16Azimuth = pAzimuth->GetAzimuth();
  if (IsNeedFrameSplit(u16Azimuth)) {
    //超时分帧的情况下，this->last_azimuth_肯定和u16Azimuth差值很大
    //此时通过use_angle_来避免进行错误分帧的情况。
    if(this->use_angle_ == false){
      this->use_angle_ = true;
    }
    else{
      //未设置true会一直显示loading,几秒后超时退出崩溃
      frame.scan_complete = true;
    }
  }
  this->last_azimuth_ = u16Azimuth;    
  if (frame.scan_complete)
    return 0;
  
  int index = packet_index_use * frame.per_points_num;
  int point_num = 0;
  for (int blockid = 0; blockid < pHeader->GetBlockNum(); blockid++) {
    int current_block_echo_count = ((pHeader->GetEchoCount() > 0 && pHeader->GetEchoNum() > 0) ?
            ((pHeader->GetEchoCount() - 1 + blockid) % pHeader->GetEchoNum() + 1) : 0);
    if (frame.fParam.echo_mode_filter != 0 && current_block_echo_count != 0 && frame.fParam.echo_mode_filter != current_block_echo_count) {
      continue;
    }
    const HS_LIDAR_BODY_AZIMUTH_ST_V3 *pAzimuth =
        reinterpret_cast<const HS_LIDAR_BODY_AZIMUTH_ST_V3 *>(
            (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V3) +
            (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3) + sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3) +
            sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V3) * pHeader->GetLaserNum()) * blockid);
    const HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3 *pFineAzimuth =
        reinterpret_cast<const HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3 *>(
            (const unsigned char *)pAzimuth +
            sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3));
    const HS_LIDAR_BODY_CHN_NNIT_ST_V3 *pChnUnit =
        reinterpret_cast<const HS_LIDAR_BODY_CHN_NNIT_ST_V3 *>(
            (const unsigned char *)pAzimuth +
            sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3) +
            sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3));
    uint16_t u16Azimuth = pAzimuth->GetAzimuth();
    uint8_t u8FineAzimuth = pFineAzimuth->GetFineAzimuth();
    int azimuth = u16Azimuth * kFineResolutionInt + u8FineAzimuth;
    int field = 0;
    if ( this->get_correction_file_) {
      int count = 0; 
      while (count < AT_corrections.frame_number &&
             (((azimuth + CIRCLE - AT_corrections.floatCorr.start_frame[field]) % CIRCLE +
             (AT_corrections.floatCorr.end_frame[field] + CIRCLE - azimuth) % CIRCLE) !=
             (AT_corrections.floatCorr.end_frame[field] + CIRCLE - AT_corrections.floatCorr.start_frame[field]) % CIRCLE)) {
        field = (field + 1) % AT_corrections.frame_number;
        count++;
      }
      if (count >= AT_corrections.frame_number) continue;
    }
    for (int i = 0; i < pHeader->GetLaserNum(); i++) {
      if (AT_corrections.display[i] == false) {
        pChnUnit++;
        continue;
      }
      frame.pointData[index].channel_index = i;
      frame.pointData[index].azimuth = static_cast<float>(azimuth) / kAllFineResolutionFloat;
      frame.pointData[index].distance = pChnUnit->GetDistance();
      frame.pointData[index].reflectivity = pChnUnit->GetReflectivity(); 
      frame.pointData[index].data.dAT.confidence = pChnUnit->GetConfidenceLevel();  
      frame.pointData[index].data.dAT.field = field;
      frame.pointData[index].data.dAT.azimuth_offset = 0;
      pChnUnit++;
      index++;
      point_num++;
    }
  }
  frame.valid_points[packet_index_use] = point_num;
  frame.packet_num++;
  return 0;
}

template<typename T_Point>
int Udp4_3Parser<T_Point>::ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info) {
  FaultMessageVersion4_3 *fault_message_ptr =  
      reinterpret_cast< FaultMessageVersion4_3*> (&(udp_packet.buffer[0]));
  fault_message_ptr->ParserFaultMessage(fault_message_info, this->last_utc_time);
  return 0;
}

