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
#include "udp4_7_parser.h"

using namespace hesai::lidar;
template<typename T_Point>
Udp4_7Parser<T_Point>::Udp4_7Parser() {
  this->default_remake_config.min_azi = 27.0f;
  this->default_remake_config.max_azi = 153.0f;
  this->default_remake_config.ring_azi_resolution = 0.1f;
  this->default_remake_config.max_azi_scan = 1260;   // (max_azi - min_azi) / ring_azi_resolution
  this->default_remake_config.min_elev = -15.f;
  this->default_remake_config.max_elev = 10.f;
  this->default_remake_config.ring_elev_resolution = 0.1f;
  this->default_remake_config.max_elev_scan = 250;   // (max_elev - min_elev) / ring_elev_resolution
  LogInfo("init 4_7 parser");
}

template<typename T_Point>
Udp4_7Parser<T_Point>::~Udp4_7Parser() { LogInfo("release 4_7 Parser"); }

template <typename T_Point>
void Udp4_7Parser<T_Point>::setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame) {
  frame.resetMalloc(2500, 256);
}

template<typename T_Point>
void Udp4_7Parser<T_Point>::LoadCorrectionFile(const std::string& lidar_correction_file) {
  try {
    LogInfo("load correction file from local correction.dat now!");
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
int Udp4_7Parser<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  try {
    if (len < 32) {
      throw std::invalid_argument("correction string length is too short");
    }
    const char *p = correction_string;
    ATX::ATXCorrectionsHeader header = *(ATX::ATXCorrectionsHeader *)p;
    if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
      switch (header.version[1]) {
        case 1: {
          m_ATX_corrections.header = header;
          p += sizeof(ATX::ATXCorrectionsHeader);
          m_ATX_corrections.channel_number = *((uint8_t *)p);
          p++;
          m_ATX_corrections.angle_division = *((uint16_t *)p);
          p += sizeof(uint16_t);

          if (m_ATX_corrections.channel_number > ATX::ATX_MAX_CHANNEL_NUM || m_ATX_corrections.angle_division == 0) {
            throw std::invalid_argument("correction error, channel_number or angle_division out of range");
          }
          if (static_cast<size_t>(len) < 9 + sizeof(int16_t) * m_ATX_corrections.channel_number * 2 + 32) {
            throw std::invalid_argument("correction string length is too short");
          }
          memcpy((void *)&m_ATX_corrections.raw_azimuths, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_elevations, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void*)&m_ATX_corrections.SHA_value, p, 32);
        } break;
        case 2: {
          m_ATX_corrections.header = header;
          p += sizeof(ATX::ATXCorrectionsHeader);
          m_ATX_corrections.channel_number = *((uint8_t *)p);
          p++;
          m_ATX_corrections.angle_division = *((uint16_t *)p);
          p += sizeof(uint16_t);

          if (m_ATX_corrections.channel_number > ATX::ATX_MAX_CHANNEL_NUM || m_ATX_corrections.angle_division == 0) {
            throw std::invalid_argument("correction error, channel_number or angle_division out of range");
          }
          if (static_cast<size_t>(len) < 9 + sizeof(int16_t) * m_ATX_corrections.channel_number * 3 + 32) {
            throw std::invalid_argument("correction string length is too short");
          }
          memcpy((void *)&m_ATX_corrections.raw_azimuths_even, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_azimuths_odd, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_elevations, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void*)&m_ATX_corrections.SHA_value, p, 32);
        } break;
        case 3: {
          m_ATX_corrections.header = header;
          p += sizeof(ATX::ATXCorrectionsHeader);
          m_ATX_corrections.channel_number = *((uint8_t *)p);
          p++;
          m_ATX_corrections.angle_division = *((uint16_t *)p);
          p += sizeof(uint16_t);

          if (m_ATX_corrections.channel_number > ATX::ATX_MAX_CHANNEL_NUM || m_ATX_corrections.angle_division == 0) {
            throw std::invalid_argument("correction error, channel_number or angle_division out of range");
          }
          if (static_cast<size_t>(len) < 9 + sizeof(int16_t) * m_ATX_corrections.channel_number * 3 + sizeof(int16_t) * m_ATX_corrections.floatCorr.kLenElevationAdjust + 32) {
            throw std::invalid_argument("correction string length is too short");
          }
          memcpy((void *)&m_ATX_corrections.raw_azimuths_even, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_azimuths_odd, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_elevations, p,
                 sizeof(int16_t) * m_ATX_corrections.channel_number);
          p += sizeof(int16_t) * m_ATX_corrections.channel_number;
          memcpy((void *)&m_ATX_corrections.raw_elevations_adjust, p,
                 sizeof(int16_t) * m_ATX_corrections.floatCorr.kLenElevationAdjust);
          p += sizeof(int16_t) * m_ATX_corrections.floatCorr.kLenElevationAdjust;
          memcpy((void*)&m_ATX_corrections.SHA_value, p, 32);
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
  if(m_ATX_corrections.setToFloatUseAngleDivision()) {
    this->loadCorrectionSuccess();
    return 0;
  }
  return -1;
}

template <typename T_Point>
void Udp4_7Parser<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  try {
    int type = 0;
    size_t len = firetimes_path.length();
    if (len >= 4) {
      std::string extension = firetimes_path.substr(len - 4);
      if (extension == ".bin" || extension == ".dat") {
        type = 1; //  .bin
      } else if (extension == ".csv") {
        type = 2; //  .csv
      } else {
        type = 0; //  wrong
      }   
    }

    if (type == 0) {
      throw std::invalid_argument("firetime name is error(not .dat / .bin / .csv)");
    }

    if (type == 2) {
      std::ifstream inFile(firetimes_path, std::ios::in);
      if (inFile.is_open()) {
        std::string lineStr;
        //skip first line
        std::getline(inFile, lineStr); 
        while (getline(inFile, lineStr)) {
          std::stringstream ss(lineStr);
          std::string index, deltTime1, deltTime2;
          std::getline(ss, index, ',');
          std::getline(ss, deltTime1, ',');
          std::getline(ss, deltTime2, ',');
          int idx = std::stoi(index) - 1;
          if (idx < 0 || idx >= ATX::ATX_MAX_CHANNEL_NUM) throw std::invalid_argument("index out of range");
          m_ATX_firetimes.floatCorr.even_firetime_correction_[idx] = std::stof(deltTime1);
          m_ATX_firetimes.floatCorr.odd_firetime_correction_[idx] = std::stof(deltTime2);
        }
        m_ATX_firetimes.header.version[0] = 0;
        m_ATX_firetimes.header.version[1] = 1;
        this->loadFiretimeSuccess();
        inFile.close();
        return;
      } else {
        throw std::invalid_argument("Open firetime file failed");
      }
    } 
    else if (type == 1) {
      std::ifstream fin(firetimes_path, std::ios::binary);
      if (fin.is_open()) {
        LogDebug("Open firetime file success");
        fin.seekg(0, std::ios::end);
        int length = static_cast<int>(fin.tellg());
        fin.seekg(0, std::ios::beg);
        char *buffer = new char[length];
        fin.read(buffer, length);
        fin.close();
        int ret = LoadFiretimesString(buffer, length);
        delete[] buffer;
        if (ret != 0) {
          throw std::invalid_argument("Parse local firetime file Error");
        } else {
          LogInfo("Parse local firetime file Success!!!");
        }
      } else {
        throw std::invalid_argument("Open firetime file failed");
      }
    } else {
      throw std::invalid_argument("firetime name is error(not .dat / .bin / .csv)");
    }
  } catch (const std::exception &e) {
    this->get_firetime_file_ = false;
    LogError("load firetimes error: %s", e.what());
  }
}

template<typename T_Point>
int Udp4_7Parser<T_Point>::LoadFiretimesString(const char *firetimes_string, int len) {
  try {
    if (len < 32) {
      throw std::invalid_argument("firetimes string length is too short");
    }
    const char *p = firetimes_string;
    ATX::ATXFiretimesHeader header = *(ATX::ATXFiretimesHeader *)p;
    if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
      switch (header.version[1])
      {
        case 1: {
          m_ATX_firetimes.header = header;
          p += sizeof(ATX::ATXFiretimesHeader);
          m_ATX_firetimes.channel_number = *((uint8_t *)p);
          p++;
          m_ATX_firetimes.angle_division = *((uint16_t *)p);
          p += sizeof(uint16_t);
          if (m_ATX_firetimes.channel_number > ATX::ATX_MAX_CHANNEL_NUM || m_ATX_firetimes.angle_division == 0) {
            throw std::invalid_argument("correction error, channel_number or angle_division out of range");
          }
          if (static_cast<size_t>(len) < 9 + sizeof(uint16_t) * m_ATX_firetimes.channel_number * 2 + 32) {
            throw std::invalid_argument("firetimes string length is too short");
          }
          memcpy((void *)&m_ATX_firetimes.raw_even_firetime_correction_, p,
                 sizeof(uint16_t) * m_ATX_firetimes.channel_number);
          p += sizeof(uint16_t) * m_ATX_firetimes.channel_number;       
          memcpy((void *)&m_ATX_firetimes.raw_odd_firetime_correction_, p,
                 sizeof(uint16_t) * m_ATX_firetimes.channel_number);       
          p += sizeof(uint16_t) * m_ATX_firetimes.channel_number;
          memcpy((void*)&m_ATX_firetimes.SHA_value, p, 32);          
        } break;
        default:
          throw std::invalid_argument("min_version is wrong!");
          break;
      }
    } else {
        throw std::invalid_argument("firetimes file delimiter is error");
    }
  } catch (const std::exception &e) {
    this->get_firetime_file_ = false;
    LogError("load firetimes error: %s", e.what());
    return -1;
  }
  if (m_ATX_firetimes.setToFloatUseAngleDivision()) {
    this->loadFiretimeSuccess();
    return 0;
  }
  return -1;
}

template<typename T_Point>
bool Udp4_7Parser<T_Point>::IsNeedFrameSplit(uint16_t frame_id) {
  if ( frame_id != this->last_frameid_  && this->last_frameid_ >= 0) {
      return true;
  }
  return false;
}

template <typename T_Point>
void* Udp4_7Parser<T_Point>::getStruct(const int type) {
  if (type == CORRECTION_STRUCT)
    return (void*)&(m_ATX_corrections);
  else if (type == FIRETIME_STRUCT)
    return (void*)&(m_ATX_firetimes);
  return nullptr;
}

template <typename T_Point>
int Udp4_7Parser<T_Point>::getDisplay(bool **display) {
  *display = m_ATX_corrections.display;
  return ATX::ATX_MAX_CHANNEL_NUM;
}

template<typename T_Point>
int Udp4_7Parser<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index) {
  if ((packet_index + 1) * frame.per_points_num >= frame.maxPackerPerFrame * frame.maxPointPerPacket) {
    LogFatal("total points exceeded memory storage limit");
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
    float distance = static_cast<float>(pointData.distance * frame.distance_unit);
    if (this->get_firetime_file_ && frame.fParam.firetimes_flag) {
      raw_azimuth += ((packetData.data.dATX.frame_id % 2 == 0) ? m_ATX_firetimes.floatCorr.even_firetime_correction_[pointData.channel_index] : 
                        - m_ATX_firetimes.floatCorr.odd_firetime_correction_[pointData.channel_index]) * (abs(static_cast<int16_t>(packetData.spin_speed)) * 1E-9 / 8);
    }
    if (this->get_correction_file_) {
      if (m_ATX_corrections.floatCorr.min_version == 1) {
        raw_azimuth += m_ATX_corrections.floatCorr.azimuth[pointData.channel_index];
      }
      else if (m_ATX_corrections.floatCorr.min_version == 2 || m_ATX_corrections.floatCorr.min_version == 3) {
        raw_azimuth += ((packetData.data.dATX.frame_id % 2 == 0) ? m_ATX_corrections.floatCorr.azimuth_even[pointData.channel_index] : 
                          m_ATX_corrections.floatCorr.azimuth_odd[pointData.channel_index]);
      }
      if(m_ATX_corrections.floatCorr.min_version == 1 || m_ATX_corrections.floatCorr.min_version == 2) {
        raw_elevation = m_ATX_corrections.floatCorr.elevation[pointData.channel_index];
      } 
      else if (m_ATX_corrections.floatCorr.min_version == 3) {
        raw_elevation = m_ATX_corrections.floatCorr.elevation[pointData.channel_index] + m_ATX_corrections.floatCorr.ElevationAdjust(raw_azimuth);
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
      set_confidence(ptinfo, pointData.data.dcfd.confidence);

      point_num++;
    }
  }
  frame.valid_points[packet_index] = point_num;
  GeneralParser<T_Point>::FrameNumAdd();
  return 0;
}

template<typename T_Point>
int Udp4_7Parser<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index) {
  uint32_t packet_index_use = (packet_index >= 0 ? packet_index : frame.packet_num);
  if (udpPacket.buffer[0] != 0xEE || udpPacket.buffer[1] != 0xFF ||
      udpPacket.buffer[2] != 4 || udpPacket.buffer[3] != 7) {
    LogDebug("Invalid point cloud");
    return -1;
  }
  const HS_LIDAR_HEADER_ST_V7 *pHeader =
      reinterpret_cast<const HS_LIDAR_HEADER_ST_V7 *>(
          &(udpPacket.buffer[0]) + sizeof(HS_LIDAR_PRE_HEADER));
  
  const HS_LIDAR_TAIL_ST_V7 *pTail =
      reinterpret_cast<const HS_LIDAR_TAIL_ST_V7 *>(
          (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V7) +
          (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) +
           sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
           sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * pHeader->GetLaserNum()) *
              pHeader->GetBlockNum() +
          sizeof(HS_LIDAR_BODY_CRC_ST_V7));

  if (pHeader->HasSeqNum()) {
    const HS_LIDAR_TAIL_SEQ_NUM_ST_V7 *pTailSeqNum =
        reinterpret_cast<const HS_LIDAR_TAIL_SEQ_NUM_ST_V7 *>(
            (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V7) +
            (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) +
             sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
             sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * pHeader->GetLaserNum()) *
                pHeader->GetBlockNum() +
            sizeof(HS_LIDAR_BODY_CRC_ST_V7) + sizeof(HS_LIDAR_TAIL_ST_V7));
    this->CalPktLoss(pTailSeqNum->GetSeqNum(), frame.fParam);
  }
  this->CalPktTimeLoss(pTail->GetMicroLidarTimeU64(this->last_utc_time), frame.fParam);
  // const HS_LIDAR_E2E_HEADER_ST_V7 *pE2EHeader = 
  //     reinterpret_cast<const HS_LIDAR_E2E_HEADER_ST_V7 *>(
  //           (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V7) +
  //           (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) +
  //            sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
  //            sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * pHeader->GetLaserNum()) *
  //               pHeader->GetBlockNum() +
  //           sizeof(HS_LIDAR_BODY_CRC_ST_V7) + sizeof(HS_LIDAR_TAIL_ST_V7)
  //           + (pHeader->HasSeqNum() ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ST_V7) : 0));
  frame.scan_complete = false;
  frame.block_num = pHeader->GetBlockNum();
  frame.laser_num = pHeader->GetLaserNum();
  frame.per_points_num = pHeader->GetBlockNum() * pHeader->GetLaserNum();
  frame.distance_unit = pHeader->GetDistUnit();
  if (frame.per_points_num > frame.maxPointPerPacket) {
    LogFatal("per_points_num(%u) out of %d", frame.per_points_num, frame.maxPointPerPacket);
    return -1;
  }
  if (frame.laser_num > ATX::ATX_MAX_CHANNEL_NUM) {
    LogFatal("laser_num(%u) out of %d", frame.laser_num, ATX::ATX_MAX_CHANNEL_NUM);
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
  frame.return_mode = pTail->GetReturnMode();
  frame.packetData[packet_index_use].spin_speed = pTail->GetMotorSpeed();

  // 分帧标记位为该字节的最后一个bit
  uint8_t frameID = pTail->GetFrameID() % 2;
  frame.packetData[packet_index_use].data.dATX.frame_id = frameID;
  if (IsNeedFrameSplit(frameID)) {
    frame.scan_complete = true;
  }
  this->last_frameid_ = frameID;
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
    const HS_LIDAR_BODY_AZIMUTH_ST_V7 *pAzimuth =
      reinterpret_cast<const HS_LIDAR_BODY_AZIMUTH_ST_V7 *>(
          (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ST_V7) +
          (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) + sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7) +
          sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V7) * pHeader->GetLaserNum()) * blockid);

    int32_t u32Azimuth = pAzimuth->GetAzimuth();
    const HS_LIDAR_BODY_CHN_NNIT_ST_V7 * pChnUnit = reinterpret_cast<const HS_LIDAR_BODY_CHN_NNIT_ST_V7 *>(
        (const unsigned char *)pAzimuth + sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V7) + sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V7));
    for (int i = 0; i < pHeader->GetLaserNum(); i++) {
      if (m_ATX_corrections.display[i] == false) {
        pChnUnit++;
        continue;
      }
      frame.pointData[index].channel_index = i;
      frame.pointData[index].azimuth = 1.f * u32Azimuth / kFineResolutionFloat;
      frame.pointData[index].reflectivity = pChnUnit->GetReflectivity();  
      frame.pointData[index].distance = pChnUnit->GetDistance();
      frame.pointData[index].data.dcfd.confidence = pChnUnit->GetConfidenceLevel();
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
int Udp4_7Parser<T_Point>::ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info) {
  FaultMessageVersion4_7 *fault_message_ptr =  
      reinterpret_cast< FaultMessageVersion4_7*> (&(udp_packet.buffer[0]));
  fault_message_ptr->ParserFaultMessage(fault_message_info, this->last_utc_time);
  return 0;
}

