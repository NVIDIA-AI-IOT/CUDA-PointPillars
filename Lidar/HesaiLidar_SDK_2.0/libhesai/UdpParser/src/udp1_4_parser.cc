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
 * File:       udp1_4_parser.cc
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Implemente Udp1_4Parser class
*/

#include "udp1_4_parser.h"

using namespace hesai::lidar;
template<typename T_Point>
Udp1_4Parser<T_Point>::Udp1_4Parser(std::string lidar_type) {
  this->lidar_type_ = lidar_type;
  if (lidar_type == STR_PANDARN) {
    this->optical_center.setNoFlag(LidarOpticalCenter{-0.012, 0.04356, 0});
  } 
  else if (lidar_type == STR_OT128) {
    this->optical_center.setNoFlag(LidarOpticalCenter{-0.01, 0.045, 0});
  }
  if (lidar_type == STR_PANDARN || lidar_type == STR_OT128) {
    this->default_remake_config.min_azi = 0.f;
    this->default_remake_config.max_azi = 360.f;
    this->default_remake_config.ring_azi_resolution = 0.1f;
    this->default_remake_config.max_azi_scan = 3600;   // (max_azi - min_azi) / ring_azi_resolution
    this->default_remake_config.min_elev = -25.f;
    this->default_remake_config.max_elev = 15.f;
    this->default_remake_config.ring_elev_resolution = 0.125f;
    this->default_remake_config.max_elev_scan = 320;   // (max_elev - min_elev) / ring_elev_resolution
  }
  LogInfo("init 1_4 parser (%s)", lidar_type.c_str());
}

template<typename T_Point>
Udp1_4Parser<T_Point>::~Udp1_4Parser() { LogInfo("release 1_4 parser"); }

template <typename T_Point>
void Udp1_4Parser<T_Point>::setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame) {
  frame.resetMalloc(4000, 400);
}

template <typename T_Point>
int Udp1_4Parser<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  try {
    std::string correction_content_str(correction_string, len);
    std::istringstream ifs(correction_content_str);
    std::string line;
    std::string hash = "";
    int is_hav_eeff = 0;
    // skip first line "Laser id,Elevation,Azimuth" or "eeff"
    std::getline(ifs, line);  
    float elevation_list[DEFAULT_MAX_LASER_NUM], azimuth_list[DEFAULT_MAX_LASER_NUM];
    std::vector<std::string> vfirstLine;
    split_string(vfirstLine, line, ',');
    if (vfirstLine[0] == "EEFF" || vfirstLine[0] == "eeff") {
      // skip second line
      std::getline(ifs, line);  
      is_hav_eeff = 1;
    }

    int lineCount = 0;
    while (std::getline(ifs, line)) {
      std::vector<std::string> vLineSplit;
      split_string(vLineSplit, line, ',');
      // skip error line or hash value line

      if (this->lidar_type_ != STR_PANDARN && vLineSplit.size() > 0 && vLineSplit[0].size() == 64) {
        hash = vLineSplit[0];
        SHA256_USE sha256;
        // 按行读取内容  
        int readCount = lineCount + 1 + is_hav_eeff;
        int i = 0;
        for (i = 0; i < len; i++) {
          if (correction_string[i] == '\n') {
            readCount--;
          }
          if (readCount <= 0) break;
        }
        sha256.update(correction_string, i + 1);
        uint8_t u8Hash[32];
        sha256.hexdigest(u8Hash);
        std::ostringstream oss;  
        for (size_t i = 0; i < sizeof(u8Hash); ++i) {  
            oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(u8Hash[i]);  
        }  
        std::string hashString = oss.str(); 
        if (hashString != hash) {
          LogWarning("correction file is invalid, hash is error, lineCount: %d", lineCount);
        }
        continue;
      }
      if (vLineSplit.size() != 3) {  
        continue;
      } else {
        lineCount++;
        if (lineCount > pandarN::kMaxChannelPandarN) {
          throw std::invalid_argument("invalid correction input file!(count)");
        }
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

    memset(this->correction.elevation, 0, sizeof(this->correction.elevation));
    memset(this->correction.azimuth, 0, sizeof(this->correction.azimuth));
    for (int i = 0; i < lineCount; ++i) {
      this->correction.elevation[i] = elevation_list[i];
      this->correction.azimuth[i] = azimuth_list[i];
    }
    this->correction.hash = hash;
    this->loadCorrectionSuccess();
  } catch (const std::exception &e) {
    LogFatal("load correction error: %s", e.what());
    this->get_correction_file_ = false;
    return -1;
  }
  return 0;
}

template<typename T_Point>
void Udp1_4Parser<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  try {
    std::ifstream inFile(firetimes_path, std::ios::in);
    if (inFile.is_open()) {
      int uselessLine = 0;
      int count = 0;
      bool is_OT = false;
      std::string lineStr;
      {
        while (std::getline(inFile, lineStr)) {
          std::stringstream ss(lineStr);
          std::string str;
          std::vector<std::string> strList;
          strList.reserve(20);
          while(std::getline(ss, str, ',')){
            strList.push_back(str);
          }
          if (uselessLine == 0 && strList[0] == "Operational State") {
            is_OT = true;
          } else if(uselessLine == 0) {
            std::string dist;
            for (auto e : strList[0]) {
              if (e == '.' || (e >= '0' && e <= '9')) {
                dist.push_back(e);
              }
            }
            if (dist.size() != 0) {
              firetimes.section_distance = std::stod(dist);
            } 
          }
          if (is_OT == false) {
            if(uselessLine < 3) {
              uselessLine++;
              continue;
            }
            if (lineStr[lineStr.size() - 1] == '\n') {
              lineStr = lineStr.substr(lineStr.size() - 1);
            }
            if (strList.size() < 17) {
              throw std::invalid_argument("invalid firetime input file!(list)");
            }
            int idx = std::stoi(strList[0]) - 1;
            if (idx >= pandarN::kMaxChannelPandarN || idx < 0) {
              throw std::invalid_argument("laser id is wrong in firetimes file.");
            }
            for (int i = 1; i <= 15; i += 2) {
              int a = std::stoi(strList[i]);
              int b = std::stoi(strList[i + 1]);
              firetimes.firetime_section_values[idx].section_values[i / 2].firetime[0] = a;
              firetimes.firetime_section_values[idx].section_values[i / 2].firetime[1] = b;
            }
          } else {
            if(uselessLine < 2) {
              uselessLine++;
              continue;
            }
            if (lineStr[lineStr.size() - 1] == '\n') {
              lineStr = lineStr.substr(lineStr.size() - 1);
            }
            if (strList.size() < 7) {
              throw std::invalid_argument("invalid firetime input file!(list)");
            }
            int idx = std::stoi(strList[0]) - 1;
            if (idx >= pandarN::kMaxChannelPandarN || idx < 0) {
              throw std::invalid_argument("laser id is wrong in firetimes file.");
            }
            for (int i = 1; i <= 6; i++) {
              int a = 0;
              if (strList[i] != "-") {
                a = int(std::stof(strList[i]) * 1000);
              }
              firetimes.firetime_section_values[idx].section_values[i - 1].firetime[0] = a;
              firetimes.firetime_section_values[idx].section_values[i - 1].firetime[1] = a;
            }
          }
          count++;
          if (count > pandarN::kMaxChannelPandarN) {
            throw std::invalid_argument("invalid firetime input file!(count)");
          }
        }
      }
    } else {
      throw std::invalid_argument("Open firetime file failed");
    }
    this->loadFiretimeSuccess();
    LogInfo("Open firetime file success!");
  } catch (const std::exception &e) {
    LogError("load firetimes error: %s", e.what());
    this->get_firetime_file_ = false;
  }
  return;
}

template <typename T_Point>
void* Udp1_4Parser<T_Point>::getStruct(const int type) { 
  if (type == CORRECTION_STRUCT)
    return (void*)&(this->correction);
  else if (type == FIRETIME_STRUCT)
    return (void*)&firetimes;
  return nullptr;
}

template<typename T_Point>
int Udp1_4Parser<T_Point>::GetFiretimes(int laserId, uint8_t optMode, uint8_t angleState, float dist) {
  int idx = 0;
  switch (optMode)
  {
    case 0:
      idx = angleState;
      break;
    case 2:
      idx = 4 + angleState;
      break;
    case 3:
      idx = 6 + angleState;
      break;

    default:
      return 0.0;
      break;
  }
  int k = ((dist >= firetimes.section_distance) ? 0 : 1);
  if (laserId >= pandarN::kMaxChannelPandarN || idx >= 8) {
    LogWarning("firetimes laserId or idx is error");
    return 0;
  }
  return firetimes.firetime_section_values[laserId].section_values[idx].firetime[k];
}

template<typename T_Point>
double Udp1_4Parser<T_Point>::GetFiretimesCorrection(int laserId, double speed, uint8_t optMode, uint8_t angleState, float dist) {
  return GetFiretimes(laserId, optMode, angleState, dist) * speed * 6E-9;
}

template<typename T_Point>
int Udp1_4Parser<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index) {
  if ((packet_index + 1) * frame.per_points_num >= frame.maxPackerPerFrame * frame.maxPointPerPacket) {
    LogFatal("total points exceeded memory storage limit");
    GeneralParser<T_Point>::FrameNumAdd();
    return -1;
  }
  int point_index = packet_index * frame.per_points_num;
  int point_num = 0;
  for (int i = point_index; i < point_index + static_cast<int>(frame.valid_points[packet_index]); i++) {
    auto &pointData = frame.pointData[i];
    auto &packetData = frame.packetData[packet_index];
    int elevation = 0;
    int azimuth = floatToInt(pointData.azimuth * kAllFineResolutionInt);
    float distance = static_cast<float>(pointData.distance * frame.distance_unit);
    if (this->get_firetime_file_ && frame.fParam.firetimes_flag) {
      azimuth += (frame.fParam.rotation_flag > 0 ? 1 : -1) * 
        doubleToInt(GetFiretimesCorrection(pointData.channel_index, packetData.spin_speed * (this->lidar_type_ != STR_OTHER ? 1.0 : 0.1), 
        packetData.data.d1_4.optMode, pointData.data.d1_4.angleState, distance) * kAllFineResolutionInt);
    }
    if (this->get_correction_file_) {
      int azimuth_coll = doubleToInt(this->correction.azimuth[pointData.channel_index] * kAllFineResolutionFloat);
      int elevation_corr = doubleToInt(this->correction.elevation[pointData.channel_index] * kAllFineResolutionFloat);
      if (frame.fParam.distance_correction_flag) {
        GeneralParser<T_Point>::GetDistanceCorrection(this->optical_center, azimuth_coll, elevation_corr, distance, this->lidar_type_ != STR_OTHER ? GeometricCenter : OpticalCenter);
      }
      elevation = elevation_corr;
      azimuth += azimuth_coll;
    } 
    this->CircleRevise(azimuth);
    this->CircleRevise(elevation);
    if (frame.fParam.config.fov_start != -1 && frame.fParam.config.fov_end != -1) {
      int fov_transfer = azimuth / kAllFineResolutionInt;
      if (fov_transfer < frame.fParam.config.fov_start || fov_transfer > frame.fParam.config.fov_end) { //不在fov范围continue
        continue;
      }
    }
    uint64_t timestamp = packetData.t.sensor_timestamp * kMicrosecondToNanosecondInt + pointData.data.d1_4.ns_offset;
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
      set_timeSecond(ptinfo, timestamp / kNanosecondToSecondInt);
      set_timeNanosecond(ptinfo, timestamp % kNanosecondToSecondInt);
      set_confidence(ptinfo, pointData.data.d1_4.confidence);
      set_weightFactor(ptinfo, pointData.data.d1_4.weightFactor);
      set_envLight(ptinfo, pointData.data.d1_4.envLight);

      point_num++;
    }
  }
  frame.valid_points[packet_index] = point_num;
  GeneralParser<T_Point>::FrameNumAdd();
  return 0;
}

template<typename T_Point>
int Udp1_4Parser<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index)
{
  uint32_t packet_index_use = (packet_index >= 0 ? packet_index : frame.packet_num);
  if (udpPacket.buffer[0] != 0xEE || udpPacket.buffer[1] != 0xFF ||
      udpPacket.buffer[2] != 1 || udpPacket.buffer[3] != 4) {
    LogDebug("Invalid point cloud");
    return -1;
  }
  const HS_LIDAR_HEADER_ME_V4 *pHeader =
      reinterpret_cast<const HS_LIDAR_HEADER_ME_V4 *>(
          udpPacket.buffer + sizeof(HS_LIDAR_PRE_HEADER));
  int unitSize = pHeader->unitSize();
  if(hasFunctionSafety(pHeader->m_u8Status)) {
    const auto *function_savety_ptr = reinterpret_cast<const HS_LIDAR_FUNC_SAFETY_ME_V4 *>(
      (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ME_V4) +
      (sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + unitSize * pHeader->GetLaserNum()) *
      pHeader->GetBlockNum() + sizeof(HS_LIDAR_BODY_CRC_ME_V4)
    );
    frame.lidar_state = function_savety_ptr->GetLidarState();
    if (frame.fParam.update_function_safety_flag) {
      frame.funcSafety[packet_index_use].is_valid = true;
      frame.funcSafety[packet_index_use].fs_version = function_savety_ptr->m_u8Version;
      frame.funcSafety[packet_index_use].status = function_savety_ptr->m_u8Status;
      frame.funcSafety[packet_index_use].fault_info = function_savety_ptr->m_u8FaultInfo;
      frame.funcSafety[packet_index_use].fault_code = function_savety_ptr->GetFaultCode();
    }
  } else {
    frame.lidar_state = -1;
  }

  const auto *pTail = reinterpret_cast<const HS_LIDAR_TAIL_ME_V4 *>(
      (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ME_V4) +
      (sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + unitSize * pHeader->GetLaserNum()) *
      pHeader->GetBlockNum() + sizeof(HS_LIDAR_BODY_CRC_ME_V4) +
      (hasFunctionSafety(pHeader->m_u8Status) ? sizeof(HS_LIDAR_FUNC_SAFETY_ME_V4) : 0));
  if (hasSeqNum(pHeader->m_u8Status)) {
    const HS_LIDAR_TAIL_SEQ_NUM_ME_V4 *pTailSeqNum =
        reinterpret_cast<const HS_LIDAR_TAIL_SEQ_NUM_ME_V4 *>(
            (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ME_V4) +
            (sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + unitSize * pHeader->GetLaserNum()) *
            pHeader->GetBlockNum() + sizeof(HS_LIDAR_BODY_CRC_ME_V4) +
            (hasFunctionSafety(pHeader->m_u8Status) ? sizeof(HS_LIDAR_FUNC_SAFETY_ME_V4) : 0) +
            sizeof(HS_LIDAR_TAIL_ME_V4));
    
    this->CalPktLoss(pTailSeqNum->GetSeqNum(), frame.fParam);
  }
  this->CalPktTimeLoss(pTail->GetMicroLidarTimeU64(this->last_utc_time), frame.fParam);
  const HS_LIDAR_TAIL_IMU_ME_V4 *pTailImu = 
    reinterpret_cast<const HS_LIDAR_TAIL_IMU_ME_V4 *>(
          (const unsigned char *)pTail + sizeof(HS_LIDAR_TAIL_ME_V4) + 
          (hasSeqNum(pHeader->m_u8Status) ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ME_V4) : 0));

  frame.packetData[packet_index_use].spin_speed = pTail->GetMotorSpeed();
  frame.return_mode = pTail->GetReturnMode();

  frame.scan_complete = false;
  frame.block_num = pHeader->GetBlockNum();
  frame.laser_num = pHeader->GetLaserNum();
  frame.per_points_num = pHeader->GetBlockNum() * pHeader->GetLaserNum();
  frame.distance_unit = pHeader->GetDistUnit();
  frame.work_mode = pTail->getOperationMode();
  if (frame.per_points_num > frame.maxPointPerPacket) {
    LogFatal("per_points_num(%u) out of %d", frame.per_points_num, frame.maxPointPerPacket);
    return -1;
  }
  if (frame.laser_num > DEFAULT_MAX_LASER_NUM) {
    LogFatal("laser_num(%u) out of %d", frame.laser_num, DEFAULT_MAX_LASER_NUM);
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

  // get imu
  frame.imu_config.timestamp = frame.frame_end_timestamp;
  frame.imu_config.imu_accel_x = pTailImu->GetIMUXAccel();
  frame.imu_config.imu_accel_y = pTailImu->GetIMUYAccel();
  frame.imu_config.imu_accel_z = pTailImu->GetIMUZAccel();
  frame.imu_config.imu_ang_vel_x = pTailImu->GetIMUXAngVel();
  frame.imu_config.imu_ang_vel_y = pTailImu->GetIMUYAngVel();
  frame.imu_config.imu_ang_vel_z = pTailImu->GetIMUZAngVel();
  frame.imu_config.flag = true;

  uint16_t u16Azimuth = 0; 
  const HS_LIDAR_BODY_AZIMUTH_ME_V4 *pAzimuth =
      reinterpret_cast<const HS_LIDAR_BODY_AZIMUTH_ME_V4 *>(
          (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ME_V4));
  u16Azimuth = pAzimuth->GetAzimuth();
  if (this->IsNeedFrameSplit(u16Azimuth, frame.fParam)) {
    frame.scan_complete = true;
  }
  if (u16Azimuth != this->last_azimuth_) {
    this->last_last_azimuth_ = this->last_azimuth_;
    this->last_azimuth_ = u16Azimuth;
  }
  if (frame.scan_complete)
    return 0;

  int index = packet_index_use * frame.per_points_num;
  int point_num = 0;
  int32_t block_ns_offset = 0;
  frame.packetData[packet_index_use].data.d1_4.optMode = pTail->getOperationMode();
  for (int blockid = 0; blockid < pHeader->GetBlockNum(); blockid++) {
    int current_block_echo_count = ((pHeader->GetEchoCount() > 0 && pHeader->GetEchoNum() > 0) ?
            ((pHeader->GetEchoCount() - 1 + blockid) % pHeader->GetEchoNum() + 1) : 0);
    if (frame.fParam.echo_mode_filter != 0 && current_block_echo_count != 0 && frame.fParam.echo_mode_filter != current_block_echo_count) {
      continue;
    }
    if (this->lidar_type_ == STR_OT128) {
      block_ns_offset = PandarN::OT128_BLOCK_NS_OFFSET2 * int((pHeader->GetBlockNum() - blockid - 1) / (frame.return_mode < RETURN_MODE_MULTI ? 1 : 2)) * (pTail->getOperationMode() == 0 ? 1 : 2);
    }
    else {
      if (pHeader->GetBlockNum() == 40)
        block_ns_offset = PandarN::PandarN_BLOCK_NS_OFFSET1 + PandarN::Pandar40S_BLOCK_NS_OFFSET2 * int((pHeader->GetBlockNum() - blockid - 1) / (frame.return_mode < RETURN_MODE_MULTI ? 1 : 2));
      else 
        block_ns_offset = PandarN::PandarN_BLOCK_NS_OFFSET1 + PandarN::PandarN_BLOCK_NS_OFFSET2 * int((pHeader->GetBlockNum() - blockid - 1) / (frame.return_mode < RETURN_MODE_MULTI ? 1 : 2)) * (pTail->getOperationMode() == 0 ? 1 : 2);
    }
    uint8_t angleState = pTail->getAngleState(blockid);
    pAzimuth =
        reinterpret_cast<const HS_LIDAR_BODY_AZIMUTH_ME_V4 *>(
            (const unsigned char *)pHeader + sizeof(HS_LIDAR_HEADER_ME_V4) + 
            (sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + unitSize * pHeader->GetLaserNum()) * blockid);
    u16Azimuth = pAzimuth->GetAzimuth();
    for (int i = 0; i < pHeader->GetLaserNum(); ++i) {
      if (this->correction.display[i] == false) {
        continue;
      }
      const HS_LIDAR_BODY_CHN_UNIT_ME_V4* pChnUnit = 
          reinterpret_cast<const HS_LIDAR_BODY_CHN_UNIT_ME_V4 *>(
              (const unsigned char *)pAzimuth + sizeof(HS_LIDAR_BODY_AZIMUTH_ME_V4) + 
              unitSize * i);
      frame.pointData[index].channel_index = i;
      frame.pointData[index].azimuth = static_cast<float>(u16Azimuth) / kResolutionFloat;
      frame.pointData[index].distance = pChnUnit->GetDistance();
      frame.pointData[index].reflectivity = pChnUnit->GetReflectivity();   
      int k = 0;
      if (hasConfidence(pHeader->m_u8Status)) frame.pointData[index].data.d1_4.confidence = pChnUnit->reserved[k++];
      if (hasWeightFactor(pHeader->m_u8Status)) frame.pointData[index].data.d1_4.weightFactor = pChnUnit->reserved[k++];
      if (hasEnvLight(pHeader->m_u8Status)) frame.pointData[index].data.d1_4.envLight = pChnUnit->reserved[k];
      frame.pointData[index].data.d1_4.angleState = angleState;
      if (this->get_firetime_file_) frame.pointData[index].data.d1_4.ns_offset = block_ns_offset + GetFiretimes(i, pTail->getOperationMode(), angleState, pChnUnit->GetDistance() * frame.distance_unit);
      index++;   
      point_num++;
    }
  }
  frame.valid_points[packet_index_use] = point_num;
  frame.packet_num++;
  return 0;
}


