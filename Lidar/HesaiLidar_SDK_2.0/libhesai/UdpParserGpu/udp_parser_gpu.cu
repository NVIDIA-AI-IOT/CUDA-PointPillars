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
#ifndef UDP_PARSER_GPU_CU_
#define UDP_PARSER_GPU_CU_
#include "udp_parser_gpu.h"
#include "udp_p40_parser_gpu.h"
#include "udp_p64_parser_gpu.h"
#include "udp1_4_parser_gpu.h"
#include "udp1_8_parser_gpu.h"
#include "udp3_1_parser_gpu.h"
#include "udp3_2_parser_gpu.h"
#include "udp4_3_parser_gpu.h"
#include "udp4_7_parser_gpu.h"
#include "udp6_1_parser_gpu.h"
#include "udp7_2_parser_gpu.h"

using namespace hesai::lidar;
template <typename T_Point>
int UdpParserGpu<T_Point>::LoadCorrectionStruct(void* correction) {
  if (correction == nullptr) {
    LogWarning("correction is nullptr when loading gpu correction struct");
    return -1;
  }
  if (m_generalParserGpu != nullptr) {
    m_generalParserGpu->LoadCorrectionStruct(correction);
  }
  else {
    LogWarning("m_generalParserGpu is nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
int UdpParserGpu<T_Point>::LoadFiretimesStruct(void* firetimes) {
  if (firetimes == nullptr) {
    LogWarning("firetimes is nullptr when loading gpu firetimes struct");
    return -1;
  }
  if (m_generalParserGpu != nullptr) {
    m_generalParserGpu->LoadFiretimesStruct(firetimes);
  }
  else {
    LogWarning("m_generalParserGpu is nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
int UdpParserGpu<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame) {
  if (m_generalParserGpu == nullptr) {
    LogFatal("please set the lidar type before parsing");
    return -1;
  }
  return m_generalParserGpu->ComputeXYZI(frame);
}

template <typename T_Point>
void UdpParserGpu<T_Point>::SetLidarType(std::string lidar_type, uint16_t maxPacket, uint16_t maxPoint) {
  if (m_generalParserGpu != nullptr) {
    delete m_generalParserGpu;
    m_generalParserGpu = nullptr;
  }
  if (lidar_type == "Pandar40" || lidar_type == "Pandar40P") {
    m_generalParserGpu = new UdpP40ParserGpu<T_Point>(maxPacket, maxPoint);
  } 
  else if (lidar_type == "Pandar64") {
    m_generalParserGpu = new UdpP64ParserGpu<T_Point>(maxPacket, maxPoint);
  }
  else if (lidar_type == "Pandar128") {
    m_generalParserGpu = new Udp1_4ParserGpu<T_Point>(STR_PANDARN, maxPacket, maxPoint);
  }
  else if (lidar_type == "OT128") {
    m_generalParserGpu = new Udp1_4ParserGpu<T_Point>(STR_OT128, maxPacket, maxPoint);
  }
  else if (lidar_type == "JT16") {
    m_generalParserGpu = new Udp1_8ParserGpu<T_Point>(maxPacket, maxPoint);
  }
  else if (lidar_type == "PandarQT") {
    m_generalParserGpu = new Udp3_1ParserGpu<T_Point>(maxPacket, maxPoint);
  } 
  else if (lidar_type == "PandarQT128" || lidar_type == "QT128C2X") {
    m_generalParserGpu = new Udp3_2ParserGpu<T_Point>(maxPacket, maxPoint);
  } 
  else if (lidar_type == "AT128" || lidar_type == "AT128E2X" || lidar_type == "AT128E3X") {
    m_generalParserGpu = new Udp4_3ParserGpu<T_Point>(maxPacket, maxPoint);;
  }
  else if (lidar_type == "ATX") {
    m_generalParserGpu = new Udp4_7ParserGpu<T_Point>(maxPacket, maxPoint);
  } 
  else if (lidar_type == "PandarXT" || lidar_type == "PandarXT16" || lidar_type == "PandarXT-16" ||
      lidar_type == "PandarXT32" || lidar_type == "PandarXT-32") {
    m_generalParserGpu = new Udp6_1ParserGpu<T_Point>(STR_XTM1, maxPacket, maxPoint);
  }
  else if (lidar_type == "PandarXTM" || lidar_type == "XT32M2X") {
    m_generalParserGpu = new Udp6_1ParserGpu<T_Point>(STR_XTM2, maxPacket, maxPoint);
  }
  else if (lidar_type == "PandarFT120" || lidar_type == "FT120C1X") {
    m_generalParserGpu = new Udp7_2ParserGpu<T_Point>(maxPacket, maxPoint);
  }
  else {
    LogFatal("GPU does not support this type of lidar");
  }
}

#endif //UDP_PARSER_GPU_CU_