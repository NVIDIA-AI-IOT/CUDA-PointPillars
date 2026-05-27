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
#ifndef UDP_PARSER_GPU_H_
#define UDP_PARSER_GPU_H_
#include <stdint.h>
#include <iostream>
#include "lidar_types.h"
#include "general_parser_gpu.h"

namespace hesai
{
namespace lidar
{
// class UdpParserGpu
// the UdpParserGpu class is an interface layer.it instantiates a specific udp parser class,
// which is determined by lidar type.
// UdpParserGpu mainly computes xyzi of points with gpu
// you can compute xyzi of points using the ComputeXYZI fuction, which uses gpu to compute
template <typename T_Point>
class UdpParserGpu {
 public:
  UdpParserGpu() {
    m_generalParserGpu = nullptr;
  };
  ~UdpParserGpu(){
    if (m_generalParserGpu != nullptr) {
      delete m_generalParserGpu;
      m_generalParserGpu = nullptr;
    }
  };
  int LoadCorrectionStruct(void*);
  int LoadFiretimesStruct(void*);
  int ComputeXYZI(LidarDecodedFrame<T_Point> &frame);
  void SetLidarType(std::string lidar_type, uint16_t maxPacket, uint16_t maxPoint);

  void setCorrectionLoadFlag(bool* flag) { if (m_generalParserGpu != nullptr) m_generalParserGpu->setCorrectionLoadFlag(flag); }
  void setFiretimeLoadFlag(bool* flag) { if (m_generalParserGpu != nullptr) m_generalParserGpu->setFiretimeLoadFlag(flag); }
  void setCorrectionLoadSequenceNum(uint32_t* num) { if (m_generalParserGpu != nullptr) m_generalParserGpu->setCorrectionLoadSequenceNum(num); }
  void setFiretimeLoadSequenceNum(uint32_t* num) { if (m_generalParserGpu != nullptr) m_generalParserGpu->setFiretimeLoadSequenceNum(num); }
  private:
  GeneralParserGpu<T_Point> * m_generalParserGpu;

};
}
}
#include "udp_parser_gpu.cu"
#endif  // UDP_PARSER_GPU_H_

