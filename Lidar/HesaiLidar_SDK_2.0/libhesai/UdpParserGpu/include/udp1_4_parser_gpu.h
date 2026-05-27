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
#ifndef Udp1_4_PARSER_GPU_H_
#define Udp1_4_PARSER_GPU_H_
#include "general_parser_gpu.h"
#include "udp_protocol_v1_4.h"

namespace hesai
{
namespace lidar
{
// class Udp1_4ParserGpu
// computes points for Pandar128
// you can compute xyzi of points using the ComputeXYZI fuction, which uses gpu to compute
template <typename T_Point>
class Udp1_4ParserGpu: public GeneralParserGpu<T_Point>{
 private:
  pandarN::FiretimesPandarN* firetimes_cu_;
 public:
  Udp1_4ParserGpu(std::string lidar_type, uint16_t maxPacket, uint16_t maxPoint);
  ~Udp1_4ParserGpu();

  // compute xyzi of points from decoded packet， use gpu device
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame);
  virtual void LoadFiretimesStruct(void *);
  virtual void updateFiretimeFile();
  const pandarN::FiretimesPandarN* pandar_firetimes_ptr_;
  std::string lidar_type_;
};
}
}
#include "udp1_4_parser_gpu.cu"
#endif  // Udp1_4_PARSER_GPU_H_
