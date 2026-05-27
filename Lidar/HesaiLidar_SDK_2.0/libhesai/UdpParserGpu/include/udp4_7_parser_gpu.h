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
#ifndef UDP4_7_PARSER_GPU_H_
#define UDP4_7_PARSER_GPU_H_
#include "general_parser_gpu.h"
#include "udp_protocol_v4_7.h"
namespace hesai
{
namespace lidar
{
// class Udp4_7ParserGpu
// computes points for PandarAT128
template <typename T_Point>
class Udp4_7ParserGpu: public GeneralParserGpu<T_Point> {
 private:
  ATX::ATXCorrectionFloat* ATX_correction_cu_;
  const ATX::ATXCorrections* ATX_correction_ptr;
  ATX::ATXFiretimesFloat* ATX_firetimes_cu;
 public:
  Udp4_7ParserGpu(uint16_t maxPacket, uint16_t maxPoint);
  ~Udp4_7ParserGpu();

  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame);
  virtual void LoadCorrectionStruct(void *);
  virtual void LoadFiretimesStruct(void *);
  virtual void updateCorrectionFile();
  virtual void updateFiretimeFile();
  const ATX::ATXFiretimes* ATX_firetimes_ptr_;
};
}
}
#include "udp4_7_parser_gpu.cu"
#endif  // UDP4_7_PARSER_GPU_H_
