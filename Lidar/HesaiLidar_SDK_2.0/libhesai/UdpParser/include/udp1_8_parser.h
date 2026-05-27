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
 * File:       udp1_8_parser.h
 * Author:     Chang XingShuo <changxingshuo@hesaitech.com>
 * Description: Declare Udp1_8Parser class
*/

#ifndef UDP1_8_PARSER_H_
#define UDP1_8_PARSER_H_

#include "general_parser.h"
#include "udp_protocol_v1_8.h"
namespace hesai
{
namespace lidar
{
// class Udp1_8Parser
// parsers packets and computes points for JT16
template<typename T_Point>
class Udp1_8Parser : public GeneralParser<T_Point> {
 public:
  Udp1_8Parser();
  virtual ~Udp1_8Parser();

  virtual int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index = -1);    
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index);

  virtual void LoadFiretimesFile(const std::string& firetimes_path);
  virtual void LoadCorrectionFile(const std::string& correction_path);
  virtual int LoadCorrectionString(const char *correction_string, int len);
  int LoadCorrectionCsvData(char *correction_string, int len);
  // get the pointer to the struct of the parsed correction file or firetimes file
  virtual void* getStruct(const int type);
  virtual void setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame);
 private:
};
}  // namespace lidar
}  // namespace hesai

#include "udp1_8_parser.cc"

#endif  // UDP1_8_PARSER_H_
