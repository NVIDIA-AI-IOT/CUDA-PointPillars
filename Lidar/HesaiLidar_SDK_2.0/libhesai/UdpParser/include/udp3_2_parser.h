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
 * File:       udp3_2_parser.h
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Declare Udp3_2Parser class
*/

#ifndef UDP3_2_PARSER_H_
#define UDP3_2_PARSER_H_
#include "general_parser.h"
#include "udp_protocol_v3_2.h"
namespace hesai
{
namespace lidar
{


// class Udp3_2Parser
// parsers packets and computes points for PandarQT128
template<typename T_Point>
class Udp3_2Parser : public GeneralParser<T_Point> {
 public:
  Udp3_2Parser();
  virtual ~Udp3_2Parser();
  virtual int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index = -1);    
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index);

  // get lidar firetime correction file from local file,and pass to udp parser 
  virtual void LoadFiretimesFile(const std::string& firetimes_path);
  virtual int LoadFiretimesString(const char *firetimes_string, int len);
  // get the pointer to the struct of the parsed correction file or firetimes file
  virtual void* getStruct(const int type);

  // get lidar correction file from local file,and pass to udp parser 
  int LoadChannelConfigFile(const std::string& channel_config_path);
  int LoadChannelConfigString(char *channel_config_content);
  virtual void setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame);
 private:
  int horizontal_resolution_mode = -1;
  int loop_num = -1;
  QT128::FiretimesQt128 qt128_firetimes;
  QT128::PandarQTChannelConfig channel_config_;
};
}  // namespace lidar
}  // namespace hesai

#include "udp3_2_parser.cc"

#endif  // UDP3_2_PARSER_H_
