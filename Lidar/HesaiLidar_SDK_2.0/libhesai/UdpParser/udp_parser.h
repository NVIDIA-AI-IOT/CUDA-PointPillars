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
#ifndef UDP_PARSER_H_
#define UDP_PARSER_H_

#include <vector>
#include "general_parser.h"
#include "lidar_types.h"
#include "udp_p40_parser.h"
#include "udp_p64_parser.h"
#include "udp1_4_parser.h"
#include "udp1_8_parser.h"
#include "udp3_1_parser.h"
#include "udp3_2_parser.h"
#include "udp4_3_parser.h"
#include "udp4_7_parser.h"
#include "udp6_1_parser.h"
#include "udp7_2_parser.h"
namespace hesai
{
namespace lidar
{
// class UdpParser
// the UdpParser class is an interface layer.it instantiates a specific udp parser class,
// which is determined by lidar type.
// UdpParser mainly parsers udp or pcap packets and computes xyzi of points 
// you can parser the udp or pcap packets using the DecodePacket function
// you can compute xyzi of points using the ComputeXYZI function, which uses cpu to compute
template <typename T_Point>
class UdpParser {
 public:
  explicit UdpParser(const std::string &lidar_type);
  explicit UdpParser(const UdpPacket &packet);
  UdpParser();
  virtual ~UdpParser();
  // set parser
  void CreatGeneralParser(const std::string& lidar_type);
  void CreatGeneralParser(const UdpPacket& packet);
  void SetGeneralParser(GeneralParser<T_Point> *parser) { parser_ = parser; }
  // get parser
  GeneralParser<T_Point> *GetGeneralParser() { return parser_; }

  // load correction file, which is necessary for DecodePacket.
  void LoadCorrectionFile(const std::string& correction_path);
  int LoadCorrectionString(const char *correction_string, int len);
  // load firetimes file
  void LoadFiretimesFile(const std::string& firetimes_path);
  int LoadFiretimesString(const char *firetimes_string, int len);
  // load channel config file
  int LoadChannelConfigFile(const std::string channel_config_path);
  // get the pointer to the struct of the parsed correction file or firetimes file
  void *getStruct(const int type) { if (parser_ != nullptr) return parser_->getStruct(type); else return nullptr; }
  int getDisplay(bool **display) { if (parser_ != nullptr) return parser_->getDisplay(display); else return 0; }
  // get/set correction/firetimes file loading flag
  bool isSetCorrectionSucc() { if (parser_ != nullptr) return parser_->isSetCorrectionSucc(); else return false; }
  bool isSetFiretimeSucc() { if (parser_ != nullptr) return parser_->isSetFiretimeSucc(); else return false; }

  // covert a origin udp packet to decoded data, and pass the decoded data to a frame struct to reduce memory copy
  int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index = -1);  
  // xyzi of points after computed is puted in frame  
  int ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index);
  // parse the detailed content of the fault message message
  int ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info);
  // get/clear the number of parsed packets
  uint32_t GetComputePacketNum() { if (parser_ != nullptr) return parser_->GetComputePacketNum(); else return 0; }
  void SetComputePacketNumToZero() { if (parser_ != nullptr) parser_->SetComputePacketNumToZero(); }
  //
  void setFrameRightMemorySpace(LidarDecodedFrame<T_Point> &frame) { if (parser_ != nullptr) parser_->setFrameRightMemorySpace(frame); }

  // set frame azimuth
  void SetFrameAzimuth(float frame_start_azimuth);
  // set the parsing type
  void SetPcapPlay(int source_type);
  // fet the basic lidar model
  std::string GetLidarType() {return lidar_type_decoded_;}


  
 private:
  GeneralParser<T_Point> *parser_;
  std::string lidar_type_decoded_;
  bool first_packet_;
  uint64_t last_host_timestamp_;
  uint64_t last_sensor_timestamp_;
  uint16_t packet_count_;
  int source_type_;
  bool printErrorBool;
};
}  // namespace lidar
}  // namespace hesai

#include "udp_parser.cc"

#endif  // UDP_PARSER_H_
