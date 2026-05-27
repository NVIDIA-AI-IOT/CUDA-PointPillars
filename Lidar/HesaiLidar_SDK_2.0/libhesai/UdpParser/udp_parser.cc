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
#ifndef UDP_PARSER_CC_
#define UDP_PARSER_CC_
#include "udp_parser.h"
#include "general_parser.h"
using namespace hesai::lidar;

template<typename T_Point>
UdpParser<T_Point>::UdpParser(const UdpPacket& packet) {
  last_host_timestamp_ = 0;
  last_sensor_timestamp_ = 0;
  packet_count_ = 0;
  source_type_ = -1;
  parser_ = nullptr;
  first_packet_ = true;
  printErrorBool = true;
  lidar_type_decoded_ = "unknow";
  this->CreatGeneralParser(packet);
}

template<typename T_Point>
UdpParser<T_Point>::UdpParser(const std::string &lidar_type) {
  last_host_timestamp_ = 0;
  last_sensor_timestamp_ = 0;
  packet_count_ = 0;
  source_type_ = -1;
  parser_ = nullptr;
  first_packet_ = true;
  printErrorBool = true;
  lidar_type_decoded_ = "unknow";
  this->CreatGeneralParser(lidar_type);
}

template<typename T_Point>
UdpParser<T_Point>::UdpParser() {
  last_host_timestamp_ = 0;
  last_sensor_timestamp_ = 0;
  packet_count_ = 0;
  source_type_ = -1;
  parser_ = nullptr;
  first_packet_ = true;
  printErrorBool = true;
  lidar_type_decoded_ = "unknow";
}

template<typename T_Point>
UdpParser<T_Point>::~UdpParser() {
  if (parser_ != nullptr) {
    delete parser_;
    parser_ = nullptr;
  }
}

template<typename T_Point>
void UdpParser<T_Point>::LoadCorrectionFile(const std::string& correction_path) {
  if (parser_ != nullptr) {
    parser_->LoadCorrectionFile(correction_path);
  }
  return;
}

template<typename T_Point>
int UdpParser<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  if (parser_ != nullptr) {
    return parser_->LoadCorrectionString(correction_string, len);
  }
  return -1;
}

template<typename T_Point>
void UdpParser<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  if (parser_ != nullptr) {
    parser_->LoadFiretimesFile(firetimes_path);
  }
  return;
}

template<typename T_Point>
int UdpParser<T_Point>::LoadFiretimesString(const char *firetimes_string, int len) {
  if (parser_ != nullptr) {
    return parser_->LoadFiretimesString(firetimes_string, len);
  }
  return false;
}

template<typename T_Point>
int UdpParser<T_Point>::LoadChannelConfigFile(const std::string channel_config_path) {
  if (parser_ != nullptr) {
    return parser_->LoadChannelConfigFile(channel_config_path);
  }
  return -1;
}

template<typename T_Point>
void UdpParser<T_Point>::CreatGeneralParser(const UdpPacket& packet) {
  if (parser_ != nullptr) {
    delete parser_;
    parser_ = nullptr;
  }
  if (PKT_SIZE_40P == packet.packet_len ||
      PKT_SIZE_40P + 4 == packet.packet_len || PKT_SIZE_AC == packet.packet_len) {
    // Pandar40
    parser_ = new UdpP40Parser<T_Point>();  
    lidar_type_decoded_ = "Pandar40";
    return;
  }
  if (PKT_SIZE_64_B6 == packet.packet_len || PKT_SIZE_64_B6 + 4 == packet.packet_len ||
      PKT_SIZE_20_B7 == packet.packet_len || PKT_SIZE_20_B7 + 4 == packet.packet_len) {
    // Pandar64
    parser_ = new UdpP64Parser<T_Point>();  
    lidar_type_decoded_ = "Pandar64";
    return;
  }
  auto data = packet.buffer;
  auto packet_len = packet.packet_len;
  if (data[0] != 0xEE || data[1] != 0xFF) {
    if (data[SOMEIP_OFFSET] == 0xEE && data[SOMEIP_OFFSET + 1] == 0xFF) {
      data += SOMEIP_OFFSET;
      packet_len -= SOMEIP_OFFSET;
    } else {
      LogWarning("Packet with invaild delimiter");
      return;
    }
  }
  uint8_t major = data[2];
  uint8_t minor = data[3];

  switch (major) {
    case 1:
    {
      switch (minor) {
        case 4: {
          const HS_LIDAR_HEADER_ME_V4 *pHeader =
              reinterpret_cast<const HS_LIDAR_HEADER_ME_V4 *>(
                  data + sizeof(HS_LIDAR_PRE_HEADER));
          if (pHeader->packetSize() != packet_len) {
            LogWarning("V1_4 pkt len: %u, due pkt len: %u", packet_len, pHeader->packetSize());
            break;
          }
          uint8_t statusInfoVersion = data[4];
          if (statusInfoVersion > 127) {
            lidar_type_decoded_ = "OT128";
            parser_ = new Udp1_4Parser<T_Point>(STR_OT128);
          } else {
            lidar_type_decoded_ = "Pandar128";
            parser_ = new Udp1_4Parser<T_Point>(STR_PANDARN);
          }
        } break;
        case 8:
          parser_ = new Udp1_8Parser<T_Point>();
          lidar_type_decoded_ = "JT16";
          break;
        default:
          break;
      }
    } break;
    case 2: 
    {
      switch (minor) {
        default:
          break;
      }
    } break;
    case 3:
    {
      switch (minor) {
        case 1:
          parser_ = new Udp3_1Parser<T_Point>();  // QT32 QT64
          lidar_type_decoded_ = "PandarQT";
          break;
        case 2:
          parser_ = new Udp3_2Parser<T_Point>();  // QT128
          lidar_type_decoded_ = "PandarQT128";
          break;
        default:
          break;
      }

    } break;
    case 4:
    {
      switch (minor) {
        case 1:
        case 3:
          parser_ = new Udp4_3Parser<T_Point>();
          lidar_type_decoded_ = "AT128";
          break;
        case 7:
          parser_ = new Udp4_7Parser<T_Point>();
          lidar_type_decoded_ = "ATX";
          break;
        default:
          break;
      }
    } break;
    case 5:
    {
      switch (minor) {
        default:
          break;
      }
    } break;
    case 6:
    {
      switch (minor) {
        case 1: {
          uint8_t blockNum = data[7];
          if (blockNum == 8) {
            lidar_type_decoded_ = "PandarXT";
            parser_ = new Udp6_1Parser<T_Point>(STR_XTM1);
          } else if (blockNum == 6) {
            lidar_type_decoded_ = "PandarXTM";
            parser_ = new Udp6_1Parser<T_Point>(STR_XTM2);
          } else {
            // It doesn't normally go here.
            lidar_type_decoded_ = "PandarXTM";
            parser_ = new Udp6_1Parser<T_Point>(STR_XTM2);
            LogError("XT version unknown, unable to set distance correction coordinates");
          }
        } break;
        default:
          break;
      }
    } break;
    case 7:
    {
      switch (minor) {
        case 2:
          parser_ = new Udp7_2Parser<T_Point>();
          lidar_type_decoded_ = "PandarFT120";
          break;
        default:
          break;
      }

    } break;
    case 24:
    {
      switch (minor) {
        default:
          break;
      }
    } break;
    default:
      break;
  }
  return;
}
template<typename T_Point>
void UdpParser<T_Point>::CreatGeneralParser(const std::string& lidar_type) {
  if (parser_ != nullptr) {
    delete parser_;
    parser_ = nullptr;
  }

  if (lidar_type == "Pandar64") {
    parser_ = new UdpP64Parser<T_Point>();
    lidar_type_decoded_ = "Pandar64";
  } 
  else if (lidar_type == "Pandar40" || lidar_type == "Pandar40P") {
    parser_ = new UdpP40Parser<T_Point>();
    lidar_type_decoded_ = "Pandar40";
  } 
  else if (lidar_type == "Pandar128E3X" || lidar_type == "Pandar128" || 
             lidar_type == "Pandar40S" || lidar_type == "Pandar40E3X" ||
             lidar_type == "Pandar60S" || lidar_type == "Pandar64E3X" ||
             lidar_type == "Pandar90" || lidar_type == "Pandar90E3X") 
  {
    parser_ = new Udp1_4Parser<T_Point>(STR_PANDARN);
    lidar_type_decoded_ = "Pandar128";
  } 
  else if (lidar_type == "OT128") {
    parser_ = new Udp1_4Parser<T_Point>(STR_OT128);
    lidar_type_decoded_ = "OT128";
  }
  else if (lidar_type == "JT16") {
    parser_ = new Udp1_8Parser<T_Point>();
    lidar_type_decoded_ = "JT16";
  } 
  else if (lidar_type == "PandarQT") {
    parser_ = new Udp3_1Parser<T_Point>();
    lidar_type_decoded_ = "PandarQT";
  } 
  else if (lidar_type == "PandarQT128" || lidar_type == "QT128C2X") {
    parser_ = new Udp3_2Parser<T_Point>();
    lidar_type_decoded_ = "PandarQT128";
  } 
  else if (lidar_type == "AT128" || lidar_type == "AT128E2X" || lidar_type == "AT128E3X") {
    parser_ = new Udp4_3Parser<T_Point>();
    lidar_type_decoded_ = "AT128";
  } 
  else if (lidar_type == "ATX" ) {
    parser_ = new Udp4_7Parser<T_Point>();
    lidar_type_decoded_ = "ATX";
  } 
  else if (lidar_type == "PandarXT" || lidar_type == "PandarXT16" || lidar_type == "PandarXT-16" || 
             lidar_type == "PandarXT32" || lidar_type == "PandarXT-32") 
  {
    parser_ = new Udp6_1Parser<T_Point>(STR_XTM1);
    lidar_type_decoded_ = "PandarXT";
  } 
  else if (lidar_type == "PandarXTM" || lidar_type == "XT32M2X") {
    parser_ = new Udp6_1Parser<T_Point>(STR_XTM2);  
    lidar_type_decoded_ = "PandarXTM";
  } 
  else if (lidar_type == "PandarFT120" || lidar_type == "FT120C1X") {
    parser_ = new Udp7_2Parser<T_Point>();
    lidar_type_decoded_ = "PandarFT120";
  }
}

template<typename T_Point>
int UdpParser<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame, uint32_t packet_index) {
  if (parser_ == nullptr) {
    return -1;
  } else {
    return parser_->ComputeXYZI(frame, packet_index);
  }
}

template<typename T_Point>
int UdpParser<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket, const int packet_index) {
  if (udpPacket.packet_len < 6) {   // sizeof(HS_LIDAR_PRE_HEADER)
    frame.scan_complete = false;
    return -1;
  }
  if (parser_ == nullptr) {
    // Udp raw_udp_packet
    this->CreatGeneralParser(udpPacket);
    return 0;
  } else {
    int res = 0;
    uint32_t packet_index_use = packet_index >= 0 ? packet_index : frame.packet_num;
    parser_->setRemakeDefaultConfig(frame);
    do {
      if (!parser_->isSetCorrectionSucc()) {
        if (printErrorBool) {
          LogInfo("No available angle calibration files, prohibit parsing of point cloud packages");
          printErrorBool = false;
        }
        res = -1;
        break;
      }
      if (packet_index_use >= frame.maxPackerPerFrame) {
        LogFatal("packet_index_use(%u) out of %d", packet_index_use, frame.maxPackerPerFrame);
        res = -1;
        break;
      }
      res = parser_->DecodePacket(frame, udpPacket, packet_index);
    } while(0);
    if (res < 0 && packet_index >= 0 && packet_index < frame.maxPackerPerFrame) {
      frame.valid_points[packet_index] = 0;
    }
    //data from pcap and play rate synchronize with the host time
    if (source_type_ == DATA_FROM_PCAP && frame.fParam.pcap_time_synchronization == true && res == 0) {
      uint64_t sensor_timestamp = frame.frame_end_timestamp * kMicrosecondToSecond;
      if(first_packet_ == true) {
        last_host_timestamp_ = frame.host_timestamp;
        last_sensor_timestamp_ = sensor_timestamp;
        packet_count_ = 1;
        first_packet_ = false;
      } else {
        packet_count_ += 1;
        if (packet_count_ >= kPcapPlaySynchronizationCount) {
          int reset_time = static_cast<int>((sensor_timestamp - last_sensor_timestamp_) - (frame.host_timestamp - last_host_timestamp_));
          last_host_timestamp_ = frame.host_timestamp;
          last_sensor_timestamp_ = sensor_timestamp;
          packet_count_ = 0;
          if (reset_time > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(reset_time));
          }
        }
      }
    }
    return res;
  }
}

template<typename T_Point>
int UdpParser<T_Point>::ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info) {
  if (parser_ == nullptr) {
    return -1;
  } else {
    return parser_->ParserFaultMessage(udp_packet, fault_message_info);
  }
  return 0;
}

template<typename T_Point>
void UdpParser<T_Point>::SetPcapPlay(int source_type) {
  source_type_ = source_type;
}

template<typename T_Point>
void UdpParser<T_Point>::SetFrameAzimuth(float frame_start_azimuth) {
  if (parser_ != nullptr) {
    parser_->SetFrameAzimuth(frame_start_azimuth);
  } else {
    LogWarning("parser is nullptr");
  }
  return;
}

#endif 