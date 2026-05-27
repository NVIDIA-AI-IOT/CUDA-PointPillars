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
 * File:   socket_source.h
 * Author: Felix Zou<zouke@hesaitech.com>
 *
 * Created on Sep 5, 2019, 10:46 AM
 */

#ifndef SOCKETSTREAMER_H
#define SOCKETSTREAMER_H

#include <stdint.h>
#include <string>
#include "source.h"

#ifdef _MSC_VER
#include <winsock2.h>
#include <ws2tcpip.h> 
#pragma comment(lib, "ws2_32.lib")  // Winsock Library
#include <BaseTsd.h>

#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#endif

#pragma pack(push, 1)
struct SomeIPHeader {
    uint16_t someip_service_id;
    uint16_t someip_method_id;
    uint32_t someip_length;
    uint16_t someip_client_id;
    uint16_t someip_session_id;
    uint8_t someip_version;
    uint8_t someip_interface_version;
    uint8_t someip_message_type;
    uint8_t someip_return_code;
};
static_assert(sizeof(SomeIPHeader) == 16);
struct SomeIPMessage {
    SomeIPHeader header;
    char payload[40];
};
#pragma pack(pop)

namespace hesai
{
namespace lidar
{
#define SOMEIP_PORT 30490

class SocketSource : public Source{
 public:
  SocketSource(const uint16_t port = kUdpPort, const std::string& multicastIp = "");
  virtual ~SocketSource();

  virtual bool Open();
  virtual void Close();
  virtual bool IsOpened();
  virtual int Send(uint8_t* u8Buf, uint16_t u16Len, int flags = 0);
  virtual int Receive(UdpPacket& udpPacket, uint16_t u16Len, int flags = 0,
                      int timeout = 1000);
  virtual void SetSocketBufferSize(uint32_t u32BufSize);
  bool sendsomeip_subscribe_ipv4(std::string someip_ip, std::string host_addr, int port, int fault_message_port);
  void SetClientIp(std::string client_ip);
private:
  uint16_t udp_port_;
  int udp_sock_;
  bool is_select_;
  static const int32_t kUDPBufferSize = 26214400;  // udp buffer size
  static const uint16_t kUdpPort = 2368;
  std::string multicast_ip_;
  std::string client_ip_;
  sockaddr_in send_addr_;
};
}  // namespace lidar
}  // namespace hesai
#endif /* SOCKETSTREAMER_H */
