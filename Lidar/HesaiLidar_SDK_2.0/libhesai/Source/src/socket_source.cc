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
 * File:   socket_source.cc
 * Author: Felix Zou<zouke@hesaitech.com>
 */

#include "socket_source.h"
#include "logger.h" 
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <cstring>
using namespace hesai::lidar;
SocketSource::SocketSource(const uint16_t port, const std::string& multicastIp) {
  client_ip_.clear();
  udp_port_ = port;
  udp_sock_ = -1;
  multicast_ip_ = multicastIp;
  is_select_ = false;
}

SocketSource::~SocketSource() { Close(); }

void SocketSource::Close() {
  LogInfo("SocketSource::Close()");

  client_ip_.clear();
  udp_port_ = 0;

  if (udp_sock_ != -1) {
#ifdef _MSC_VER
    closesocket(udp_sock_);
    WSACleanup();
#else
    close(udp_sock_);
#endif
    udp_sock_ = -1;
  }
}


bool SocketSource::Open() {
#ifdef _MSC_VER
    WSADATA wsaData;
    WORD version = MAKEWORD(2, 2);
    int res = WSAStartup(version, &wsaData);  // win sock start up
    if (res) {
        LogError("Initilize winsock error !");
        return false;
    }
#endif
  int retVal = 0;
  struct sockaddr_in serverAddr;

  udp_sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  if (udp_sock_ == -1) return false;

  memset(&serverAddr, 0, sizeof(serverAddr));

  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serverAddr.sin_port = htons(udp_port_);

  int reuseaddr = 1;
  retVal = setsockopt(udp_sock_, SOL_SOCKET, SO_REUSEADDR, (char*)&reuseaddr,
                      sizeof(reuseaddr));
  int nRecvBuf = 400000000;
  setsockopt(udp_sock_, SOL_SOCKET, SO_RCVBUF, (const char *)&nRecvBuf,
             sizeof(int));
  int curRcvBufSize = -1;
  socklen_t optlen = sizeof(curRcvBufSize);
  if (getsockopt(udp_sock_, SOL_SOCKET, SO_RCVBUF, (char*)&curRcvBufSize, &optlen) <
      0) {
    LogWarning("getsockopt error=%d(%s)!!!", errno, strerror(errno));
  }
  LogInfo("OS current udp socket recv buff size is: %d", curRcvBufSize); 

  if (retVal == 0) {
#ifdef _MSC_VER
    int timeout_ms = 20;
    retVal = setsockopt(udp_sock_, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout_ms,
        sizeof(timeout_ms));
    // SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#else
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 20000;

    retVal = setsockopt(udp_sock_, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout,
                        sizeof(struct timeval));
#endif
    if (retVal == 0) {
      if (bind(udp_sock_, (sockaddr*)&serverAddr, sizeof(sockaddr)) == -1) {
        if (EINPROGRESS != errno && EWOULDBLOCK != errno) {
#ifdef _MSC_VER
          closesocket(udp_sock_);
#else
          close(udp_sock_);
#endif
          udp_sock_ = -1;
          LogError("SocketSource::Open(), bind failed, errno: %d", errno);
          return false;
        }
      } else {
        LogInfo("SocketSource::Open succeed, sock:%d", udp_sock_);
      }
    } else {
      LogError("setsockopt SO_RCVTIMEO failed, errno:%d", errno);
    }
  } else {
    LogError("setsockopt SO_REUSEADDR failed, errno:%d", errno);
  }
#ifdef _MSC_VER
  unsigned long nonBlockingMode = 1;
  if (ioctlsocket(udp_sock_, FIONBIO, &nonBlockingMode) != 0) {
      LogError("non-block");
  }
#else
  if (fcntl(udp_sock_, F_SETFL, O_NONBLOCK | FASYNC) < 0) {
    LogError("non-block");
  }
#endif

  int32_t rcvBufSize = kUDPBufferSize;
  setsockopt(udp_sock_, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBufSize,
      sizeof(rcvBufSize));

  if(multicast_ip_ != ""){
    struct ip_mreq mreq;                    
    mreq.imr_multiaddr.s_addr=inet_addr(multicast_ip_.c_str());
    mreq.imr_interface.s_addr = htonl(INADDR_ANY); 
    int ret = setsockopt(udp_sock_, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char *)&mreq, sizeof(mreq));
    if (ret < 0) {
      LogError("Multicast IP error, set correct multicast ip address or keep it empty");
    } 
    else {
      LogInfo("Recive data from multicast ip address %s", multicast_ip_.c_str());
    }
  }
  return retVal == 0;
}

bool SocketSource::IsOpened() {
  bool ret = true;

  if (udp_sock_ == -1) {
    ret = false;
    LogWarning("SocketSource::IsOpened(), port %d, sock %d is not open", udp_port_,
           udp_sock_);
  }

  return ret;
}

int SocketSource::Send(uint8_t* u8Buf, uint16_t u16Len, int flags) {
  int len = -1;
  if (!IsOpened()) {
    LogError("SocketSource::Send, port %d, sock %d is not open", udp_port_,
           udp_sock_);
    return len;
  }
  if (client_ip_.empty()) {
    LogError("SocketSource::Send, client ip is empty");
    return len;
  }

  len = sendto(udp_sock_, (char*)u8Buf, u16Len, flags, (const struct sockaddr*)&send_addr_, sizeof(send_addr_));
  if (len == -1) {
    LogError("SocketSource::Send, errno:%d", errno);
    Close();
  }
  return len;
}

bool isTimeout = false;

int SocketSource::Receive(UdpPacket& udpPacket, uint16_t u16Len, int flags,
                       int iTimeout) {  
  int len = -1;
  bool ret = true;
  if (!IsOpened()) ret = Open();

  if (ret) {
    fd_set rfd;
    timeval timeout;

    timeout.tv_sec = 0;
    timeout.tv_usec = iTimeout;
    FD_ZERO(&rfd);

    FD_SET(udp_sock_, &rfd);
    if (!is_select_) {
      sockaddr_in clientAddr;
      socklen_t addrLen = sizeof(sockaddr);
      len = recvfrom(udp_sock_, (char*)udpPacket.buffer, u16Len, flags,
                      (sockaddr*)&clientAddr, &addrLen);
      if(len == -1) { is_select_ = true; }
      else {
        udpPacket.ip = clientAddr.sin_addr.s_addr;
        udpPacket.port = ntohs(clientAddr.sin_port);
      }
    } else {
      int cnt = select(udp_sock_ + 1, &rfd, NULL, NULL, &timeout);
      if (cnt > 0) {
        is_select_ = false;
        sockaddr_in clientAddr;
        socklen_t addrLen = sizeof(sockaddr);
        len = recvfrom(udp_sock_, (char*)udpPacket.buffer, u16Len, flags,
                     (sockaddr*)&clientAddr, &addrLen);      
        udpPacket.ip = clientAddr.sin_addr.s_addr;
        udpPacket.port = ntohs(clientAddr.sin_port);  
      } else if (cnt == 0) {
        len = 0;
        udpPacket.is_timeout = true;        
      } else {
        LogWarning("Select timeout error");
      }
    } 
  }

  return len;
}

void SocketSource::SetSocketBufferSize(uint32_t u32BufSize) {
  setsockopt(udp_sock_, SOL_SOCKET, SO_RCVBUF, (char*)&u32BufSize,
      sizeof(u32BufSize));
}

void SocketSource::SetClientIp(std::string client_ip) {
  client_ip_ = client_ip;

  if (!IsOpened()) Open();

  if(udp_sock_ < 0) {
      LogError("Error creating socket");
  }

  if(client_ip.substr(client_ip.length() - 3) == "255") {
    int on = 1;
    int ret = setsockopt(udp_sock_, SOL_SOCKET, SO_BROADCAST, (char*)&on, sizeof(on));
    if(ret < 0){
      LogError("SocketSource::Send setsocketopt failed!");
    }
  }
  std::memset(&send_addr_, 0, sizeof(send_addr_));
  send_addr_.sin_family = AF_INET;
  send_addr_.sin_port = htons(udp_port_);
  send_addr_.sin_addr.s_addr = inet_addr(client_ip_.c_str());
}

bool SocketSource::sendsomeip_subscribe_ipv4(std::string someip_ip, std::string host_addr, int port, int fault_message_port) {
  if (udp_sock_ == -1) return false;
  sockaddr_in addr2;
  memset(&addr2, 0, sizeof(sockaddr_in));
  // Prepare the sockaddr_in structure
  addr2.sin_family = AF_INET;
  addr2.sin_addr.s_addr = inet_addr(someip_ip.c_str());
  addr2.sin_port = htons(30490);
  // 构造SomeIP消息
  SomeIPMessage msg;
  msg.header.someip_service_id = 0xffff;
  msg.header.someip_method_id = htons(0x8100);
  msg.header.someip_length = 0x30000000;
  msg.header.someip_client_id = 0x0000;
  msg.header.someip_session_id = htons(0x0001);
  msg.header.someip_version = 0x01;
  msg.header.someip_interface_version = 0x01;
  msg.header.someip_message_type = 0x02;
  msg.header.someip_return_code = 0x00;
  uint8_t data[] = {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x06, 0x00, 0x00, 0x10, 0xa2, 0xa0, \
                  0x00, 0x0a, 0x01, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, \
                  0x00, 0x09, 0x04, 0x00, 0x0a, 0x17, 0x01, 0x65, 0x00, 0x11, 0xb5, 0xb1};
  {
    data[38] = port / 256;
    data[39] = port % 256;
    std::istringstream ss(host_addr);
    std::string segment;  
    int segmentsCount = 0;  
    while (std::getline(ss, segment, '.')) {  
        if (segment.empty() || segmentsCount >= 4)  
            return false;
        data[32 + segmentsCount] = std::stoi(segment);  
        segmentsCount++;
    }
  }
  // 发送消息
  memcpy(msg.payload, data, 40);
  int bytes_sent = sendto(udp_sock_, reinterpret_cast<char*>(&msg), sizeof(msg), 0, reinterpret_cast<const sockaddr*>(&addr2), sizeof(sockaddr_in));
  LogInfo("someip subscribe");
  if (bytes_sent == SOCKET_ERROR) {
      LogError("Error sending someip subscribe message");
      return false;
  }
  if (fault_message_port != 0) {  
    uint8_t fm_data[] = {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x06, 0x00, 0x00, 0x10, 0xa2, 0x9e, \
                    0x00, 0x0a, 0x01, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, \
                    0x00, 0x09, 0x04, 0x00, 0x0a, 0x17, 0x01, 0x65, 0x00, 0x11, 0x9c, 0x40};
    {
        fm_data[38] = fault_message_port / 256;
        fm_data[39] = fault_message_port % 256;
        std::istringstream ss(host_addr);
        std::string segment;  
        int segmentsCount = 0;  
        while (std::getline(ss, segment, '.')) {  
            if (segment.empty() || segmentsCount >= 4)  
                return false;
            fm_data[32 + segmentsCount] = std::stoi(segment);  
            segmentsCount++;
        }
    }
    memcpy(msg.payload, fm_data, 40);
    int bytes_sent_fm = sendto(udp_sock_, reinterpret_cast<char*>(&msg), sizeof(msg), 0, reinterpret_cast<const sockaddr*>(&addr2), sizeof(sockaddr_in));
    LogInfo("someip fault message subscribe");
    if (bytes_sent_fm == SOCKET_ERROR) {
        LogError("Error sending someip fault message subscribe message");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));//sleep 10毫秒
  }
  else {
    LogInfo("someip fault message subscribe is not enabled");
  }
  return true;
}