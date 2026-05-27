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
 * File:   tcp_client.cc
 * Author: Felix Zou<zouke@hesaitech.com>
 */

#include "tcp_client.h"
#ifdef _MSC_VER
#include <winsock2.h>
#include <ws2tcpip.h> 
#pragma comment(lib, "ws2_32.lib")  // Winsock Library
#include <BaseTsd.h>

typedef int socklen_t;
#define MSG_DONTWAIT (0x40)
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h> 
#endif
#include <plat_utils.h>
#include <string.h>
#include <algorithm>
#include <iostream>
using namespace hesai::lidar;
TcpClient::TcpClient() {
  m_sServerIP.clear();
  ptc_port_ = 0;
  m_tcpSock = (SOCKET)(-1);
  m_bLidarConnected = false;
  m_u32ReceiveBufferSize = 4096;
}

TcpClient::~TcpClient() {
  Close();
}

void TcpClient::Close() {

  // m_sServerIP.clear();
  // ptc_port_ = 0;
  m_bLidarConnected = false;

  if ((int)m_tcpSock != -1) {
#ifdef _MSC_VER
          closesocket(m_tcpSock);
          WSACleanup();
#else
          close(m_tcpSock);
#endif
    m_tcpSock = (SOCKET)(-1);
  }
}

bool TcpClient::TryOpen(std::string IPAddr, uint16_t u16Port, bool bAutoReceive,
          const char* cert, const char* private_key, const char* ca, uint32_t timeout) {
  (void)bAutoReceive;
  (void)cert;          
  (void)private_key;
  (void)ca;
  if (IsOpened(true) && m_sServerIP == IPAddr && u16Port == ptc_port_) {
    return true;
  }
  if (IsOpened()) Close();
  m_sServerIP = IPAddr;
  ptc_port_ = u16Port;
  
#ifdef _MSC_VER
  WSADATA wsaData;
  WORD version = MAKEWORD(2, 2);
  int res = WSAStartup(version, &wsaData);  // win sock start up
  if (res) {
      LogError("Initilize winsock error !");
      return false;
  }
#endif  
  struct sockaddr_in serverAddr;

  m_tcpSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  if ((int)m_tcpSock == -1) { 
#ifdef _MSC_VER
    WSACleanup();
#endif
    return false;
  }

  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(ptc_port_);
  if (inet_pton(AF_INET, m_sServerIP.c_str(), &serverAddr.sin_addr) <= 0) {
    Close();
    LogError("TryOpen inet_pton error:%s", m_sServerIP.c_str());
    return false;
  }

  // 设置非阻塞模式  
#ifdef _MSC_VER  
  u_long mode = 1; // 1为非阻塞模式  
  ioctlsocket(m_tcpSock, FIONBIO, &mode);
#else  
  int flags = fcntl(m_tcpSock, F_GETFL, 0); 
  fcntl(m_tcpSock, F_SETFL, flags | O_NONBLOCK);  
#endif 

  int result = connect(m_tcpSock, (sockaddr*)&serverAddr, sizeof(serverAddr));  
  if (result < 0) {  
#ifdef _MSC_VER  
    if (WSAGetLastError() != WSAEWOULDBLOCK) 
#else  
    if (errno != EINPROGRESS)  
#endif  
    {
      LogError("socket Connection error.");  
      Close();
      return false;  
    }  
  }

  fd_set writefds;  
  FD_ZERO(&writefds);  
  FD_SET(m_tcpSock, &writefds);  
  struct timeval tv;  
  tv.tv_sec = timeout; // 超时1秒  
  tv.tv_usec = 0;   
  result = select((int)m_tcpSock + 1, nullptr, &writefds, nullptr, &tv);  
  if (result <= 0) {  
    Close();
    return false;  
  } else {
    int slen = sizeof(int);
    int error = -1;
    getsockopt(m_tcpSock, SOL_SOCKET, SO_ERROR, (char *)&error, (socklen_t *)&slen);
    if (error != 0) {
      Close();
      return false;
    }
  }
  LogInfo("TryOpen succeed, IP %s port %u", m_sServerIP.c_str(), ptc_port_);
  
#ifdef _MSC_VER  
  mode = 0; // 0为阻塞模式  
  ioctlsocket(m_tcpSock, FIONBIO, &mode);  
#else  
  flags = fcntl(m_tcpSock, F_GETFL, 0); 
  fcntl(m_tcpSock, F_SETFL, flags & ~O_NONBLOCK); 
#endif  

  m_bLidarConnected = true;
  return true;
}

bool TcpClient::Open(std::string IPAddr, uint16_t u16Port, bool bAutoReceive,
          const char* cert, const char* private_key, const char* ca) {
  (void)bAutoReceive;
  (void)cert; 
  (void)private_key;
  (void)ca;
  if (IsOpened(true) && m_sServerIP == IPAddr && u16Port == ptc_port_) {
    return true;
  }
#ifdef _MSC_VER
  LogInfo("Open() IP %s port %u", IPAddr.c_str(), u16Port);
#else
  LogInfo("Open() IP %s port %u", IPAddr.c_str(), u16Port);
#endif
  Close();

  m_sServerIP = IPAddr;
  ptc_port_ = u16Port;
  return Open();
}

bool TcpClient::Open() {
#ifdef _MSC_VER
  WSADATA wsaData;
  WORD version = MAKEWORD(2, 2);
  int res = WSAStartup(version, &wsaData);  // win sock start up
  if (res) {
      std::cerr << "Initilize winsock error !" << std::endl;
      return false;
  }
#endif  
  struct sockaddr_in serverAddr;

  m_tcpSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  if ((int)m_tcpSock == -1) { 
#ifdef _MSC_VER
    WSACleanup();
#endif
    return false;
  }

  memset(&serverAddr, 0, sizeof(serverAddr));

  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(ptc_port_);
  if (inet_pton(AF_INET, m_sServerIP.c_str(), &serverAddr.sin_addr) <= 0) {
    Close();
    LogError("Open inet_pton error:%s", m_sServerIP.c_str());

    return false;
  }

  if (::connect(m_tcpSock, (sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
    if (EINPROGRESS != errno && EWOULDBLOCK != errno) {
      LogError("connect failed %d", errno);
    } else if (EINPROGRESS == errno) {
      LogWarning("connect lidar time out");
    }
    Close();
    LogError("connect lidar fail errno %d", errno);
    return false;
  }
  
  LogInfo(" succeed, IP %s port %u", m_sServerIP.c_str(), ptc_port_);

  m_bLidarConnected = true;

  return true;
}

bool TcpClient::IsOpened() {
  return m_bLidarConnected;
}

bool TcpClient::IsOpened(bool bExpectation) {
  return m_bLidarConnected == bExpectation;
}

int TcpClient::Send(uint8_t *u8Buf, uint16_t u16Len, int flags) {
  int len = -1;
  bool ret = true;

  if (!IsOpened()) ret = Open();

  if (ret) {
    len = send(m_tcpSock, (char*)u8Buf, u16Len, flags);
    if (len != u16Len && errno != EAGAIN && errno != EWOULDBLOCK &&
        errno != EINTR) {
      LogError("Send errno %d", errno);
      Close();
    }
  }
  return len;
}

int TcpClient::Receive(uint8_t *u8Buf, uint32_t u32Len, int flags) {
  int len = -1;
  bool ret = true;

  if (!IsOpened()) ret = Open();

  int tick = GetMicroTickCount();
  if (ret) {
    if (flags == 0xFF) {
  // 设置非阻塞模式  
#ifdef _MSC_VER  
      u_long mode = 1; // 1为非阻塞模式  
      ioctlsocket(m_tcpSock, FIONBIO, &mode);
#else  
      int flags = fcntl(m_tcpSock, F_GETFL, 0); 
      fcntl(m_tcpSock, F_SETFL, flags | O_NONBLOCK);  
#endif 
    }
    len = recv(m_tcpSock, (char*)u8Buf, u32Len, flags);
    if (len == 0 || (len == -1 && errno != EINTR && errno != EAGAIN &&
                     errno != EWOULDBLOCK)) {
      if (flags != 0xFF) {
        LogError("Receive, len: %d errno: %d", len, errno);
        Close();
      }
    }
    if (flags == 0xFF) {
#ifdef _MSC_VER  
      u_long mode = 0; // 0为阻塞模式  
      ioctlsocket(m_tcpSock, FIONBIO, &mode);  
#else  
      flags = fcntl(m_tcpSock, F_GETFL, 0); 
      fcntl(m_tcpSock, F_SETFL, flags & ~O_NONBLOCK); 
#endif 
    }
  }

  int delta = GetMicroTickCount() - tick;

  if (delta >= 1000000) {
    LogDebug("Receive execu: %dus", delta);
  }

  return len;
}


bool TcpClient::SetReceiveTimeout(uint32_t u32Timeout) {
  if ((int)m_tcpSock == -1) {
    LogWarning("TcpClient not open");
    return false;
  }
#ifdef _MSC_VER
  int timeout_ms = u32Timeout;
  int retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout_ms,
      sizeof(timeout_ms));
#else
  uint32_t sec = u32Timeout / 1000;
  uint32_t msec = u32Timeout % 1000;
  struct timeval timeout;
  timeout.tv_sec = sec;
  timeout.tv_usec = msec * 1000;
  int retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_RCVTIMEO,
                          (const void *)&timeout, sizeof(timeval));
#endif
  return retVal == 0;
}

int TcpClient::SetTimeout(uint32_t u32RecMillisecond,
                          uint32_t u32SendMillisecond) {
  if ((int)m_tcpSock == -1) {
    LogWarning("TcpClient not open");
    return -1;
  }
  m_u32RecTimeout = u32RecMillisecond;
  m_u32SendTimeout = u32SendMillisecond;
#ifdef _MSC_VER
  int timeout_ms = u32RecMillisecond;
  int retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_RCVTIMEO,
                        (char*)&timeout_ms, sizeof(timeout_ms));
#else
  uint32_t sec = u32RecMillisecond / 1000;
  uint32_t msec = u32RecMillisecond % 1000;

  struct timeval timeout;
  timeout.tv_sec = sec;
  timeout.tv_usec = msec * 1000;
  int retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_RCVTIMEO,
                          (const void *)&timeout, sizeof(timeval));
#endif
  if (retVal == 0) {
#ifdef _MSC_VER
    int send_timeout_ms = u32SendMillisecond;
    retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_SNDTIMEO,
                        (char*)&send_timeout_ms, sizeof(send_timeout_ms));    
#else    
    uint32_t send_sec = u32SendMillisecond / 1000;
    uint32_t send_msec = u32SendMillisecond % 1000;

    struct timeval send_timeout;
    send_timeout.tv_sec = send_sec;
    send_timeout.tv_usec = send_msec * 1000;
    retVal = setsockopt(m_tcpSock, SOL_SOCKET, SO_SNDTIMEO,
                        (const void *)&send_timeout, sizeof(timeval));
#endif
  }
  return retVal;
}

void TcpClient::SetReceiveBufferSize(const uint32_t &size) {
  if ((int)m_tcpSock == -1) {
    LogWarning("TcpClient not open");
    return;
  }

  m_u32ReceiveBufferSize = size;
  uint32_t recbuffSize;
  socklen_t optlen = sizeof(recbuffSize);
  int ret = getsockopt(m_tcpSock, SOL_SOCKET, SO_RCVBUF, (char*)&recbuffSize, &optlen);
  if (ret == 0 && recbuffSize < size) {
    setsockopt(m_tcpSock, SOL_SOCKET, SO_RCVBUF, (char*)&size, sizeof(size));
  }
}


int TcpClient::WaitFor(const int &socketfd, uint32_t timeoutSeconds) {
  struct timeval tv;
  tv.tv_sec = timeoutSeconds;
  tv.tv_usec = 0;
  fd_set fds;

  FD_ZERO(&fds);
  FD_SET(socketfd, &fds);
  const int selectRet = select(socketfd + 1, &fds, nullptr, nullptr, &tv);

  return selectRet;
}

