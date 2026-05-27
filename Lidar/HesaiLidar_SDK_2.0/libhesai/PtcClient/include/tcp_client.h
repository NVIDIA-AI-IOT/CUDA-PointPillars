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
 * File:   tcp_client.h
 * Author: Felix Zou<zouke@hesaitech.com>
 *
 * Created on Sep 5, 2019, 10:46 AM
 */

#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#ifdef _MSC_VER
#include <winsock2.h>
#include <ws2tcpip.h> 
#else
typedef unsigned int SOCKET;
#endif
#include "client_base.h"
#include <stdint.h>
#include <string.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace hesai
{
namespace lidar
{
class TcpClient : public ClientBase{
 public:
  explicit TcpClient();
  virtual ~TcpClient();

  TcpClient(const TcpClient &orig) = delete;

  virtual bool Open(std::string IPAddr, uint16_t port, bool bAutoReceive = false, 
                    const char* cert = nullptr, const char* private_key = nullptr, const char* ca = nullptr);
  virtual bool TryOpen(std::string IPAddr, uint16_t port, bool bAutoReceive = false, 
                    const char* cert  = nullptr, const char* private_key = nullptr, const char* ca = nullptr, uint32_t timeout = 1);
  bool Open();
  virtual void Close();
  virtual bool IsOpened();
  bool IsOpened(bool bExpectation);
  virtual int Send(uint8_t *u8Buf, uint16_t u16Len, int flags = 0);
  virtual int Receive(uint8_t *u8Buf, uint32_t u16Len, int flags = 0);

  /**
   * @brief 设置接收超时
   * @param u32Timeout 超时时间/ms
   * @return
   */
  virtual bool SetReceiveTimeout(uint32_t u32Timeout);

  /**
   * @brief 设置收发超时
   * @param u32RecMillisecond 接收超时/ms
   * @param u32SendMillisecond 发送超时/ms
   * @return
   */
  virtual int SetTimeout(uint32_t u32RecMillisecond, uint32_t u32SendMillisecond);

  /**
   * @brief 设置自动接收模式下Buff的大小
   * @param size
   */
  virtual void SetReceiveBufferSize(const uint32_t &size);

 private:
  /**
   * monitor file descriptor and wait for I/O operation
   */
  int WaitFor(const int &socketfd, uint32_t timeoutSeconds = 1);

 protected:
  static const uint32_t kDefaultTimeout = 500;

  std::string m_sServerIP;
  uint16_t ptc_port_;
  SOCKET m_tcpSock;
  bool m_bLidarConnected;
  uint32_t m_u32ReceiveBufferSize;
  // 收发超时/ms
  uint32_t m_u32RecTimeout = kDefaultTimeout;
  uint32_t m_u32SendTimeout = kDefaultTimeout;
};
}
}
#endif /* TCPCLIENT_H */
