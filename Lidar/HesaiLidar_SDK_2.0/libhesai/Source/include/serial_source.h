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
 * File:   serial_source.h
 * Author: chang xingshuo<changxingshuo@hesaitech.com>
 *
 * Created on Sep 10, 2024, 19:56 PM  
 */

#ifndef PANDARSERIAL_H
#define PANDARSERIAL_H

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>
#include "source.h"
#include "blocking_ring.h"
#include "inner_com.h"
#ifdef _MSC_VER
#define NOMINMAX
#include <Windows.h>
#else
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace hesai
{
namespace lidar
{
          
#define    BOTHER 0010000
#define    HS_NCCS 19
#define    SERIAL_POINT_CLOUD_RECV  0
#define    SERIAL_COMMAND_RECV      1
#define    SERIAL_CLEAR_RECV_BUF    2

#ifndef _MSC_VER
struct termios2 {
  tcflag_t c_iflag;		/* input mode flags */
  tcflag_t c_oflag;		/* output mode flags */
  tcflag_t c_cflag;		/* control mode flags */
  tcflag_t c_lflag;		/* local mode flags */
  cc_t c_line;			/* line discipline */
  cc_t c_cc[HS_NCCS];		/* control characters */
  speed_t c_ispeed;		/* input speed */
  speed_t c_ospeed;		/* output speed */
};
#endif
class SerialSource : public Source {
public:
  ~SerialSource();
  SerialSource(const std::string dev, int baudrate, int point_cloud_baudrate = 3125000);
  /* Disable copy constructor */
  SerialSource(const SerialSource &orig) = delete;

  /**
   * @param dev         path to serial device
   * @param baudrate    number of baud per second
   * @return 0 on success
   */
  int Open(const char *dev, int baudrate);
  virtual bool Open();
  virtual bool IsOpened();
  virtual void Close();
  virtual int Receive(UdpPacket& udpPacket, uint16_t u16Len, int flags = 0,
                      int timeout = 1000);
  virtual int Send(uint8_t* u8Buf, uint16_t u16Len, int flags = 0);
  virtual void SetSocketBufferSize(uint32_t u32BufSize) { return; };
  /**
   * @Desc   flush the COM
   * @return 0 on success, otherwise failure
   */
  int Flush(void);
  int FlushInput(void);
  int FlushOutput(void);
  /**
   * @Desc Check if the FD is valid.
   * @return true for valid
   */
  virtual void SetReceiveStype(int type);

private:
  void ReceivedThread();
  bool running;
  std::thread* runningRecvThreadPtr;
  BlockingRing<UdpPacket, 10> cmdAckRecvBuf;
  BlockingRing<UdpPacket, kPacketBufferSize> pointCloudRecvBuf;
  static const uint32_t kOneRecvLength = 80;
  static const uint32_t kDataMaxLength = 1024 * 80; // is greater than the u16Len parameter of the function Receive
#ifdef _MSC_VER
  HANDLE m_iFd;
#else
  int32_t m_iFd;
#endif
  std::string dev_;
  int baudrate_;
  int point_cloud_baudrate_;
  uint8_t* serialData;
  uint8_t* serialData_backup;
  uint32_t dataIndex;
  uint32_t dataLength;
  int receiveStype = SERIAL_POINT_CLOUD_RECV;

private:
  int WaitRead(int32_t timeout);
};
}
}
#endif /* PANDARSERIAL_H */

