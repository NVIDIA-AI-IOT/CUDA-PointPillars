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
 * File:        lidar_communication_header.h
 * Author:      Felix Zou<zouke@hesaitech.com>
 * Description:
 */

#ifndef LIDARCOMMUNICATIONHEADER_H
#define LIDARCOMMUNICATIONHEADER_H

#include <endian.h>
#include <arpa/inet.h>
#pragma pack(push, 1)

struct LidarCommunicationHeader {
  // [5:4] 00:NOP, 2'b01:read, 2'b10:write
  static const uint8_t kNOP = 0x00;
  static const uint8_t kRead = 0x10;
  static const uint8_t kWrite = 0x20;
  // [3:2] operation width, 0: 1byte, 1: 2bytes, 2: 4bytes
  static const uint8_t kOperationByte1 = 0x00;
  static const uint8_t kOperationByte2 = 0x04;
  static const uint8_t kOperationByte4 = 0x08;
  // [1] 0: auto increament, 1: addr fixed
  static const uint8_t kAddrAutoIncrement = 0x00;
  static const uint8_t kAddrFixed = 0x02;
  // [0] 1: send ack flag
  static const uint8_t kAckFlag = 0x01;

  uint8_t m_u8Indicator;
  uint8_t m_u8CmdOption;
  uint32_t m_u32Addr;
  uint16_t m_u16Length;

  void Init(uint8_t u8CmdOption, uint32_t u32Addr, uint16_t u16Len,
            uint8_t u8Indicator = 0xA0) {
    m_u8Indicator = u8Indicator;
    m_u8CmdOption = u8CmdOption;
    m_u32Addr = htobe32(u32Addr);
    m_u16Length = htobe16(u16Len);
  }

  uint32_t GetAddr() const { return ntohl(m_u32Addr); }

  uint16_t GetOpertationLength() const { return be16toh(m_u16Length); }
  uint8_t GetOperationByte() const {
    return (m_u8CmdOption & kOperationByte4)
               ? 4
               : ((m_u8CmdOption & kOperationByte2) ? 2 : 1);
  }

  uint16_t GetDataLength() const {
    return (m_u8CmdOption & kRead)
               ? 0
               : (GetOperationByte() * be16toh(m_u16Length));
  }
  uint16_t GetRspDataLength() const {
    return (m_u8CmdOption & kWrite)
               ? 0
               : (GetOperationByte() * be16toh(m_u16Length));
  }

  uint16_t GetDataOffset() const { return sizeof(LidarCommunicationHeader); }
  uint16_t GetCrcOffset() const {
    return sizeof(LidarCommunicationHeader) + GetDataLength();
  }

  // size: header + data + crc32
  uint16_t GetPacketSize() const { return GetCrcOffset() + sizeof(uint32_t); }
  uint16_t GetRspPacketSize() const {
    return (m_u8CmdOption & kWrite)
               ? (m_u8CmdOption & kAckFlag
                      ? sizeof(LidarCommunicationHeader) + sizeof(uint32_t) + 1
                      : 0)
               : (sizeof(LidarCommunicationHeader) +
                  GetOperationByte() * be16toh(m_u16Length) + sizeof(uint32_t));
  }

};
#pragma pack(pop)

#endif
