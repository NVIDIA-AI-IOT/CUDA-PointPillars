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


#include "serial_client.h"

using namespace hesai::lidar;

SerialClient::SerialClient() {
  CRCInit();
}

SerialClient::~SerialClient() {
}

void SerialClient::AddEndStreamEncode(u8Array_t &byteStreamOut, const CmdType type) {
  uint32_t rand = 0;
  if (type == kCmd) {
    rand = GetRandom();
    byteStreamOut.push_back(static_cast<uint8_t>(rand));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 8));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 16));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 24));
  }
  uint32_t crc_in_len = byteStreamOut.size() - crc_begin;
  uint32_t crc_in_len_true = 0;
  if (crc_in_len % 4 != 0 && crc_in_len == 30) {
    crc_in_len_true = 32;
  } else if (crc_in_len % 4 != 0 && crc_in_len != 30) {
    crc_in_len_true = 4 * (crc_in_len / 4 + 1);
  } else {
    crc_in_len_true = crc_in_len;
  }
  rand = CRCCalc(byteStreamOut.data() + crc_begin, crc_in_len, crc_in_len_true - crc_in_len);
  if (type == kCmd) {
    byteStreamOut.push_back(static_cast<uint8_t>(rand));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 8));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 16));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 24));
  } else if (type == kOta) {
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 24));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 16));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 8));
    byteStreamOut.push_back(static_cast<uint8_t>(rand >> 0));
  }
  byteStreamOut.push_back(0xEE);
  byteStreamOut.push_back(0xFF);
}

void SerialClient::SerialStreamEncode(const CmdType type, u8Array_t &byteStream) {
  SerialHeader *pHeader = (SerialHeader *)byteStream.data();
  if (type == kCmd)
    pHeader->InitCmd();
  else if (type == kOta)
    pHeader->InitOta();
  AddEndStreamEncode(byteStream, type);
}

bool SerialClient::SerialStreamDecode(const CmdType type, const u8Array_t &byteStreamIn, u8Array_t &byteStreamOut) {
  int len = byteStreamIn.size();
  if (byteStreamIn[len - 2] != 0xEE || byteStreamIn[len - 1] != 0xFF) {
    return false;
  }
  uint32_t crc_in_len = len - 13;
  uint32_t crc_in_len_true = 0;
  u8Array_t crc_in;
  if (crc_in_len % 4 != 0 && crc_in_len == 30) {
    crc_in_len_true = 32;
  } else if (crc_in_len % 4 != 0 && crc_in_len != 30) {
    crc_in_len_true = 4 * (crc_in_len / 4 + 1);
  } else {
    crc_in_len_true = crc_in_len;
  }
  uint8_t *ptr = (uint8_t *)byteStreamIn.data();
  ptr += len - 6;
  uint32_t crc = 0;
  if (type == kCmd)
    crc = *((uint32_t *)ptr);
  else if (type == kOta)
    crc = *(ptr) * 0x1000000 + *(ptr + 1) * 0x10000 + *(ptr + 2) * 0x100 + *(ptr + 3);
  uint32_t check_crc = CRCCalc(byteStreamIn.data() + crc_begin, crc_in_len, crc_in_len_true - crc_in_len);
  if(crc != check_crc) {
    return false;
  }
  byteStreamOut.resize(len - 17);
  std::copy_n(byteStreamIn.begin() + crc_begin, len - 17, byteStreamOut.begin());
  return true;
}


int SerialClient::QueryCommand(const uint8_t cmd, const u8Array_t &payload, u8Array_t &byteStreamOut, uint32_t timeout) {
  if (source_send_ == nullptr || source_recv_ == nullptr || source_send_->IsOpened() == false || source_recv_->IsOpened() == false) return kInvalidEquipment;
  u8Array_t sendCommand;
  sendCommand.resize(sizeof(SerialHeader));
  sendCommand.push_back(static_cast<uint8_t>(payload.size() + 1));
  sendCommand.push_back(cmd);
  sendCommand.resize(payload.size() + sizeof(SerialHeader) + 2);
  memcpy(sendCommand.data() + sizeof(SerialHeader) + 2, payload.data(), payload.size());
  SerialStreamEncode(kCmd, sendCommand);
  source_recv_->SetReceiveStype(SERIAL_CLEAR_RECV_BUF);
  if (source_send_->Send(sendCommand.data(), sendCommand.size(), 0) < 0) {
    return kInvalidEquipment;
  }
  UdpPacket udp_packet;
  int size = source_recv_->Receive(udp_packet, kBufSize, 1, timeout);
  if (size < 0) return kReadTimeout; 
  u8Array_t recvData;
  recvData.resize(size);
  memcpy(recvData.data(), udp_packet.buffer, size);
  bool ret = SerialStreamDecode(kCmd, recvData, byteStreamOut);
  if (ret == false) return kInvalidData;
  return 0;
}

int SerialClient::ChangeMode(uint8_t mode) {
  u8Array_t payload, byteStreamOut;
  payload.push_back(mode);
  payload.push_back(0x00);
  int ret = QueryCommand(0x03, payload, byteStreamOut, 5000);
  if (ret != 0) {
    return ret;
  }
  if (byteStreamOut.size() != 5 || byteStreamOut[1] != 0x03 || byteStreamOut[2] != mode || byteStreamOut[3] != 0x00) {
    return kInvalidData;
  }
  if (byteStreamOut[4] != 0x00) {
    return CmdErrorCode2RetCode(byteStreamOut[4]);
  }
  return 0;
}

int SerialClient::ChangeUpgradeMode() {
  u8Array_t payload, byteStreamOut;
  payload.push_back(0x04);
  payload.push_back(0x00);
  int ret = QueryCommand(0x03, payload, byteStreamOut, 20000);
  if (ret != 0) {
    return ret;
  }
  if (byteStreamOut.size() != 5 || byteStreamOut[1] != 0x03 || byteStreamOut[2] != 0x04 || byteStreamOut[3] != 0x00) {
    return kInvalidData;
  }
  if (byteStreamOut[4] != 0x00) {
    return CmdErrorCode2RetCode(byteStreamOut[4]);
  }
  return 0;
}

int SerialClient::GetCorrectionInfo(u8Array_t &dataOut) {
  ChangeMode(0x01);
  u8Array_t payload, byteStreamOut;
  payload.push_back(0x07);
  payload.push_back(0x00);
  int ret = QueryCommand(0x02, payload, byteStreamOut, 5000);
  ChangeMode(0x00);
  if (ret != 0) {
    return ret;
  }
  if (byteStreamOut.size() != (size_t)(byteStreamOut[0] + 2) || byteStreamOut[1] != 0x02 || byteStreamOut[2] != 0x07 
      || byteStreamOut[3] != 0x00 || (size_t)(byteStreamOut[4] + 6) != byteStreamOut.size()) {
    return kInvalidData;
  }
  if (byteStreamOut[byteStreamOut.size() - 1] != 0x00) {
    return CmdErrorCode2RetCode(byteStreamOut[byteStreamOut.size() - 1]);
  }
  dataOut.resize(byteStreamOut[4]);
  std::copy_n(byteStreamOut.begin() + 5, byteStreamOut[4], dataOut.begin());
  return 0;
}

int SerialClient::GetSnInfo(u8Array_t &dataOut) {
  u8Array_t payload, byteStreamOut;
  payload.push_back(0x0A);
  payload.push_back(0x01);
  payload.resize(38);
  int ret = QueryCommand(0x02, payload, byteStreamOut, 5000);
  if (ret != 0) {
    return ret;
  }
  if (byteStreamOut.size() != 41 || byteStreamOut[1] != 0x02 || byteStreamOut[2] != 0x0A || byteStreamOut[3] != 0x01) {
    return kInvalidData;
  }
  if (byteStreamOut[byteStreamOut.size() - 1] != 0x00) {
    return CmdErrorCode2RetCode(byteStreamOut[byteStreamOut.size() - 1]);
  }
  dataOut.resize(36);
  std::copy_n(byteStreamOut.begin() + 4, 36, dataOut.begin());
  return 0;
}

void SerialClient::SetSerial(Source* source_send, Source* source_recv) {
  source_send_ = source_send;
  source_recv_ = source_recv;
}

void SerialClient::CRCInit() {
  uint32_t i, j;

  for (i = 0; i < 256; i++) {
    uint32_t k = 0;
    for (j = (i << 24) | 0x800000; j != 0x80000000; j <<= 1)
      k = (k << 1) ^ (((k ^ j) & 0x80000000) ? 0x04c11db7 : 0);

    m_CRCTable[i] = k;
  }
}

uint32_t SerialClient::CRCCalc(const uint8_t *bytes, int len,  int zeros_num) {
  uint32_t i_crc = 0xffffffff;
  int i = 0;
  for (i = 0; i < len; i++)
    i_crc = (i_crc << 8) ^ m_CRCTable[((i_crc >> 24) ^ bytes[i]) & 0xff];
  for (i = 0; i < zeros_num; i++)
    i_crc = (i_crc << 8) ^ m_CRCTable[((i_crc >> 24) ^ 0) & 0xff];
  return i_crc;
}

uint32_t SerialClient::GetRandom() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<uint32_t> distribution;
  return distribution(generator);
}

int SerialClient::CmdErrorCode2RetCode(uint8_t error_code) {
  int ret = 0;
  switch(error_code) 
  {
    case 0x01:
    case 0x02:
      ret = kInvalidCmd;
      break;
    case 0x03:
    case 0x04:
      ret = kExecutionError;
      break;
    case 0x31:
      ret = kPacketLoss;
      break;
    case 0x32:
      ret = kChecksumFailure;
      break;
    case 0x33:
      ret = kUnrecognisedFormat;
      break;
    case 0x05:
      ret = kTimeout;
      break;
    default:
      ret = kUndefine;
      break;
  }
  return ret;
}