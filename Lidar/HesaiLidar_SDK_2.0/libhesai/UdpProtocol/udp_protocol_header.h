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
#include "plat_utils.h"
#ifndef LIDAR_PROTOCOL_HEADER_H
#define LIDAR_PROTOCOL_HEADER_H
#ifdef _MSC_VER
#define NOMINMAX
#include <winsock2.h>
#include <windows.h>
#endif
namespace hesai
{
namespace lidar
{
#pragma pack(push, 1)

inline int doubleToInt(double data) {
  return static_cast<int>(data + 0.0625);
}

inline int floatToInt(float data) {
  return static_cast<int>(data + 0.0625f);
}

struct HS_LIDAR_PRE_HEADER {
  static const uint16_t kDelimiter = 0xffee;

  uint16_t m_u16Delimiter;
  uint8_t m_u8VersionMajor;
  uint8_t m_u8VersionMinor;
  uint8_t m_u8StatusInfoVersion;
  uint8_t m_u8Reserved1;

  inline bool IsValidDelimiter() const {
    return little_to_native(m_u16Delimiter) == kDelimiter;
  }
  inline uint16_t GetDelimiter() const { return little_to_native(m_u16Delimiter); }
  inline uint8_t GetVersionMajor() const { return m_u8VersionMajor; }
  inline uint8_t GetVersionMinor() const { return m_u8VersionMinor; }
  inline uint8_t GetStatusInfoVersion() const { return m_u8StatusInfoVersion; }
};

struct ReservedInfo1 {
  uint16_t m_u16Sts;
  uint8_t m_u8ID;

  inline uint8_t GetID() const { return m_u8ID; }
  inline uint16_t GetData() const { return little_to_native(m_u16Sts); }
};

struct ReservedInfo2 {
  uint16_t m_u16Sts;
  uint8_t m_u8ID;
  inline uint8_t GetID() const { return m_u8ID; }
  inline uint16_t GetData() const { return little_to_native(m_u16Sts); }
};

struct ReservedInfo3 {
  uint16_t m_u16Sts;
  uint8_t m_u8ID;
  inline uint8_t GetID() const { return m_u8ID; }
  inline uint16_t GetData() const { return little_to_native(m_u16Sts); }
};

inline bool hasSeqNum(uint8_t status) { return status & 1; }
inline bool hasImu(uint8_t status) { return status & 2; }
inline bool hasFunctionSafety(uint8_t status) { return status & 4; }
inline bool hasSignature(uint8_t status) { return status & 8; }
inline bool hasConfidence(uint8_t status) { return status & 0x10; }
inline bool hasWeightFactor(uint8_t status) { return status & 0x20; }
inline bool hasEnvLight(uint8_t status) { return status & 0x40; }
inline bool hasSlope(uint8_t status) { return status & 0x20; }
inline bool hasSelfDefine(uint8_t status) { return status & 0x40; }

#pragma pack(pop)
}  // namespace lidar
}  // namespace hesai
#endif
