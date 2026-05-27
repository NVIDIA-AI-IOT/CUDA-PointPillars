/************************************************************************************************
  Copyright(C)2023 Hesai Technology Co., Ltd.
  All code in this repository is released under the terms of the following [Modified BSD License.]
  Modified BSD License:
  Redistribution and use in source and binary forms,with or without modification,are permitted 
  provided that the following conditions are met:
  *Redistributions of source code must retain the above copyright notice,this list of conditions 
   and the following disclaimer.
  *Redistributions in binary form must reproduce the above copyright notice,this list of conditions and 
   the following disclaimer in the documentation and/or other materials provided with the distribution.
  *Neither the names of the University of Texas at Austin,nor Austin Robot Technology,nor the names of 
   other contributors maybe used to endorse or promote products derived from this software without 
   specific prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGH THOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
  WARRANTIES,INCLUDING,BUT NOT LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
  PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR 
  ANY DIRECT,INDIRECT,INCIDENTAL,SPECIAL,EXEMPLARY,OR CONSEQUENTIAL DAMAGES(INCLUDING,BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;LOSS OF USE,DATA,OR PROFITS;OR BUSINESS INTERRUPTION)HOWEVER 
  CAUSED AND ON ANY THEORY OF LIABILITY,WHETHER IN CONTRACT,STRICT LIABILITY,OR TORT(INCLUDING NEGLIGENCE 
  OR OTHERWISE)ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF 
  SUCHDAMAGE.
************************************************************************************************/

/*
 * File:       ptc_parser.h
 * Author:     Zhang Xu <int_zhangxu@hesaitech.com>
 * Description: ptc parser class is an interface layer.
 */
#ifndef PTC_PARSER_H_
#define PTC_PARSER_H_
#include "general_ptc_parser.h"
#include "ptc_1_0_parser.h"
#include "ptc_2_0_parser.h"
#include "inner_com.h"
#include <memory>

namespace hesai
{
namespace lidar
{
// class PtcParser
// the PtcParser class is an interface layer. It instantiates a specific ptc parser class,
// which is determined by the version of ptc's protocol, 1.0 or 2.0.
// PtcParser mainly parsers ptc packets and get values the request want.
class PtcParser {
public:
  explicit PtcParser(uint8_t ptc_version);
  PtcParser() { parser_ = nullptr; ptc_version_ = 0; };
  virtual ~PtcParser();
  PtcParser(const PtcParser&) = delete;
  PtcParser& operator=(const PtcParser&) = delete;
  // 字节流的打包。
  // 因为要将header和payload进行组装
  bool PtcStreamEncode(const u8Array_t &payload, u8Array_t &byteStreamOut, uint8_t u8Cmd);

  // 字节流的解析（拆包）
  bool PtcStreamDecode(uint8_t cmd, uint8_t retcode, const u8Array_t &payload, int start_pos, int length, u8Array_t &res);

  uint8_t GetHeaderIdentifier0() { return parser_->GetHeaderIdentifier0(); }
  uint8_t GetHeaderIdentifier1() { return parser_->GetHeaderIdentifier1(); }
  uint32_t GetPtcParserHeaderSize() { return parser_->GetHeaderSize(); }
  uint8_t GetHeaderReturnCode() { return parser_->GetHeaderReturnCode(); }
  uint8_t GetHeaderCmd() { return parser_->GetHeaderCmd(); }
  uint32_t GetHeaderPayloadLen() { return parser_->GetHeaderPayloadLen(); }

  bool SplitFileFrames(const u8Array_t &file, uint8_t u8Cmd, std::vector<u8Array_t>& packages);

private:
  GeneralPtcParser* parser_;
  uint8_t ptc_version_;
};
}
}
#endif