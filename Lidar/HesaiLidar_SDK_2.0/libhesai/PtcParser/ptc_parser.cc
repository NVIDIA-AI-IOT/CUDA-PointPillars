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
 * File:       ptc_parser.cc
 * Author:     Zhang Xu <int_zhangxu@hesaitech.com>
 * Description: ptc parser class is an interface layer.
 */
#include "ptc_parser.h"

using namespace hesai::lidar;

PtcParser::~PtcParser() {
  delete parser_;
  parser_ = nullptr;
};

PtcParser::PtcParser(uint8_t ptc_version) : ptc_version_(ptc_version) {
  if(ptc_version == 1) {
    // parser_ = std::dynamic_pointer_cast<Ptc_1_0_parser>(parser_);
    parser_ = new Ptc_1_0_parser();
  } else if(ptc_version == 2) {
    // parser_ = std::dynamic_pointer_cast<Ptc_2_0_parser>(parser_);
    parser_ = new Ptc_2_0_parser();
  }
}

// 字节流的打包。
// 因为要将header和payload进行组装
bool PtcParser::PtcStreamEncode(const u8Array_t &payload, u8Array_t &byteStreamOut, uint8_t u8Cmd) {
  if(parser_ != nullptr) {
    return parser_->PtcStreamEncode(payload, byteStreamOut, u8Cmd);
  }
  return false;
}

// 字节流的解析（拆包）
bool PtcParser::PtcStreamDecode(uint8_t cmd, uint8_t retcode, const u8Array_t &payload, int start_pos, int length, u8Array_t &res) {
  if(parser_ != nullptr) {
    return parser_->PtcStreamDecode(cmd, retcode, payload, start_pos, length, res);
  }
  return false;
}

bool PtcParser::SplitFileFrames(const u8Array_t &file, uint8_t u8Cmd, std::vector<u8Array_t>& packages) {
  if(parser_ != nullptr) {
    return parser_->SplitFileFrames(file, u8Cmd, packages);
  }
  return false;
}