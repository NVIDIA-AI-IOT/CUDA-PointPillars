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
 * File:       general_parser.h
 * Author:     Zhang Xu <int_zhangxu@hesaitech.com>
 * Description: ptc protocol general parser class.
 */

#ifndef GENERAL_PTC_PARSER_H_
#define GENERAL_PTC_PARSER_H_
#include <iostream>
#include <fstream>
#include "inner_com.h"
#include "client_base.h"

namespace hesai
{
namespace lidar
{
class GeneralPtcParser {
public:
  GeneralPtcParser(){};
  virtual ~GeneralPtcParser();

  // 字节流的打包。
  // 因为要将header和payload进行组装，通用类中无法实现。
  virtual bool PtcStreamEncode(const u8Array_t &payload, u8Array_t &byteStreamOut, uint8_t u8Cmd) = 0;

  // 字节流的解析（拆包）
  virtual bool PtcStreamDecode(uint8_t cmd, uint8_t retcode, const u8Array_t &payload, int start_pos, int length, u8Array_t &res) = 0;

  // 对文件数据划分成包，并且对每一个包进行数据帧的封装
  virtual bool SplitFileFrames(const u8Array_t &file, uint8_t u8Cmd, std::vector<u8Array_t>& packages);

  virtual uint8_t GetHeaderIdentifier0() = 0;
  virtual uint8_t GetHeaderIdentifier1() = 0;
  virtual uint32_t GetHeaderSize() = 0;
  virtual uint8_t GetHeaderReturnCode() const = 0;
  virtual uint8_t GetHeaderCmd() const = 0;
  virtual uint32_t GetHeaderPayloadLen() const = 0;
};
}
}
#endif