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
 * File:       general_parser.cc
 * Author:     Zhang Xu <int_zhangxu@hesaitech.com>
 */

#include "general_ptc_parser.h"
#include "ptc_client.h"
using namespace hesai::lidar;

GeneralPtcParser::~GeneralPtcParser(){}

// 对文件数据划分成包，并且对每一个包进行数据帧的封装
// 对需要分成多个包的payload进行split和封装成包
bool GeneralPtcParser::SplitFileFrames(const u8Array_t &file, uint8_t u8Cmd, std::vector<u8Array_t>& packages) {
  // const int FRAME_LENGTH = 1024; 
  // int file_length = file.size();
  std::vector<u8Array_t> frames;
  // split file -> frames
  // int pos = 0;
  // while(file_length > 0) {
    // u8Array_t tmp = u8Array_t(pos, pos + min(FRAME_LENGTH, file_length));
    // frames.push_back(tmp);
    // pos += FRAME_LENGTH;
    // file_length -= FRAME_LENGTH;
  // }
  // 对frame进行封装成包
  for(auto &frame : frames) {
    u8Array_t cur;
    bool f = PtcStreamEncode(frame, cur, u8Cmd);
    if(!f) {
      std::cout << "GeneralPtcParser::PtcFilestreamSend pack frame failed!" << std::endl;
      return false;
    }
    packages.push_back(cur);
  }
  return true;
}