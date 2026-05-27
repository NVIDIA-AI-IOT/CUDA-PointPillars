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
 * File:       ptc_2_0_parser.cc
 * Author:     Zhang Xu <int_zhangxu@hesaitech.com>
 */

#include "ptc_2_0_parser.h"
using namespace hesai::lidar;

// 字节流的打包。
// 因为要将header和payload进行组装
bool Ptc_2_0_parser::PtcStreamEncode(const u8Array_t &payload, u8Array_t &byteStreamOut, uint8_t u8Cmd) {
  byteStreamOut.resize(sizeof(PTCHeader_2_0) + payload.size()); //设置byteStreamOut长度
  PTCHeader_2_0 *pHeader = (PTCHeader_2_0 *)byteStreamOut.data();
  pHeader->init(u8Cmd);
  pHeader->SetPayloadLen(payload.size());
  pHeader++;

  memcpy((void *)pHeader, payload.data(), payload.size());
  return true;
}

// 字节流的解析（拆包）
bool Ptc_2_0_parser::PtcStreamDecode(uint8_t cmd, uint8_t retcode, const u8Array_t &payload, int start_pos, int length, u8Array_t &res) {
  (void)start_pos;
  (void)length;
  // PTCHeader_2_0 *header = dynamic_cast<PTCHeader_2_0*>(base_header);
  // if(header == nullptr) {
  //   std::cout << "Ptc_2_0_parser::PtcStreamDecode failed! header is not a 2.0 version!" << std::endl;
  //   return false;
  // }
  // uint8_t cmd = header->GetCmd();
  // uint8_t retcode = header->GetReturnCode();

  if(cmd == 0x01) {           // 读变量
    res = payload;
  } else if(cmd == 0x02) {    // 写变量
    if(payload.size() == 0) {
      return true;
    }
    res = payload; // 写值不成功时返回对应的变量ID
    return false;
  } else if(cmd == 0x03) {    // 读变量组
    res = payload;
  } else if(cmd == 0x04) {    // 写变量组
    if(payload.size() == 0) {
      return true;
    }
    res = payload; // 写值不成功时返回对应的变量ID
    return false;
  } else if(cmd == 0x05) {    // 读设备地址
    res = payload;
  } else if(cmd == 0x06) {    // 写设备地址
    if(payload.size() == 0) {
      return true;
    }
    res = payload; // 写值不成功时返回对应的变量ID
    return false;
  } else if(cmd == 0x07) {    // 读设备连续地址
    res = payload;
  } else if(cmd == 0x08) {    // 写设备连续地址
    if(payload.size() == 0) {
      return true;
    }
    res = payload; // 写值不成功时返回对应的变量ID
    return false;
  } else if(cmd == 0x09) {    // 读文件
    res = payload;
  } else if(cmd == 0x0A) {    // 写文件
    if(payload.size() == 0 && retcode == 0) {
      return true;
    }
    res.push_back(retcode);   // 写文件失败时用res记录return code的值
    return false;
  }
  std::cout << "Ptc_2_0_parser::PtcStreamDecode success!" << std::endl;
  return true;
}