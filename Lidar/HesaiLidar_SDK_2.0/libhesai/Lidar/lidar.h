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
 * File:       lidar.h
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Define Lidar class 
 */

#ifndef Lidar_H
#define Lidar_H
#include <time.h>
#include "lidar_types.h"
#include "ptc_client.h"
#include "tcp_client.h"
#include "udp_parser.h"
#include "pcap_source.h"
#include "socket_source.h"
#include "blocking_ring.h"
#include "ring.h"
#include "driver_param.h"
#include "serial_source.h"
#include "serial_client.h"
#include "pcap_saver.h"
#ifndef _MSC_VER
#include <endian.h>
#include <semaphore.h>
#endif
#define AT128E2X_PACKET_LEN (1180)
#define GPS_PACKET_LEN (512)
#define CMD_SET_STANDBY_MODE (0x1c)
#define CMD_SET_SPIN_SPEED (0x17)

namespace hesai
{
namespace lidar
{

enum LidarInitFinishStatus {
  FaultMessParse = 0,
  PtcInitFinish = 1,
  PointCloudParse = 2,
  AllInitFinish = 3,
  FailInit = 4,

  TotalStatus
};

// class Lidar
// the Lidar class will start a udp data Receive thread and parser thread when init()
// udp packet data will be recived in udp data Receive thread and put into origin_packets_buffer_
// you can get origin udp packet data from origin_packets_buffer_ when you need
// you can put decoded packet into decoded_packets_buffer_ , parser thread will convert it pointsxyzi info in variable frame_
// you can control lidar with variable ptc_client_
template <typename T_Point>
class Lidar
{
public:
  ~Lidar();
  Lidar();
  Lidar(const Lidar &orig) = delete;
  Lidar& operator=(const Lidar&) = delete;
  // init lidar with param. init logger, udp parser, source, ptc client, start receive/parser thread
  int Init(const DriverParam& param);
  void InitSetPtc(const DriverParam param);
  // set lidar type for Lidar object and udp parser, udp parser will initialize the corresponding subclass
  int SetLidarType(std::string lidar_type);
  // get one packet from origin_packets_buffer_, 
  int GetOnePacket(UdpPacket &packet);
  // start to record pcap file with name record_path,the record source is udp data Receive from source
  int StartRecordPcap(const std::string& record_path);
  // stop record pcap
  int StopRecordPcap();
  // control record pcap
  void EnableRecordPcap(bool bRecord);
  // start to record pcap file with name record_path,the record source is the parameter UdpFrame_t
  // port is the udp port recorded in pcap
  int SaveUdpPacket(const std::string &record_path, const UdpFrame_t &packets,
                    int port = 2368);
  // start to record pcap file with name record_path,the record source is the parameter UdpFrameArray_t
  // port is the udp port recorded in pcap                   
  int SaveUdpPacket(const std::string &record_path,
                    const UdpFrameArray_t &packets, int port = 2368);
  /* this founction whill put a int into decoded_packets_buffer_, and then the parser thread will 
  convert decoded packet dato into pointxyzi info*/
  int ComputeXYZI(int packet_index);
  // covert a origin udp packet to decoded packet, the decode function is in UdpParser module
  // covert a origin udp packet to decoded data, and pass the decoded data to a frame struct to reduce memory copy
  int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udp_packet);
  // Determine whether all pointxyzi info is parsed in this frame
  bool ComputeXYZIComplete(uint32_t index);
  // parse the detailed content of the fault message message
  int ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info);
  // save lidar correction file from ptc
  int SaveCorrectionFile(const std::string& correction_save_path);
  // get lidar correction file from ptc,and pass to udp parser
  int LoadCorrectionForUdpParser();
  // load firetimes file from ptc
  int LoadFiretimesForUdpParser();
  int LoadCorrectionForSerialParser(std::string correction_save_path);
  // load correction file from Ros bag 
  int LoadCorrectionFromROSbag();
  // get lidar correction file from local file,and pass to udp parser 
  void LoadCorrectionFile(const std::string& correction_path); 
  int LoadCorrectionString(const char *correction_string, int len);
  // get lidar firetime correction file from local file,and pass to udp parser     
  void LoadFiretimesFile(const std::string& firetimes_path);
  // set the parser thread number
  void SetThreadNum(int thread_num);
  void ClearPacketBuffer();
  // get pcap status
  bool IsPlayEnded();
  // get init_finish state
  bool GetInitFinish(LidarInitFinishStatus type) { return init_finish_[type]; }
  void SetAllFinishReady() { init_finish_[AllInitFinish] = true; }
  
  void SetSource(Source **source);
  UdpParser<T_Point> *GetUdpParser();
  GeneralParser<T_Point> *GetGeneralParser();
  std::string GetLidarType();

  std::shared_ptr<PtcClient> ptc_client_;
  LidarDecodedFrame<T_Point> frame_;
  u8Array_t correction_string_;
  BlockingRing<UdpPacket, kPacketBufferSize> origin_packets_buffer_;

private:
  bool init_finish_[TotalStatus];           // 0: 基本初始化完成， 1：ptc初始化完成， 2：角度校准文件加载完成， 3：全部初始化完成
  std::shared_ptr<UdpParser<T_Point>> udp_parser_;
  std::shared_ptr<PcapSaver> pcap_saver_;
  std::shared_ptr<Source> source_;
  std::shared_ptr<Source> source_fault_message_;
  std::shared_ptr<Source> source_rs232_;
  std::shared_ptr<SerialClient> serial_client_;
  BlockingRing<int, kPacketBufferSize> decoded_packets_buffer_;
  uint16_t ptc_port_;
  uint16_t udp_port_;
  uint16_t fault_message_port_;
  std::string device_ip_address_;
  bool source_fault_message_waiting_;
  // Receive udp data thread, this function will Receive udp data in while(),exit when variable running_ is false
  void ReceiveUdpThread();
  void ReceiveUdpThreadFaultMessage();
  /* parser decoded packet data thread. when thread num <  2, this function will parser decoded packet data in this thread
  when thread num >= 2,decoded packet data will be put in handle_thread_packet_buffer_, and the decoded packet will be parered in handle_thread*/
  void ParserThread();
  /* this function will parser decoded packet data  in handle_thread_packet_buffer_ into pointxyzi info  */
  void HandleThread(int thread_num);
  // this variable is the exit condition of udp/parser thread
  bool running_;
  // this variable decide whether Receive_packet_thread will run
  bool udp_thread_running_;
  // this variable decide whether parser_thread will run
  bool parser_thread_running_;
  bool init_running;
  std::shared_ptr<std::thread> Receive_packet_thread_ptr_;
  std::shared_ptr<std::thread> Receive_packet_thread_ptr_fault_message_;
  std::shared_ptr<std::thread> parser_thread_ptr_;
  std::shared_ptr<std::thread> init_set_ptc_ptr_;
  std::mutex *mutex_list_;
  std::vector<std::list<int>> handle_thread_packet_buffer_;
  std::vector<std::thread *> handle_thread_vec_;
  std::condition_variable* condition_vars_;
  uint32_t handle_buffer_size_;
  int handle_thread_count_;
  bool is_record_pcap_;
  bool is_timeout_ = false;

};
}  // namespace lidar
}  // namespace hesai

#include "lidar.cc"
#endif /* Lidar_H */


