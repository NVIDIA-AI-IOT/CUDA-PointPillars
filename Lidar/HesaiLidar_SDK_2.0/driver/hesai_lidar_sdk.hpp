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
#pragma once
#include "lidar.h"
#ifdef USE_CUDA
#include "udp_parser_gpu.h"
#endif
namespace hesai
{
namespace lidar
{

template <typename T_Point>
class HesaiLidarSdk 
{
private:
  std::thread *runing_thread_ptr_;
  std::function<void(const UdpFrame_t&, double)> pkt_cb_;
  std::function<void(const LidarDecodedFrame<T_Point>&)> point_cloud_cb_;
  std::function<void(const u8Array_t&)> correction_cb_;
  std::function<void(const uint32_t &, const uint32_t &)> pkt_loss_cb_;
  std::function<void(const uint8_t&, const u8Array_t&)> ptp_cb_;
  std::function<void(const FaultMessageInfo&)> fault_message_cb_;
  std::function<void(const LidarImuData&)> imu_cb_;
  bool is_thread_runing_;
  uint32_t device_ip_address_ = 0;
  uint16_t device_udp_port_ = 0;
  uint16_t device_fault_port_ = 0;
  LidarImuData last_imu_data_;
public:
  Lidar<T_Point> *lidar_ptr_;
#ifdef USE_CUDA
  UdpParserGpu<T_Point> gpu_parser_ptr_;
#endif
  DriverParam param_;
  std::thread *init_thread_ptr_;
  SourceType source_type_;
  HesaiLidarSdk() {
#ifdef USE_CUDA
    std::cout << "-------- Hesai Lidar SDK Gpu V" << 2 << "." << 0 << "." << 10 << " --------" << std::endl;
#else
    std::cout << "-------- Hesai Lidar SDK V" << 2 << "." << 0 << "." << 10 << " --------" << std::endl;
#endif
    runing_thread_ptr_ = nullptr;
    lidar_ptr_ = nullptr;
    init_thread_ptr_ = nullptr;
    is_thread_runing_ = false;
    source_type_ = DATA_FROM_PCAP;
  }

  ~HesaiLidarSdk() {
    Stop();
    LogInfo("Hesai Lidar SDK is stopping...");
  }

  void Init_thread()
  {
    //init lidar with param
    if (nullptr != lidar_ptr_) {
      lidar_ptr_->Init(param_);
      if (lidar_ptr_->GetInitFinish(FailInit)) return;
      LogDebug("finish 3: Initialisation all complete !!!");
#ifdef USE_CUDA
      if (lidar_ptr_->frame_.fParam.use_cuda == false) {
        LogWarning("-------- param setup not to use GPU !!! --------");
      }
      std::string lidar_type_from_parser = lidar_ptr_->GetLidarType();
      gpu_parser_ptr_.SetLidarType(lidar_type_from_parser, lidar_ptr_->frame_.maxPackerPerFrame, lidar_ptr_->frame_.maxPointPerPacket);
      gpu_parser_ptr_.setCorrectionLoadSequenceNum(lidar_ptr_->GetGeneralParser()->getCorrectionLoadSequenceNum());
      gpu_parser_ptr_.setFiretimeLoadSequenceNum(lidar_ptr_->GetGeneralParser()->getFiretimeLoadSequenceNum());
      gpu_parser_ptr_.setCorrectionLoadFlag(lidar_ptr_->GetGeneralParser()->getCorrectionLoadFlag());
      gpu_parser_ptr_.setFiretimeLoadFlag(lidar_ptr_->GetGeneralParser()->getFiretimeLoadFlag());
      gpu_parser_ptr_.LoadCorrectionStruct(lidar_ptr_->GetGeneralParser()->getStruct(CORRECTION_STRUCT));
      gpu_parser_ptr_.LoadFiretimesStruct(lidar_ptr_->GetGeneralParser()->getStruct(FIRETIME_STRUCT));
#else 
      lidar_ptr_->frame_.fParam.use_cuda = false;
      LogWarning("cpu version, not support for gpu");
#endif
      lidar_ptr_->SetAllFinishReady();
    }
  }

  //init lidar with param. init logger, udp parser, source, ptc client, start receive/parser thread
  bool Init(const DriverParam& param) 
  {
   /*****************************Init decoder******************************************************/ 
    lidar_ptr_ = new Lidar<T_Point>;
    if (nullptr == lidar_ptr_) {
      printf("create Lidar fail\n");
      return false;
    }
    source_type_ = param.input_param.source_type;
    param_ = param;
    init_thread_ptr_ = new std::thread(std::bind(&HesaiLidarSdk::Init_thread, this));
    if (param.input_param.device_ip_address != "") {
      device_ip_address_ = inet_addr(param.input_param.device_ip_address.c_str());
      if(param.input_param.device_fault_port >= 1024 && param.input_param.device_fault_port <= 65535 && device_ip_address_ != INADDR_NONE){
        device_fault_port_ = param.input_param.device_fault_port;
      }
      else{
        device_fault_port_ = 0;
      }
      if(param.input_param.device_udp_src_port >= 1024 && param.input_param.device_udp_src_port <= 65535 && device_ip_address_ != INADDR_NONE){
        device_udp_port_ = param.input_param.device_udp_src_port;
      }
      else{
        device_udp_port_ = 0;
      }
    }
    /***********************************************************************************/ 
    return true;
  }

  // stop process thread
  void Stop() {
    if (nullptr != runing_thread_ptr_) {
      is_thread_runing_ = false;
      runing_thread_ptr_->join();
      delete runing_thread_ptr_;
      runing_thread_ptr_ = nullptr;
    }

    if (nullptr != lidar_ptr_) {
      delete lidar_ptr_;
      lidar_ptr_ = nullptr;
    } 

    if (nullptr != init_thread_ptr_) {
      init_thread_ptr_->join();
      delete init_thread_ptr_;
      init_thread_ptr_ = nullptr;
    } 
  }

  void onRelease() { is_thread_runing_ = false; }

  // start process thread
  void Start() {
    while(1) {
      if (lidar_ptr_ == nullptr) { 
        printf("lidar_ptr_ is nullptr, start failed!!!!!!!!!\n");
        return;
      }
      if ((source_type_ == DATA_FROM_LIDAR && lidar_ptr_->GetInitFinish(FaultMessParse)) || lidar_ptr_->GetInitFinish(AllInitFinish)) {
        runing_thread_ptr_ = new std::thread(std::bind(&HesaiLidarSdk::Run, this));
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (lidar_ptr_->GetInitFinish(FailInit)) break;
    }
  }


  // process thread
  void Run()
  {
    LogInfo("--------begin to parse udp package--------");
    is_thread_runing_ = true;
    UdpFrame_t udp_packet_frame;
    uint32_t packet_index = 0;
    // uint32_t start = GetMicroTickCount();
    UdpPacket packet;
    FaultMessageInfo fault_message_info;

    uint32_t total_packet_count = 0;
    uint32_t total_packet_loss_count = 0;
    while (is_thread_runing_)
    {

      //get one packte from origin_packets_buffer_, which receive data from upd or pcap thread
      int ret = lidar_ptr_->GetOnePacket(packet);
      if (ret == -1) continue;
      //get fault message
      if (packet.packet_len == kFaultMessageLength 
          || (packet.buffer[0] == 0xCD && packet.buffer[1] == 0xDC)
          || (packet.buffer[SOMEIP_OFFSET] == 0xCD && packet.buffer[SOMEIP_OFFSET + 1] == 0xDC)) {
        if(device_fault_port_ != 0){
          if(device_ip_address_ != packet.ip || device_fault_port_ != packet.port){
            continue;
          }
        }
        ret = lidar_ptr_->ParserFaultMessage(packet, fault_message_info);
        if (ret == 0) {
          if (fault_message_cb_)
            fault_message_cb_(fault_message_info);
        }
        continue;
      }

      // Wait for initialization to complete before starting to parse the point cloud
      if(!lidar_ptr_->GetInitFinish(AllInitFinish)) continue;

      if(device_udp_port_ != 0 && (packet.is_timeout == false || packet_index == 0)){
        if(device_ip_address_ != packet.ip || device_udp_port_ != packet.port) {
          continue;
        }
      }
      //get distance azimuth reflection, etc.and put them into decode_packet
      ret = lidar_ptr_->DecodePacket(lidar_ptr_->frame_, packet);
      if (lidar_ptr_->frame_.imu_config.flag) {
        if (imu_cb_ && !last_imu_data_.isSameImuValue(lidar_ptr_->frame_.imu_config)) {
          imu_cb_(lidar_ptr_->frame_.imu_config);
          last_imu_data_ = lidar_ptr_->frame_.imu_config;
        }
      }
      if(ret != 0) {
        continue;
      }
      

      //one frame is receive completely, split frame
      if(lidar_ptr_->frame_.scan_complete) {
        if (lidar_ptr_->frame_.fParam.use_cuda) {
#ifdef USE_CUDA
          if (lidar_ptr_->frame_.packet_num > 0) {
            ret = gpu_parser_ptr_.ComputeXYZI(lidar_ptr_->frame_);  
          }
          else {
            ret = -1;
            LogError("packet_num is 0, cuda compute xyz error");
          }
#endif
        } else {
          //waiting for parser thread compute xyzi of points in the same frame
          while(!lidar_ptr_->ComputeXYZIComplete(packet_index)) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (lidar_ptr_->frame_.fParam.remake_config.flag) {
          auto& rq = lidar_ptr_->frame_.fParam.remake_config;
          lidar_ptr_->frame_.points_num = rq.max_azi_scan * rq.max_elev_scan;
        }
        if (lidar_ptr_->frame_.points_num == 0) {
          uint32_t points_num = 0;
          for (uint32_t i = 0; i < lidar_ptr_->frame_.packet_num; i++) {
            if (points_num != i * lidar_ptr_->frame_.per_points_num) {
              memcpy(lidar_ptr_->frame_.points + points_num, lidar_ptr_->frame_.points + 
                      i * lidar_ptr_->frame_.per_points_num, lidar_ptr_->frame_.getPointSize() * lidar_ptr_->frame_.valid_points[i]);
            }
            points_num += lidar_ptr_->frame_.valid_points[i];
          }
          lidar_ptr_->frame_.points_num = points_num;
        }
        //log info, display frame message
        if (ret == 0 && lidar_ptr_->frame_.points_num > kMinPointsOfOneFrame) {
          //publish point cloud topic
          if(point_cloud_cb_) point_cloud_cb_(lidar_ptr_->frame_);

          //publish upd packet topic
          if(pkt_cb_) {
            pkt_cb_(udp_packet_frame, lidar_ptr_->frame_.frame_start_timestamp);
          }

          if (pkt_loss_cb_ )
          {
            total_packet_count = lidar_ptr_->GetGeneralParser()->seqnum_loss_message_.total_packet_count;
            total_packet_loss_count = lidar_ptr_->GetGeneralParser()->seqnum_loss_message_.total_loss_count;
            pkt_loss_cb_(total_packet_count, total_packet_loss_count);
          }
          if (ptp_cb_ && lidar_ptr_->frame_.frame_index % 100 == 1)
          {
            u8Array_t ptp_status;
            u8Array_t ptp_lock_offset;
            int ret_status = lidar_ptr_->ptc_client_->GetPTPDiagnostics(ptp_status, 1); // ptp_query_type = 1
            int ret_offset = lidar_ptr_->ptc_client_->GetPTPLockOffset(ptp_lock_offset);
            if (ret_status != 0 || ret_offset != 0)
            {
              LogInfo("-->%d %d %zu %zu", ret_status, ret_offset, ptp_status.size(), ptp_lock_offset.size());
            }
            else
            {
              ptp_cb_(ptp_lock_offset.front(), ptp_status);
            }
          }
          if (correction_cb_ && lidar_ptr_->frame_.frame_index % 1000 == 1)
          {
            correction_cb_(lidar_ptr_->correction_string_);
          }
        } 
        //reset frame variable
        lidar_ptr_->frame_.Update();
        if (lidar_ptr_->frame_.fParam.update_function_safety_flag) {
          lidar_ptr_->frame_.clearFuncSafety();
        }
        if (lidar_ptr_->frame_.fParam.remake_config.flag) {
          memset(lidar_ptr_->frame_.points, 0, lidar_ptr_->frame_.getPointSize() * lidar_ptr_->frame_.maxPackerPerFrame * lidar_ptr_->frame_.maxPointPerPacket);
        }
        //clear udp packet vector
        udp_packet_frame.clear();
        packet_index = 0;

        //if the packet which contains split frame msgs is valid, it will be the first packet of new frame
        if(packet.is_timeout == false) {
          lidar_ptr_->DecodePacket(lidar_ptr_->frame_, packet);
          if (lidar_ptr_->frame_.fParam.use_cuda == false) {
            lidar_ptr_->ComputeXYZI(packet_index);
          }
          udp_packet_frame.emplace_back(packet);
          packet_index++;
        }
      }
      else {
        //new decoded packet of one frame, put it into decoded_packets_buffer_ and compute xyzi of points
        if(lidar_ptr_->frame_.packet_num != packet_index) {
          if (lidar_ptr_->frame_.fParam.use_cuda == false) {
            lidar_ptr_->ComputeXYZI(packet_index);
          }
          udp_packet_frame.emplace_back(packet);
          packet_index++;
        } 

        //update status manually if start a new frame failedly
        if (packet_index >= lidar_ptr_->frame_.maxPackerPerFrame) {
          LogError("fail to start a new frame, packet_index: %d", packet_index);
          packet_index = 0;
          udp_packet_frame.clear();
          lidar_ptr_->ClearPacketBuffer();
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          lidar_ptr_->frame_.Update();
          lidar_ptr_->GetUdpParser()->SetComputePacketNumToZero();
        }
      }
    }
  } 

  bool OriginPacketIsBufferFull() {
    if (lidar_ptr_ == nullptr) {
      return true;
    }
    return lidar_ptr_->origin_packets_buffer_.full();
  }

  int PushOriginPacketBuffer(const uint8_t *data, int len, uint64_t recv_timestamp = 0) {
    if (lidar_ptr_ == nullptr) {
      return -1;
    }
    if (lidar_ptr_->origin_packets_buffer_.full()) {
      return -2;
    }
    UdpPacket packet(data, len, recv_timestamp);
    lidar_ptr_->origin_packets_buffer_.emplace_back(packet);
    return 0;
  }

  bool ClearOriginPacketBuffer() {
    if (lidar_ptr_ == nullptr) {
      return false;
    }
    lidar_ptr_->origin_packets_buffer_.eff_clear();
    return true;
  }

  bool PopFrontOriginPacketBuffer(int num) {
    if (lidar_ptr_ == nullptr) {
      return false;
    }
    for (int i = 0; i < num; i++) {
      lidar_ptr_->origin_packets_buffer_.eff_pop_front();
    }
    return true;
  }

  // assign callback fuction
  void RegRecvCallback(const std::function<void(const LidarDecodedFrame<T_Point>&)>& callback) {
    point_cloud_cb_ = callback;
  }

  // assign callback fuction
  void RegRecvCallback(const std::function<void(const UdpFrame_t&, double)>& callback) {
    pkt_cb_ = callback;
  }

  // assign callback fuction
  void RegRecvCallback(const std::function<void (const u8Array_t&)>& callback) {
    correction_cb_ = callback;
  }

  void RegRecvCallback(const std::function<void (const uint32_t &, const uint32_t &)>& callback) {
    pkt_loss_cb_ = callback;
  }
  void RegRecvCallback(const std::function<void (const uint8_t&, const u8Array_t&)>& callback) {
    ptp_cb_ = callback;
  }
  void RegRecvCallback(const std::function<void (const FaultMessageInfo&)>& callback) {
    fault_message_cb_ = callback;
  }
  void RegRecvCallback(const std::function<void (const LidarImuData&)>& callback) {
    imu_cb_ = callback;
  }
};

}  // namespace lidar
}  // namespace hesai
