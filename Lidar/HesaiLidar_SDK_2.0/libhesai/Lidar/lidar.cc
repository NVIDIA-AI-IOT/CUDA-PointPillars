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
#ifndef Lidar_CC
#define Lidar_CC
#include "lidar.h"
#include <inttypes.h>
#include <stdio.h>
#include <chrono>
using namespace hesai::lidar;

template <typename T_Point>
Lidar<T_Point>::Lidar() {
  udp_parser_ = std::make_shared<UdpParser<T_Point>>();
  pcap_saver_ = std::make_shared<PcapSaver>();
  is_record_pcap_ = false;
  running_ = true;
  init_running = true;
  udp_thread_running_ = true;
  parser_thread_running_ = true;
  handle_thread_count_ = 0;
  mutex_list_ = new std::mutex[GetAvailableCPUNum()];
  condition_vars_ = new std::condition_variable[GetAvailableCPUNum()];
  handle_buffer_size_ = kPacketBufferSize;
  std::fill(init_finish_, init_finish_ + TotalStatus, false);
  ptc_port_ = 0;
  udp_port_ = 0;
  source_fault_message_waiting_ = true;
}

template <typename T_Point>
Lidar<T_Point>::~Lidar() {
  running_ = false;
  init_running = false;
  udp_thread_running_ = false;
  parser_thread_running_ = false;
  if (Receive_packet_thread_ptr_) {
    Receive_packet_thread_ptr_->join();
    Receive_packet_thread_ptr_.reset();
  }

  if (Receive_packet_thread_ptr_fault_message_) {
    Receive_packet_thread_ptr_fault_message_->join();
    Receive_packet_thread_ptr_fault_message_.reset();
  }

  if (parser_thread_ptr_) {
    parser_thread_ptr_->join();
    parser_thread_ptr_.reset();
  }

  if (init_set_ptc_ptr_) {
    init_set_ptc_ptr_->join();
    init_set_ptc_ptr_.reset();
  }
  if (handle_thread_count_ > 1) {
    for (int i = 0; i < handle_thread_count_; i++) {
      condition_vars_[i].notify_all();
      if (handle_thread_vec_[i]) {
        handle_thread_vec_[i]->join();
        delete handle_thread_vec_[i];
        handle_thread_vec_[i] = nullptr;
      }
    }
  }

  if (mutex_list_ != nullptr) {
    delete[] mutex_list_;
    mutex_list_ = nullptr;
  }
  if (condition_vars_ != nullptr) {
    delete[] condition_vars_;
    condition_vars_ = nullptr;
  }
  Logger::GetInstance().Stop();
}

template <typename T_Point>
int Lidar<T_Point>::Init(const DriverParam& param) {
    init_running = true;
    int res = -1;
    /*******************************Init log*********************************************/
    Logger::GetInstance().SetFileName(param.log_path.c_str());
    Logger::GetInstance().setLogTargetRule(param.log_Target);
    Logger::GetInstance().setLogLevelRule(param.log_level);
    // Logger::GetInstance().bindLogCallback(logCallback);
    Logger::GetInstance().Start(); 
    /**********************************************************************************/
    /***************************Init source****************************************/
    if (param.input_param.source_type == DATA_FROM_LIDAR && param.input_param.use_someip) {
      auto source_tmp = std::make_shared<SocketSource>(SOMEIP_PORT);
      source_tmp->Open();
      auto ret = source_tmp->sendsomeip_subscribe_ipv4(param.input_param.device_ip_address, param.input_param.host_ip_address, param.input_param.udp_port, param.input_param.fault_message_port);
      if (ret == false) {
        LogFatal("send someip subscribe error");
        init_finish_[FailInit] = true;
        return -1;
      }
    }
    udp_port_ = param.input_param.udp_port;
    fault_message_port_ = param.input_param.fault_message_port;
    device_ip_address_ = param.input_param.device_ip_address;
    if (param.input_param.source_type == DATA_FROM_PCAP) {
      int packet_interval = 10;
      source_ = std::make_shared<PcapSource>(param.input_param.pcap_path, packet_interval);
      if(source_->Open() == false) {
        LogFatal("open pcap file error");
        init_finish_[FailInit] = true;
        return -1;
      }
      source_->SetPcapLoop(param.decoder_param.pcap_play_in_loop);
    }
    else if(param.input_param.source_type == DATA_FROM_LIDAR) {
      source_ = std::make_shared<SocketSource>(param.input_param.udp_port, param.input_param.multicast_ip_address);
      source_->Open();
      if (param.input_param.fault_message_port > 0 && param.input_param.fault_message_port != param.input_param.udp_port) {
        source_fault_message_ = std::make_shared<SocketSource>(param.input_param.fault_message_port, param.input_param.multicast_ip_address);
        source_fault_message_->Open();
      }
    }
    else if(param.input_param.source_type == DATA_FROM_SERIAL) {
      source_ = std::make_shared<SerialSource>(param.input_param.rs485_com, param.input_param.rs485_baudrate, param.input_param.point_cloud_baudrate);
      source_rs232_ = std::make_shared<SerialSource>(param.input_param.rs232_com, param.input_param.rs232_baudrate);
      source_->SetReceiveStype(SERIAL_POINT_CLOUD_RECV);
      source_->Open();
      if (!source_->IsOpened()) {
        LogFatal("recv point cloud serial open error");
        init_finish_[FailInit] = true;
        return -1;
      }
      source_rs232_->SetReceiveStype(SERIAL_COMMAND_RECV);
      source_rs232_->Open();
      if (!source_rs232_->IsOpened()) LogError("send cmd serial open error");
      serial_client_ = std::make_shared<SerialClient>();
      serial_client_->SetSerial(source_rs232_.get(), source_.get());
    }
    parser_thread_running_ = param.decoder_param.enable_parser_thread;
    udp_thread_running_ = param.decoder_param.enable_udp_thread;
    if (param.input_param.source_type == DATA_FROM_EXTERNAL_INPUT) {
      udp_thread_running_ = false;
    }
    if (param.decoder_param.socket_buffer_size > 0 && source_ != nullptr) {
      source_->SetSocketBufferSize(param.decoder_param.socket_buffer_size);
    }

    frame_.fParam.Init(param);
   
    SetThreadNum(param.decoder_param.thread_num);
    /********************************************************************************/
    if (param.input_param.use_ptc_connected) { 
      if (param.input_param.source_type == DATA_FROM_LIDAR) {
        ptc_client_ = std::make_shared<PtcClient>(param.input_param.device_ip_address
                                                    , param.input_param.ptc_port
                                                    , false
                                                    , param.input_param.ptc_mode
                                                    , 1
                                                    , param.input_param.certFile
                                                    , param.input_param.privateKeyFile
                                                    , param.input_param.caFile
                                                    , 2000
                                                    , 2000
                                                    , param.input_param.ptc_connect_timeout);
        init_set_ptc_ptr_ = std::make_shared<std::thread>(std::bind(&Lidar::InitSetPtc, this, param));
      }
    }
    UdpPacket udp_packet;
    auto init_start_time = std::chrono::high_resolution_clock::now();
    while (GetGeneralParser() == nullptr && init_running) {
      int ret = this->GetOnePacket(udp_packet);
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = end_time - init_start_time;
      if (param.input_param.recv_point_cloud_timeout >= 0 && 
           elapsed_seconds.count() > param.input_param.recv_point_cloud_timeout) break;
      if (ret == -1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      this->DecodePacket(frame_, udp_packet);
    }
    if (GetGeneralParser() == nullptr) {
      LogFatal("get cloud point fail, init exit");
      init_finish_[FailInit] = true;
      return res;
    }
    else {
      if (param.input_param.source_type == DATA_FROM_LIDAR && param.input_param.device_ip_address == "" && param.input_param.use_ptc_connected) {
        ptc_client_->SetLidarIP(udp_packet.ip);
      }
    }
    udp_parser_->setFrameRightMemorySpace(frame_);
    source_fault_message_waiting_ = false;
    init_finish_[FaultMessParse] = true;
    LogInfo("finish 0: The basic initialisation is complete");
    
    /***************************Init decoder****************************************/   
    udp_parser_->SetPcapPlay(param.input_param.source_type);
    udp_parser_->SetFrameAzimuth(param.decoder_param.frame_start_azimuth);
    switch (param.input_param.source_type)
    {
    case DATA_FROM_LIDAR: 
    {
      if (param.input_param.use_ptc_connected) { 
        auto ptc_start_time = std::chrono::high_resolution_clock::now();
        while (ptc_client_ != nullptr && (!ptc_client_->IsOpen()) && init_running) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          auto end_time = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed_seconds = end_time - ptc_start_time;
          if (param.input_param.ptc_connect_timeout >= 0 && 
                elapsed_seconds.count() > param.input_param.ptc_connect_timeout) {
            LogError("ptc connect timeout, get correction from local");
            break;
          }
        }
        if (LoadCorrectionForUdpParser() == -1) {
          LogError("---Failed to obtain correction file from lidar!---");
          LoadCorrectionFile(param.input_param.correction_file_path);
        }
        if (udp_parser_->GetLidarType() == "ATX") {
          if (LoadFiretimesForUdpParser() == -1) {
            LogWarning("---Failed to obtain firetimes file from lidar!---");
            LoadFiretimesFile(param.input_param.firetimes_path);
          }
        }
        else {
          LoadFiretimesFile(param.input_param.firetimes_path);
        }
      }
      else {
        LoadCorrectionFile(param.input_param.correction_file_path);
        LoadFiretimesFile(param.input_param.firetimes_path);
      }
    } break;
    case DATA_FROM_PCAP: {
      LoadCorrectionFile(param.input_param.correction_file_path);
      LoadFiretimesFile(param.input_param.firetimes_path);
    } break;
    case DATA_FROM_ROS_PACKET:
      LoadCorrectionFile(param.input_param.correction_file_path);
      LoadFiretimesFile(param.input_param.firetimes_path);
      break;
    case DATA_FROM_SERIAL:
      if (LoadCorrectionForSerialParser(param.input_param.correction_save_path) != 0) {
        LogError("---Failed to obtain correction file from lidar!---");
        LoadCorrectionFile(param.input_param.correction_file_path);
      }
      break;
    case DATA_FROM_EXTERNAL_INPUT:
      LoadCorrectionFile(param.input_param.correction_file_path);
      LoadFiretimesFile(param.input_param.firetimes_path);
    default:
      break;
    }
    init_finish_[PointCloudParse] = true;
    init_running = false;
    LogInfo("finish 2: The angle calibration file is finished loading");
    /********************************************************************************/
    res = 0;
    return res;
}

template <typename T_Point>
void Lidar<T_Point>::InitSetPtc(const DriverParam param) {
  while(init_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!ptc_client_ || !ptc_client_->IsOpen()) continue;
    init_finish_[PtcInitFinish] = true;
    LogDebug("finish 1: ptc connection successfully");
    if (param.input_param.use_someip) {
      ptc_client_->GetCorrectionInfo();
    }
    if (param.input_param.standby_mode != -1) {
      if (ptc_client_->SetStandbyMode(param.input_param.standby_mode)) {
        LogInfo("set standby mode successed!");
      } else {
        LogWarning("set standby mode failed!");
      }
    }
    if (param.input_param.speed != -1) {
      if (ptc_client_->SetSpinSpeed(param.input_param.speed)) {
        LogInfo("set speed successed!");
      } else {
        LogWarning("set speed failed!");
      }
    }
    break;
  }
}

template <typename T_Point>
int Lidar<T_Point>::SetLidarType(std::string lidar_type) {
  int ret = -1;
  if (udp_parser_) {
    udp_parser_->CreatGeneralParser(lidar_type);
    ret = 0;
  } else
    LogError("udp_parser_ nullptr");

  return ret;
}

template <typename T_Point>
int Lidar<T_Point>::GetOnePacket(UdpPacket &packet) {
  if (origin_packets_buffer_.try_pop_front(packet))  return 0;
  else  return -1;
}

template <typename T_Point>
int Lidar<T_Point>::StartRecordPcap(const std::string& record_path) {
  int ret = -1;
  pcap_saver_->SetPcapPath(record_path);
  EnableRecordPcap(true);
  pcap_saver_->Save();
  return ret;
}

template <typename T_Point>
int Lidar<T_Point>::StopRecordPcap() {
  EnableRecordPcap(false);
  pcap_saver_->close();
  return 0;
}

template <typename T_Point>
void Lidar<T_Point>::EnableRecordPcap(bool bRecord) {
  is_record_pcap_ = bRecord;
}

template <typename T_Point>
int Lidar<T_Point>::SaveUdpPacket(const std::string &record_path,
                              const UdpFrame_t &packets, int port) {
  return pcap_saver_->Save(record_path, packets, port);
}

template <typename T_Point>
int Lidar<T_Point>::SaveUdpPacket(const std::string &record_path,
                              const UdpFrameArray_t &packets, int port) {
  return pcap_saver_->Save(record_path, packets, port);
}

template <typename T_Point>
int Lidar<T_Point>::ComputeXYZI(int packet_index) {
  decoded_packets_buffer_.push_back(std::move(packet_index));
  return 0;
}

template <typename T_Point>
int Lidar<T_Point>::DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udp_packet) {
  if (udp_parser_) {
    return udp_parser_->DecodePacket(frame, udp_packet);
  } else
    LogError("udp_parser_ nullptr");

  return -1;
} 

template <typename T_Point>
bool Lidar<T_Point>::ComputeXYZIComplete(uint32_t index) {
  if (udp_parser_ == nullptr) return false;
  if (udp_parser_->GetComputePacketNum() == index) {
    udp_parser_->SetComputePacketNumToZero();
    return true;
  }
  return false;
}

template <typename T_Point>
int Lidar<T_Point>::ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info) {
  if (udp_parser_) {
    return udp_parser_->ParserFaultMessage(udp_packet, fault_message_info);
  } else
    LogError("udp_parser_ nullptr");

  return -1;
}

template <typename T_Point>
int Lidar<T_Point>::SaveCorrectionFile(const std::string& correction_save_path) {
  if (ptc_client_ == nullptr) return -1;
  int ret = -1;
  u8Array_t raw_data;
  if (ptc_client_->GetCorrectionInfo(raw_data) != 0) {
    LogError("SaveCorrectionFile get correction info fail");
    return ret;
  }
  correction_string_ = raw_data;
  std::ofstream out_file(correction_save_path, std::ios::out);
  if(out_file.is_open()) {
    out_file.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size());
    ret = 0;
    out_file.close();
    return ret;
  } else {
    LogError("create correction file fail");
    return ret;
  }
}

template <typename T_Point>
int Lidar<T_Point>::LoadCorrectionForUdpParser() {
  if (ptc_client_ == nullptr) return -1;
  u8Array_t sData;
  if (ptc_client_->GetCorrectionInfo(sData) != 0) {
    LogWarning("LoadCorrectionForUdpParser get correction info fail");
    return -1;
  }
  correction_string_ = sData;
  if (udp_parser_) {
    return udp_parser_->LoadCorrectionString(
        (char *)sData.data(), sData.size());
  } else {
    LogError("udp_parser_ nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
int Lidar<T_Point>::LoadFiretimesForUdpParser() {
  if (ptc_client_ == nullptr) return -1;
  u8Array_t sData;
  if (ptc_client_->GetFiretimesInfo(sData) != 0) {
    LogWarning("LoadFiretimesForUdpParser get firetimes info fail");
    return -1;
  }
  if (udp_parser_) {
    return udp_parser_->LoadFiretimesString(
        (char *)sData.data(), sData.size());
  } else {
    LogError("udp_parser_ nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
int Lidar<T_Point>::LoadCorrectionForSerialParser(std::string correction_save_path) {
  u8Array_t sData;
  int ret = serial_client_->GetCorrectionInfo(sData);
  if (ret != 0) {
    LogWarning("get correction error, ret:%d", ret);
    return ret;
  }

  // save
  if (correction_save_path != "") {
    std::ofstream out_file(correction_save_path, std::ios::out);
    if(out_file.is_open()) {
      out_file.write(reinterpret_cast<const char*>(sData.data()), sData.size());
      out_file.close();
    } else {
      LogError("create correction file fail, save correction file error");
    }
  }
  if (udp_parser_) {
    return udp_parser_->LoadCorrectionString(
        (char *)sData.data(), sData.size());
  } else {
    LogError("udp_parser_ nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
int Lidar<T_Point>::LoadCorrectionFromROSbag() {
  if (udp_parser_) {
    return udp_parser_->LoadCorrectionString(
        (char *)correction_string_.data(), correction_string_.size());
  } else {
    LogError("udp_parser_ nullptr");
    return -1;
  }
  return 0;
}

template <typename T_Point>
void Lidar<T_Point>::LoadCorrectionFile(const std::string& correction_path) {
  if (udp_parser_) {
    udp_parser_->LoadCorrectionFile(correction_path);
    return ;
  } else
    LogError("udp_parser_ nullptr");

  return ;
}

template <typename T_Point>
int Lidar<T_Point>::LoadCorrectionString(const char *correction_string, int len) {
  int ret = -1;
  if (udp_parser_) {
    return udp_parser_->LoadCorrectionString(correction_string, len);

  } else
    LogError("udp_parser_ nullptr");

  return ret;
}

template <typename T_Point>
void Lidar<T_Point>::LoadFiretimesFile(const std::string& firetimes_path) {
  if (udp_parser_) {
    udp_parser_->LoadFiretimesFile(firetimes_path);
    return ;
  } else
    LogError("udp_parser_ nullptr");

  return ;
}

template <typename T_Point>
void Lidar<T_Point>::ReceiveUdpThread() {
  if(!udp_thread_running_) return;
  // uint32_t u32StartTime = GetMicroTickCount();
  LogInfo("Lidar::Receive Udp Thread start to run");
#ifdef _MSC_VER
  SetThreadPriorityWin(THREAD_PRIORITY_TIME_CRITICAL);
#else
  SetThreadPriority(SCHED_FIFO, SHED_FIFO_PRIORITY_MEDIUM);
#endif
  while (running_) {
    if (!source_) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }
    UdpPacket udp_packet;
    int len = source_->Receive(udp_packet, kBufSize);
    if (len == -1) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }
    while(origin_packets_buffer_.full() && running_) std::this_thread::sleep_for(std::chrono::microseconds(1000));
    if(running_ == false) break;
    udp_packet.recv_timestamp = GetMicroTimeU64();
    switch (len) {
      case 0:
        if (is_timeout_ == false) {
          udp_packet.packet_len = AT128E2X_PACKET_LEN; 
          udp_packet.buffer[0] = 0;
          udp_packet.buffer[1] = 0;
          origin_packets_buffer_.emplace_back(udp_packet);
          is_timeout_ = true;
        } 
        break;
      case kFaultMessageLength: 
      {
        udp_packet.packet_len = static_cast<uint16_t>(len);
        origin_packets_buffer_.emplace_back(udp_packet);
      } break;
      case GPS_PACKET_LEN:
        break;
      default :
        if (len > 0) {
          udp_packet.packet_len = static_cast<uint16_t>(len);
          origin_packets_buffer_.emplace_back(udp_packet);
          if (!(udp_packet.buffer[0] == 0xCD && udp_packet.buffer[1] == 0xDC) 
              && !(udp_packet.buffer[SOMEIP_OFFSET] == 0xCD && udp_packet.buffer[SOMEIP_OFFSET + 1] == 0xDC))
            is_timeout_ = false;
        }
        break;
    }
    if (udp_packet.packet_len > 0 && is_record_pcap_) {
        pcap_saver_->Dump(udp_packet.buffer, udp_packet.packet_len, udp_port_);
    }
  }
  return;
}

template <typename T_Point>
void Lidar<T_Point>::ReceiveUdpThreadFaultMessage() {
  if(!udp_thread_running_) return;
  LogInfo("Lidar::Receive Udp Fault Message Thread start to run");
#ifdef _MSC_VER
  SetThreadPriorityWin(THREAD_PRIORITY_TIME_CRITICAL);
#else
  SetThreadPriority(SCHED_FIFO, SHED_FIFO_PRIORITY_MEDIUM);
#endif
  while (running_) {
    if (!source_fault_message_ || source_fault_message_waiting_) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }
    UdpPacket udp_packet_fault_message;
    int len = source_fault_message_->Receive(udp_packet_fault_message, kBufSize);
    if (len <= 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }
    if (len != kFaultMessageLength && 
        !(udp_packet_fault_message.buffer[0] == 0xCD && udp_packet_fault_message.buffer[1] == 0xDC) &&
        !(udp_packet_fault_message.buffer[SOMEIP_OFFSET] == 0xCD && udp_packet_fault_message.buffer[SOMEIP_OFFSET + 1] == 0xDC)) {
      continue;
    }
    udp_packet_fault_message.recv_timestamp = GetMicroTimeU64();
    udp_packet_fault_message.packet_len = static_cast<uint16_t>(len);
    while(origin_packets_buffer_.full() && running_) std::this_thread::sleep_for(std::chrono::microseconds(1000));
    if(running_ == false) break;
    origin_packets_buffer_.emplace_back(udp_packet_fault_message);
    if (is_record_pcap_) {
      pcap_saver_->Dump(udp_packet_fault_message.buffer, udp_packet_fault_message.packet_len, fault_message_port_);
    }
  }
  return;
}

template <typename T_Point>
void Lidar<T_Point>::ParserThread() {
  if(!parser_thread_running_) return;
  int nUDPCount = 0;
  LogInfo("Lidar::ParserThread start to run");
#ifdef _MSC_VER
  SetThreadPriorityWin(THREAD_PRIORITY_TIME_CRITICAL);
#else
  SetThreadPriority(SCHED_FIFO, SHED_FIFO_PRIORITY_MEDIUM);
#endif
  while (running_) {
    int decoded_packet_index;
    bool decoded_result = decoded_packets_buffer_.try_pop_front(decoded_packet_index);
    if (!decoded_result) {
      continue;
    }
    if (handle_thread_count_ < 2) {
      udp_parser_->ComputeXYZI(frame_, decoded_packet_index);
      continue;
    } else {
      nUDPCount = nUDPCount % handle_thread_count_;
      mutex_list_[nUDPCount].lock();
      handle_thread_packet_buffer_[nUDPCount].push_back(decoded_packet_index);

      if (handle_thread_packet_buffer_[nUDPCount].size() > handle_buffer_size_) {
        handle_thread_packet_buffer_[nUDPCount].pop_front();
      }
      mutex_list_[nUDPCount].unlock();
      condition_vars_[nUDPCount].notify_one();
      nUDPCount++;
    }
  }
  return;
}

template <typename T_Point>
void Lidar<T_Point>::HandleThread(int nThreadNum) {
  // struct timespec timeout;
#ifdef _MSC_VER
  SetThreadPriorityWin(THREAD_PRIORITY_TIME_CRITICAL);
#else
  SetThreadPriority(SCHED_FIFO, SHED_FIFO_PRIORITY_MEDIUM);
#endif
  if(!parser_thread_running_) return;
  std::unique_lock<std::mutex> lock(mutex_list_[nThreadNum]);
  while (running_) {
    condition_vars_[nThreadNum].wait(lock, [this, nThreadNum] {
        return !running_ || handle_thread_packet_buffer_[nThreadNum].size() > 0;
    });
    if (!running_) break;

    if (handle_thread_packet_buffer_[nThreadNum].size() > 0) {
      int decoded_packet_index = handle_thread_packet_buffer_[nThreadNum].front();
      handle_thread_packet_buffer_[nThreadNum].pop_front();
      lock.unlock();
      udp_parser_->ComputeXYZI(frame_, decoded_packet_index);
      lock.lock();
    }
  }
}

template <typename T_Point>
void Lidar<T_Point>::SetThreadNum(int nThreadNum) {
  if (nThreadNum > GetAvailableCPUNum() - 2) {
    nThreadNum = GetAvailableCPUNum() - 2;
    if (nThreadNum < 1) {
      nThreadNum = 1;
    }
    LogWarning("The number of threads is too large, set to the maximum number of threads %d.", nThreadNum);
  }
  LogInfo("Use thread num %d to parse packet", nThreadNum);

  if (handle_thread_count_ == nThreadNum) {
    return;
  }

  running_ = false;

  if (Receive_packet_thread_ptr_ != nullptr) {
    Receive_packet_thread_ptr_->join();
    Receive_packet_thread_ptr_.reset();
  }

  for (int i = 0; i < handle_thread_count_; i++) {
    if (handle_thread_vec_[i] != nullptr) {
      condition_vars_[i].notify_all();
      handle_thread_vec_[i]->join();
      delete handle_thread_vec_[i];
      handle_thread_vec_[i] = nullptr;
    }

    handle_thread_packet_buffer_[i].clear();
  }

  running_ = true;
  if (nThreadNum > 1) {
    handle_thread_vec_.resize(nThreadNum);
    handle_thread_packet_buffer_.resize(nThreadNum);

    for (int i = 0; i < nThreadNum; i++) {
      handle_thread_vec_[i] = new std::thread(
          std::bind(&Lidar::HandleThread, this, std::placeholders::_1), i);
    }
  }

  handle_thread_count_ = nThreadNum;
  Receive_packet_thread_ptr_ =
      std::make_shared<std::thread>(std::bind(&Lidar::ReceiveUdpThread, this));
  if (source_fault_message_) {
    Receive_packet_thread_ptr_fault_message_ =
      std::make_shared<std::thread>(std::bind(&Lidar::ReceiveUdpThreadFaultMessage, this));
  }
  parser_thread_ptr_ =
      std::make_shared<std::thread>(std::bind(&Lidar::ParserThread, this));    
}

template <typename T_Point>
void Lidar<T_Point>::ClearPacketBuffer() {
  decoded_packets_buffer_.eff_clear();
  if (handle_thread_count_ > 1) {
    for (int i = 0; i < handle_thread_count_; i++) {
      handle_thread_packet_buffer_[i].clear();
    }
  }
}

template <typename T_Point>
bool Lidar<T_Point>::IsPlayEnded()
{
  if (!source_)
  {
    return false;
  }
  return source_->is_pcap_end;
}

template <typename T_Point>
void Lidar<T_Point>::SetSource(Source **source) {
  source_.reset(*source);
}

template <typename T_Point>
UdpParser<T_Point> *Lidar<T_Point>::GetUdpParser() { return udp_parser_.get(); }

template <typename T_Point>
GeneralParser<T_Point> *Lidar<T_Point>::GetGeneralParser() {
  if (udp_parser_)
    return udp_parser_->GetGeneralParser();
  else
    LogError("udp_parser_ nullptr");
  return nullptr;
}

template <typename T_Point>
std::string Lidar<T_Point>::GetLidarType() {
  return udp_parser_->GetLidarType();
}



#endif