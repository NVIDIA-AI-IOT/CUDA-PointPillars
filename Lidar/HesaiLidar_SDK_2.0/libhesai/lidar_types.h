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
 * File:       lidar_type.h
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: 
 */

#ifndef LIDAR_TYPES_H_
#define LIDAR_TYPES_H_
#include <stdint.h>
#include <vector>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <atomic>
#include <array>
#include "inner_com.h"
#include "driver_param.h"
namespace hesai
{
namespace lidar
{

#pragma pack(push, 1)
struct LidarPointXYZI
{
    float x; 
    float y;             
    float z;             
    uint8_t intensity;     
};

struct LidarPointXYZIRT
{
    float x; 
    float y;             
    float z;             
    uint8_t intensity;  
    uint16_t ring;
    double timestamp;  
};

struct LidarPointXYZICRT
{
    float x; 
    float y;             
    float z;             
    uint8_t intensity;  
    uint8_t confidence;  
    uint16_t ring;
    double timestamp;  
};

struct LidarPointXYZICRTT
{
    float x; 
    float y;             
    float z;             
    uint8_t intensity;  
    uint8_t confidence;  
    uint16_t ring;
    uint64_t timeSecond;  
    uint32_t timeNanosecond;  
};

struct LidarPointXYZICWERT
{
    float x; 
    float y;             
    float z;             
    uint8_t intensity;  
    uint8_t confidence;  
    uint8_t weightFactor;
    uint8_t envLight;
    uint16_t ring;
    double timestamp;  
};

struct LidarPointRTHI
{
    int theta; 
    int phi;               
    int radius;            
    int intensity;         
};
#pragma pack(pop)

struct LidarImuData {
  double timestamp; 
  double imu_accel_x;
  double imu_accel_y;
  double imu_accel_z;
  double imu_ang_vel_x;
  double imu_ang_vel_y;
  double imu_ang_vel_z;
  bool flag;

  LidarImuData() {
    flag = false;
    timestamp = 0;
    imu_accel_x = 0;
    imu_accel_y = 0;
    imu_accel_z = 1;
    imu_ang_vel_x = -1;
    imu_ang_vel_y = -1;
    imu_ang_vel_z = -1;
  }

  bool isSameImuValue(const LidarImuData& other) const {
    return imu_accel_x == other.imu_accel_x &&
           imu_accel_y == other.imu_accel_y &&
           imu_accel_z == other.imu_accel_z &&
           imu_ang_vel_x == other.imu_ang_vel_x &&
           imu_ang_vel_y == other.imu_ang_vel_y &&
           imu_ang_vel_z == other.imu_ang_vel_z;
  }
};

struct FrameDecodeParam {
  bool pcap_time_synchronization; 
  bool firetimes_flag;
  bool distance_correction_flag;
  bool xt_spot_correction;
  bool update_function_safety_flag;
  bool enable_packet_loss_tool_;
  bool enable_packet_timeloss_tool_;
  bool packet_timeloss_tool_continue_;
  bool dirty_mapping_reflectance;
  uint8_t use_timestamp_type;  // 0: point cloud time, other: local time
  uint8_t echo_mode_filter;
  LidarDecodeConfig config;
  TransformParam transform;
  int rotation_flag;
  RemakeConfig remake_config;
  bool use_cuda;
  FrameDecodeParam() {
    use_timestamp_type = 0;
    pcap_time_synchronization = false;
    firetimes_flag = false;
    distance_correction_flag = false;
    xt_spot_correction = false;
    update_function_safety_flag = false;
    echo_mode_filter = 0;
    rotation_flag = 0;
    enable_packet_loss_tool_ = false;
    enable_packet_timeloss_tool_ = false;
    packet_timeloss_tool_continue_ = false;
    dirty_mapping_reflectance = false;
    use_cuda = false;
  }
  void UpdateRotation(int rotation) {
    if (abs(rotation_flag) == 10240) return;
    if (abs(rotation_flag) > 128 * 8) rotation_flag /= 128;
    rotation_flag += rotation;
  }
  void Init(const DriverParam& param) {
    use_timestamp_type = param.decoder_param.use_timestamp_type;
    pcap_time_synchronization = param.decoder_param.pcap_play_synchronization;
    firetimes_flag = true;
    distance_correction_flag = param.decoder_param.distance_correction_flag;
    xt_spot_correction = param.decoder_param.xt_spot_correction;
    config.fov_start = param.decoder_param.fov_start;
    config.fov_end = param.decoder_param.fov_end;
    transform = param.decoder_param.transform_param;
    enable_packet_loss_tool_ = param.decoder_param.enable_packet_loss_tool;
    enable_packet_timeloss_tool_ = param.decoder_param.enable_packet_timeloss_tool;
    packet_timeloss_tool_continue_ = param.decoder_param.packet_timeloss_tool_continue;
    remake_config = param.decoder_param.remake_config;
    use_cuda = param.use_gpu;
  }
};

template <typename PointT>
class LidarDecodedFrame
{
    public:
    LidarDecodedFrame(uint16_t maxPacketNum = 5000, uint16_t maxNumPerPacket = 1024) {
        resetMalloc(maxPacketNum, maxNumPerPacket);
        lidar_state = -1;
        work_mode = -1;
        packet_num = 0;
        frame_index = 0;
        host_timestamp = 0;
        block_num = 0;
        laser_num = 0; 
        channel_num = 0;
        per_points_num = 0;
        distance_unit = 0.0;
        return_mode = 0;
        points_num = 0;
        frame_start_timestamp = 0;
        frame_end_timestamp = 0;
        scan_complete = false;
    };
    ~LidarDecodedFrame() {
        if (total_memory) {
            delete[] total_memory;
            total_memory = nullptr;
            packetData = nullptr;
            points = nullptr;
            pointData = nullptr;
            funcSafety = nullptr;
        }
        if (valid_points) {
          delete[] valid_points;
          valid_points = nullptr;
        }
        if (point_cloud_raw_data) {
          delete[] point_cloud_raw_data;
          point_cloud_raw_data = nullptr;
        }
    }
    void resetMalloc(uint16_t maxPacketNum, uint16_t maxNumPerPacket) {
      maxPackerPerFrame = maxPacketNum;
      maxPointPerPacket = maxNumPerPacket;
      if (total_memory) {
          delete[] total_memory;
          total_memory = nullptr;
          packetData = nullptr;
          points = nullptr;
          pointData = nullptr;
          funcSafety = nullptr;
      }
      if (valid_points) {
        delete[] valid_points;
      }
      total_memory = new uint8_t[sizeof(PointDecodeData) * maxPackerPerFrame * maxPointPerPacket
                                   + sizeof(PacketDecodeData) * maxPackerPerFrame
                                   + sizeof(PointT) * maxPackerPerFrame * maxPointPerPacket
                                   + sizeof(FunctionSafety) * maxPackerPerFrame];
      memset(total_memory, 0, sizeof(PointDecodeData) * maxPackerPerFrame * maxPointPerPacket
                                   + sizeof(PacketDecodeData) * maxPackerPerFrame
                                   + sizeof(PointT) * maxPackerPerFrame * maxPointPerPacket
                                   + sizeof(FunctionSafety) * maxPackerPerFrame);
      uint32_t offset = 0;
      pointData = reinterpret_cast<PointDecodeData* >(total_memory + offset);
      offset += (sizeof(PointDecodeData) * maxPackerPerFrame * maxPointPerPacket);
      packetData = reinterpret_cast<PacketDecodeData* >(total_memory + offset);
      offset += (sizeof(PacketDecodeData) * maxPackerPerFrame);
      points = reinterpret_cast <PointT* >(total_memory + offset);
      offset += (sizeof(PointT) * maxPackerPerFrame * maxPointPerPacket);
      funcSafety = reinterpret_cast <FunctionSafety* >(total_memory + offset);
      offset += (sizeof(FunctionSafety) * maxPackerPerFrame);
      valid_points = new uint32_t[maxPackerPerFrame];
    }
    LidarDecodedFrame(const LidarDecodedFrame&) = delete;
    LidarDecodedFrame& operator=(const LidarDecodedFrame&) = delete;
    void Update() {
        packet_num = 0;
        points_num = 0;
        frame_start_timestamp = 0;
        frame_end_timestamp = 0;
        scan_complete = false;
        frame_index++;
    }
    void clearFuncSafety() {
      memset(funcSafety, 0, sizeof(FunctionSafety) * maxPackerPerFrame);
    }
    uint32_t getPointSize() const {
      return sizeof(PointT);
    }
    uint8_t* total_memory = nullptr; 
    uint16_t maxPackerPerFrame;
    uint16_t maxPointPerPacket;
    // configure
    FrameDecodeParam fParam;

    // frame parameter
    int16_t lidar_state;
    int16_t work_mode;
    uint16_t return_mode;
    uint32_t packet_num; 
    uint32_t* valid_points = nullptr;
    int frame_index;
    uint32_t points_num;
    double frame_start_timestamp;
    double frame_end_timestamp;
    // package parameter
    uint64_t host_timestamp;
    PacketDecodeData* packetData = nullptr; 
    FunctionSafety* funcSafety = nullptr;
    uint16_t block_num;
    uint16_t laser_num;  // channel number in point cloud
    uint16_t channel_num; // real channel number, when != 0, mean useful channel number
    uint32_t per_points_num; 
    double distance_unit;
    bool scan_complete;
    // point parameter
    PointDecodeData* pointData = nullptr;
    PointT* points = nullptr;
    //special output
    LidarImuData imu_config;
    // point cloud raw data
    bool frame_init_ = false;
    uint32_t point_cloud_size = 0;
    uint8_t* point_cloud_raw_data = nullptr;
};

}  // namespace lidar
}  // namespace hesai

 
#endif // LIDAR_TYPES_H_
