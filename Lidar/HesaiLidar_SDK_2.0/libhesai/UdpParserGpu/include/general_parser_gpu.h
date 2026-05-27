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
#ifndef GENERAL_PARSER_GPU_H_
#define GENERAL_PARSER_GPU_H_
#include <stdint.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <sstream>
#include <vector>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "lidar_types.h"
#include "plat_utils.h"
#include "general_parser.h"
#include "logger.h"
#include "udp_parser_gpu_kernel.h"
#include "general_struct_gpu.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "safe_call.cuh"
#include "return_code.h"
#define HALF_CIRCLE 180.0
#ifndef M_PI
#define M_PI 3.1415926535898
#endif

#define PUT_POINT_IN_POINT_INFO \
  auto &pointData = frame.pointData[j]; \
  auto &packetData = frame.packetData[i]; \
  int point_index_rerank = point_index + point_num; \
  float azi_ = this->points_[j].azimuthCalib / M_PI * HALF_CIRCLE; \
  float elev_ = this->points_[j].elevationCalib / M_PI * HALF_CIRCLE; \
  GeneralParserGpu<T_Point>::DoRemake(azi_, elev_, frame.fParam.remake_config, point_index_rerank); \
  if(point_index_rerank >= 0) { \
    auto& ptinfo = frame.points[point_index_rerank]; \
    set_x(ptinfo, this->points_[j].x); \
    set_y(ptinfo, this->points_[j].y); \
    set_z(ptinfo, this->points_[j].z); \
    set_ring(ptinfo, pointData.channel_index); \
    set_intensity(ptinfo, pointData.reflectivity); 


namespace hesai
{
namespace lidar
{

// class GeneralParserGpu
// the GeneralParserGpu class is a base class for computing points with gpu
// you can compute xyzi of points using the ComputeXYZI fuction, which uses gpu to compute
template <typename T_Point>
class GeneralParserGpu {
 public:
  GeneralParserGpu(uint16_t maxPacket, uint16_t maxPoint);
  virtual ~GeneralParserGpu();
  virtual void LoadCorrectionStruct(void *);
  virtual void LoadFiretimesStruct(void *);
  void setCorrectionLoadFlag(bool* flag) { get_correction_file_ = flag; }
  void setFiretimeLoadFlag(bool* flag) { get_firetime_file_ = flag; }
  void setCorrectionLoadSequenceNum(uint32_t* num) { correction_load_sequence_num_ = num; correction_load_sequence_num_cuda_use_ = *num; }
  void setFiretimeLoadSequenceNum(uint32_t* num) { firetime_load_sequence_num_ = num; firetime_load_sequence_num_cuda_use_ = *num; }
  virtual void updateCorrectionFile();
  virtual void updateFiretimeFile();
  // compute xyzi of points from decoded packet， use gpu device
  // param packet is the decoded packet; xyzi of points after computed is puted in frame  
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame) = 0;
  void DoRemake(float azi_, float elev_, const RemakeConfig &remake_config, int &point_idx);
 protected:
  bool* get_correction_file_;
  bool* get_firetime_file_;
  uint32_t* correction_load_sequence_num_;
  uint32_t* firetime_load_sequence_num_;
  uint32_t correction_load_sequence_num_cuda_use_ = 0;
  uint32_t firetime_load_sequence_num_cuda_use_ = 0;

  float* correction_azi_cu_;
  float* correction_ele_cu_;
  float* firetimes_cu_;
  PointDecodeData* point_data_cu_;
  PacketDecodeData* packet_data_cu_;
  LidarPointXYZDAE* points_;
  LidarPointXYZDAE* points_cu_;
  uint32_t* valid_points_cu_;
  const CorrectionData* correction_ptr;
  const float* firetimes_ptr;
  LidarOpticalCenter optical_center;
  uint8_t* point_cloud_cu_ = nullptr;
  // new cuda param
  uint8_t* input_data_cuda_ = nullptr;
  uint32_t point_cloud_size_ = 0;
};
}
}
#include "general_parser_gpu.cu"
#endif  // GENERAL_PARSER_GPU_H_
