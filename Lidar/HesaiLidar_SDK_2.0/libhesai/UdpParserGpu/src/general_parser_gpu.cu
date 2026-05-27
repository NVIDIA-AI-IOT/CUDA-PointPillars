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
#ifndef GENERAL_PARSER_GPU_CU_
#define GENERAL_PARSER_GPU_CU_
#include "general_parser_gpu.h"

using namespace hesai::lidar;
template <typename T_Point>
GeneralParserGpu<T_Point>::GeneralParserGpu(uint16_t maxPacket, uint16_t maxPoint) {
  cudaSafeMalloc(correction_azi_cu_, sizeof(double) * DEFAULT_MAX_LASER_NUM);
  cudaSafeMalloc(correction_ele_cu_, sizeof(double) * DEFAULT_MAX_LASER_NUM);
  cudaSafeMalloc(firetimes_cu_, sizeof(double) * DEFAULT_MAX_LASER_NUM);
  if (maxPacket > 0 && maxPoint > 0) {
    cudaSafeMalloc(point_data_cu_, sizeof(PointDecodeData) * maxPacket * maxPoint);
    cudaSafeMalloc(packet_data_cu_, sizeof(PacketDecodeData) * maxPacket);
    cudaSafeMalloc(points_cu_, sizeof(LidarPointXYZDAE) * maxPacket * maxPoint);
    cudaSafeMalloc(valid_points_cu_, sizeof(uint32_t) * maxPacket);
    points_ = new LidarPointXYZDAE[maxPacket * maxPoint];
  } else {
    point_data_cu_ = nullptr;
    packet_data_cu_ = nullptr;
    points_cu_ = nullptr;
    valid_points_cu_ = nullptr;
    points_ = nullptr;
  }
}
template <typename T_Point>
GeneralParserGpu<T_Point>::~GeneralParserGpu() {
  cudaSafeFree(correction_azi_cu_);
  cudaSafeFree(correction_ele_cu_);
  cudaSafeFree(firetimes_cu_);
  if (point_data_cu_ != nullptr) cudaSafeFree(point_data_cu_);
  if (packet_data_cu_ != nullptr) cudaSafeFree(packet_data_cu_);
  if (points_cu_ != nullptr) cudaSafeFree(points_cu_);
  if (valid_points_cu_ != nullptr) cudaSafeFree(valid_points_cu_);
  if (points_ != nullptr) delete[] points_;
  if (point_cloud_cu_ != nullptr) cudaSafeFree(point_cloud_cu_);
}

template <typename T_Point>
void GeneralParserGpu<T_Point>::LoadCorrectionStruct(void* _correction) {
  correction_ptr = (CorrectionData*)_correction;
  CUDACheck(cudaMemcpy(correction_azi_cu_, correction_ptr->azimuth, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
  CUDACheck(cudaMemcpy(correction_ele_cu_, correction_ptr->elevation, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
}

template <typename T_Point>
void GeneralParserGpu<T_Point>::LoadFiretimesStruct(void* _firetimes) {
  firetimes_ptr = (float*)_firetimes;
  CUDACheck(cudaMemcpy(firetimes_cu_, firetimes_ptr, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
}

template <typename T_Point>
void GeneralParserGpu<T_Point>::updateCorrectionFile() {
  if (*get_correction_file_ && correction_load_sequence_num_cuda_use_ != *correction_load_sequence_num_) {
    correction_load_sequence_num_cuda_use_ = *correction_load_sequence_num_;
    CUDACheck(cudaMemcpy(correction_azi_cu_, correction_ptr->azimuth, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
    CUDACheck(cudaMemcpy(correction_ele_cu_, correction_ptr->elevation, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
  }
}

template <typename T_Point>
void GeneralParserGpu<T_Point>::updateFiretimeFile() {
  if (*get_firetime_file_ && firetime_load_sequence_num_cuda_use_ != *firetime_load_sequence_num_) {
    firetime_load_sequence_num_cuda_use_ = *firetime_load_sequence_num_;
    CUDACheck(cudaMemcpy(firetimes_cu_, firetimes_ptr, sizeof(double) * DEFAULT_MAX_LASER_NUM, cudaMemcpyHostToDevice));
  }
}

template <typename T_Point>
void GeneralParserGpu<T_Point>::DoRemake(float azi_, float elev_, const RemakeConfig &remake_config, int &point_idx) {
  if (remake_config.flag == false) return;
  point_idx = -1;
  elev_ = elev_ > 180.0 ? elev_ - 360.0 : elev_;
  int new_azi_iscan = (azi_ - remake_config.min_azi) / remake_config.ring_azi_resolution;
  int new_elev_iscan = (elev_ - remake_config.min_elev) / remake_config.ring_elev_resolution;
  if (new_azi_iscan >= 0 && new_azi_iscan < remake_config.max_azi_scan && new_elev_iscan >= 0 && new_elev_iscan < remake_config.max_elev_scan) {
    point_idx = new_azi_iscan * remake_config.max_elev_scan + new_elev_iscan;
  }
}


#endif //GENERAL_PARSER_GPU_CU_