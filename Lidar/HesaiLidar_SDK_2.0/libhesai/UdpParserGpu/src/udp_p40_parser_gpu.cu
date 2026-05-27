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
#ifndef Udp_P40_PARSER_GPU_CU_
#define Udp_P40_PARSER_GPU_CU_
#include "udp_p40_parser_gpu.h"

using namespace hesai::lidar;
template <typename T_Point>
UdpP40ParserGpu<T_Point>::UdpP40ParserGpu(uint16_t maxPacket, uint16_t maxPoint) : GeneralParserGpu<T_Point>(maxPacket, maxPoint) {
  this->optical_center.setNoFlag(LidarOpticalCenter{-0.012, 0.03873, 0});
}
template <typename T_Point>
UdpP40ParserGpu<T_Point>::~UdpP40ParserGpu() {
}

template <typename T_Point>
int UdpP40ParserGpu<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame) {
  if (!*this->get_correction_file_) return int(ReturnCode::CorrectionsUnloaded);       
  cudaSafeCall(cudaMemcpy(this->point_data_cu_, frame.pointData,
                          frame.per_points_num * frame.packet_num * sizeof(PointDecodeData), 
                          cudaMemcpyHostToDevice), ReturnCode::CudaMemcpyHostToDeviceError);
  cudaSafeCall(cudaMemcpy(this->packet_data_cu_, frame.packetData,
                          frame.packet_num * sizeof(PacketDecodeData), 
                          cudaMemcpyHostToDevice), ReturnCode::CudaMemcpyHostToDeviceError); 
  cudaSafeCall(cudaMemcpy(this->valid_points_cu_, frame.valid_points,
                          frame.packet_num * sizeof(uint32_t), 
                          cudaMemcpyHostToDevice), ReturnCode::CudaMemcpyHostToDeviceError); 
  this->updateCorrectionFile();
  this->updateFiretimeFile();
  FrameDecodeParam cuda_Param = frame.fParam;
  cuda_Param.firetimes_flag = *this->get_firetime_file_ ? cuda_Param.firetimes_flag : false;
  int ret = compute_p40_cuda(this->points_cu_, this->correction_azi_cu_, this->correction_ele_cu_, this->firetimes_cu_, 
    this->point_data_cu_, this->packet_data_cu_, this->valid_points_cu_, frame.distance_unit, this->optical_center, cuda_Param,
    frame.packet_num, frame.per_points_num);
  if (ret != 0) return ret;

  cudaSafeCall(cudaMemcpy(this->points_, this->points_cu_,
                          frame.per_points_num * frame.packet_num * sizeof(LidarPointXYZDAE), 
                          cudaMemcpyDeviceToHost), ReturnCode::CudaMemcpyDeviceToHostError);
  for (uint32_t i = 0; i < frame.packet_num; i++) {
    uint32_t point_index = i * frame.per_points_num;
    int point_num = 0;
    for (uint32_t j = point_index; j < point_index + frame.valid_points[i]; j++) {
      if (frame.fParam.config.fov_start != -1 && frame.fParam.config.fov_end != -1) {
        int fov_transfer = this->points_[j].azimuthCalib / M_PI * HALF_CIRCLE;
        if (fov_transfer < frame.fParam.config.fov_start || fov_transfer > frame.fParam.config.fov_end) { //不在fov范围continue
          continue;
        }
      }
      PUT_POINT_IN_POINT_INFO
        uint64_t timestamp = packetData.t.sensor_timestamp * kMicrosecondToNanosecondInt + pointData.data.dnsOff.ns_offset;
        set_timestamp(ptinfo, double(packetData.t.sensor_timestamp) / kMicrosecondToSecond);
        set_timeSecond(ptinfo, timestamp / kNanosecondToSecondInt);
        set_timeNanosecond(ptinfo, timestamp % kNanosecondToSecondInt);

        point_num++;
      }
    }
    frame.valid_points[i] = point_num;
  }
  return 0;
}

#endif //Udp_P40_PARSER_GPU_CU_