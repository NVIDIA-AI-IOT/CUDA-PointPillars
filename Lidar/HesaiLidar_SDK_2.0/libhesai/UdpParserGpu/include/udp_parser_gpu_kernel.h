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
#ifndef Udp_PARSER_GPU_KERNEL_H_
#define Udp_PARSER_GPU_KERNEL_H_
#include "general_struct_gpu.h"
#include "general_parser_gpu.h"
#include "udp_protocol_v1_4.h"
#include "udp_protocol_v3_2.h"
#include "udp_protocol_v4_3.h"
#include "udp_protocol_v4_7.h"

namespace hesai
{
namespace lidar
{

int compute_p40_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const float* firetimes, 
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_1_4_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const pandarN::FiretimesPandarN* firetimes, 
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_1_8_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele,
    const PointDecodeData* point_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_3_2_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const QT128::FiretimesQt128* firetimes,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_4_3_cuda(LidarPointXYZDAE* points, const AT::ATCorrectionFloat* correction,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_4_7_cuda(LidarPointXYZDAE* points, const ATX::ATXCorrectionFloat* correction, const ATX::ATXFiretimesFloat* firetimes,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_6_1_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const float* firetimes, 
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

int compute_7_2_cuda(LidarPointXYZDAE* points, const PointDecodeData* point_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num);

}
}
#endif  // Udp_PARSER_GPU_KERNEL_H_
