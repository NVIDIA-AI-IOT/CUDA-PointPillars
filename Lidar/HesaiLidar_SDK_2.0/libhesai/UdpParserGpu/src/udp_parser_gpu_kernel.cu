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

#include "udp_parser_gpu_kernel.h"

namespace hesai
{
namespace lidar
{

__global__ void compute_xyzs_p40_impl(LidarPointXYZDAE *xyzs, const float* correction_azi, const float* correction_ele, 
    const float* firetimes, const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  auto pa_data = packet_data[packet_index];
  float azimuth = p_data.azimuth / HALF_CIRCLE * M_PI;
  float theta = correction_azi[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float phi = correction_ele[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;
  if (fParam.firetimes_flag) {
    azimuth += (fParam.rotation_flag > 0 ? 1 : -1) * (firetimes[p_data.channel_index] * pa_data.spin_speed * 6E-6) / HALF_CIRCLE * M_PI;
  }

  if(rho > 0.09 && fParam.distance_correction_flag) {
    float tx = cos(phi) * sin(theta);
    float ty = cos(phi) * cos(theta);
    float tz = sin(phi);
    float B = 2 * tx * optical_center.x + 2 * ty * optical_center.y + 2 * tz * optical_center.z;
    float C = optical_center.x * optical_center.x + optical_center.y * optical_center.y + optical_center.z * optical_center.z - rho * rho;
    float d_opitcal = sqrt(B * B / 4 - C) - B / 2;
    float x = d_opitcal * tx + optical_center.x;
    float y = d_opitcal * ty + optical_center.y;
    float z = d_opitcal * tz + optical_center.z;
    theta = azimuth + atan2(x, y);
    phi = asin(z / rho);
  } else {
    theta += azimuth;
  }

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_p40_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const float* firetimes, 
  const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
  const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_p40_impl<<<packet_num, per_points_num>>>(points, correction_azi, correction_ele, firetimes, 
      point_data, packet_data, valid_points, raw_distance_unit, optical_center, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}

__global__ void compute_xyzs_1_4_impl(LidarPointXYZDAE *xyzs, const float* correction_azi, const float* correction_ele, 
    const pandarN::FiretimesPandarN* firetimes, const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  auto pa_data = packet_data[packet_index];
  float azimuth = p_data.azimuth / HALF_CIRCLE * M_PI;
  float theta = correction_azi[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float phi = correction_ele[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;
  if (fParam.firetimes_flag) {
    int idx = 0;
    if (pa_data.data.d1_4.optMode == 0) idx = p_data.data.d1_4.angleState;
    else if (pa_data.data.d1_4.optMode == 2) idx = p_data.data.d1_4.angleState + 4;
    else if (pa_data.data.d1_4.optMode == 3) idx = p_data.data.d1_4.angleState + 6;
    int k = (rho >= firetimes->section_distance) ? 0 : 1;
    if (p_data.channel_index < pandarN::kMaxChannelPandarN && idx < 8)
      azimuth += (fParam.rotation_flag > 0 ? 1 : -1) * 
        (firetimes->firetime_section_values[p_data.channel_index].section_values[idx].firetime[k] * pa_data.spin_speed * 6E-9) / HALF_CIRCLE * M_PI;
  }

  if(rho > 0.09 && fParam.distance_correction_flag) {
    float tx = cos(phi) * sin(theta);
    float ty = cos(phi) * cos(theta);
    float tz = sin(phi);
    float B = 2 * tx * optical_center.x + 2 * ty * optical_center.y + 2 * tz * optical_center.z;
    float C = optical_center.x * optical_center.x + optical_center.y * optical_center.y + optical_center.z * optical_center.z - rho * rho;
    float d_opitcal = sqrt(B * B / 4 - C) - B / 2;
    float x = d_opitcal * tx + optical_center.x;
    float y = d_opitcal * ty + optical_center.y;
    float z = d_opitcal * tz + optical_center.z;
    theta = azimuth + atan2(x, y);
    phi = asin(z / rho);
  } else {
    theta += azimuth;
  }

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_1_4_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const pandarN::FiretimesPandarN* firetimes, 
  const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
  const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_1_4_impl<<<packet_num, per_points_num>>>(points, correction_azi, correction_ele, firetimes, 
      point_data, packet_data, valid_points, raw_distance_unit, optical_center, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}


__global__ void compute_xyzs_1_8_impl(LidarPointXYZDAE *xyzs, const float* correction_azi, const float* correction_ele, 
    const PointDecodeData* point_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  float azimuth = p_data.azimuth / HALF_CIRCLE * M_PI;
  float theta = correction_azi[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float phi = correction_ele[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;

  if(rho > 0.09 && fParam.distance_correction_flag) {
    theta -= 190.0 / HALF_CIRCLE * M_PI;
    float tx = cos(phi) * sin(theta);
    float ty = cos(phi) * cos(theta);
    float tz = sin(phi);
    float d = rho;
    float x = d * tx + optical_center.x;
    float y = d * ty + optical_center.y;
    float z = d * tz + optical_center.z;
    float d_geometric_center = sqrt(x * x + y * y + z * z);
    theta = azimuth + atan2(x, y);
    phi = asin(z / d_geometric_center);
    rho = d_geometric_center;
    theta += 190.0 / HALF_CIRCLE * M_PI;
  } else {
    theta += azimuth;
  }

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_1_8_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele,
  const PointDecodeData* point_data, const uint32_t* valid_points, const double raw_distance_unit, 
  const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_1_8_impl<<<packet_num, per_points_num>>>(points, correction_azi, correction_ele,
      point_data, valid_points, raw_distance_unit, optical_center, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}









__global__ void compute_xyzs_3_2_impl(LidarPointXYZDAE *xyzs, const float* correction_azi, const float* correction_ele, 
    const QT128::FiretimesQt128* firetimes, const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  auto pa_data = packet_data[packet_index];
  float azimuth = p_data.azimuth / HALF_CIRCLE * M_PI;
  float theta = correction_azi[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float phi = correction_ele[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;
  if (fParam.firetimes_flag) {
    azimuth += (fParam.rotation_flag > 0 ? 1 : -1) * 
        (firetimes->firetimes[p_data.data.dQT.loopIndex][p_data.channel_index] * pa_data.spin_speed * 6E-6) / HALF_CIRCLE * M_PI;
  }

  if(rho > 0.09 && fParam.distance_correction_flag) {
    float tx = cos(phi) * sin(theta);
    float ty = cos(phi) * cos(theta);
    float tz = sin(phi);
    float B = 2 * tx * optical_center.x + 2 * ty * optical_center.y + 2 * tz * optical_center.z;
    float C = optical_center.x * optical_center.x + optical_center.y * optical_center.y + optical_center.z * optical_center.z - rho * rho;
    float d_opitcal = sqrt(B * B / 4 - C) - B / 2;
    float x = d_opitcal * tx + optical_center.x;
    float y = d_opitcal * ty + optical_center.y;
    float z = d_opitcal * tz + optical_center.z;
    theta = azimuth + atan2(x, y);
    phi = asin(z / rho);
  } else {
    theta += azimuth;
  }

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_3_2_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const QT128::FiretimesQt128* firetimes,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_3_2_impl<<<packet_num, per_points_num>>>(points, correction_azi, correction_ele, firetimes,
      point_data, packet_data, valid_points, raw_distance_unit, optical_center, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}

__global__ void compute_xyzs_4_3_impl(LidarPointXYZDAE *xyzs, const AT::ATCorrectionFloat* correction,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  int field = p_data.data.dAT.field;
  float theta = 0;
  float phi = correction->f_elevations[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;

  {
    int STEP3 = correction->adjust_interval;    
    int i = int(floor(p_data.azimuth / STEP3));
    int m = p_data.azimuth - i * STEP3;
    float k = 1.f * m / STEP3;
    int laser_index = p_data.channel_index;
    if (correction->min_version == 5 || correction->min_version == 6) {
      laser_index = p_data.channel_index;
    }
    else if (correction->min_version == 7 || correction->min_version == 8 || correction->min_version == 9) {
      laser_index = correction->channel_laser_map[p_data.channel_index];
    } 
    phi += ((1 - k) * correction->f_elevation_adjust[laser_index * 360 / correction->adjust_interval + i] +
                k * correction->f_elevation_adjust[laser_index * 360 / correction->adjust_interval + i + 1]) / HALF_CIRCLE * M_PI;
  }
  {
    theta = (p_data.azimuth + kCircle - correction->start_frame[field] / kAllFineResolutionFloat) * 2 -
              correction->f_azimuths[p_data.channel_index];
    int STEP3 = correction->adjust_interval;    
    int i = int(floor(p_data.azimuth / STEP3));
    int m = p_data.azimuth - i * STEP3;
    float k = 1.f * m / STEP3;
    int laser_index = p_data.channel_index;
    if (correction->min_version == 5 || correction->min_version == 6) {
      laser_index = p_data.channel_index;
    }
    else if (correction->min_version == 7 || correction->min_version == 8 || correction->min_version == 9) {
      laser_index = correction->channel_laser_map[p_data.channel_index];
    } 
    theta += ((1 - k) * correction->f_azimuth_adjust[laser_index * 360 / correction->adjust_interval + i] +
                k * correction->f_azimuth_adjust[laser_index * 360 / correction->adjust_interval + i + 1]);
    theta += p_data.data.dAT.azimuth_offset;
    theta = theta / HALF_CIRCLE * M_PI;
  }
  
  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_4_3_cuda(LidarPointXYZDAE* points, const AT::ATCorrectionFloat* correction,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_4_3_impl<<<packet_num, per_points_num>>>(points, correction,
      point_data, packet_data, valid_points, raw_distance_unit, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}

__global__ void compute_xyzs_4_7_impl(LidarPointXYZDAE *xyzs, const ATX::ATXCorrectionFloat* correction, const ATX::ATXFiretimesFloat* firetimes,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  auto pa_data = packet_data[packet_index];
  float raw_azimuth = p_data.azimuth;
  float raw_elevation = 0;
  float rho = p_data.distance * raw_distance_unit;

  if (fParam.firetimes_flag) {
    raw_azimuth += ((pa_data.data.dATX.frame_id % 2 == 0) ? firetimes->even_firetime_correction_[p_data.channel_index] : 
                      - firetimes->odd_firetime_correction_[p_data.channel_index]) * (abs(pa_data.spin_speed) * 1E-9 / 8);
  }

  if (correction->min_version == 1) {
    raw_azimuth += correction->azimuth[p_data.channel_index];
  }
  else if (correction->min_version == 2 || correction->min_version == 3) {
    raw_azimuth += ((pa_data.data.dATX.frame_id % 2 == 0) ? correction->azimuth_even[p_data.channel_index] : 
                      correction->azimuth_odd[p_data.channel_index]);
  }
  if(correction->min_version == 1 || correction->min_version == 2) {
    raw_elevation = correction->elevation[p_data.channel_index];
  } 
  else if (correction->min_version == 3) {
    raw_elevation = correction->elevation[p_data.channel_index];
    const float BegElevationAdjust = 20.0;
    const float StepElevationAdjust = 2.0;
    const float EndElevationAdjust = 158.0;  // 20 + 2 * (70 - 1)
    if (raw_azimuth >= StepElevationAdjust && raw_azimuth <= EndElevationAdjust) {
      int index = (raw_azimuth - BegElevationAdjust) / StepElevationAdjust;
      float left_percent = (raw_azimuth - BegElevationAdjust - index * StepElevationAdjust) / StepElevationAdjust;
      raw_elevation += correction->elevation_adjust[index] * (1 - left_percent) + correction->elevation_adjust[index + 1] * left_percent;
    }
  }
  float theta = raw_azimuth / HALF_CIRCLE * M_PI;
  float phi = raw_elevation / HALF_CIRCLE * M_PI;
  
  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_4_7_cuda(LidarPointXYZDAE* points, const ATX::ATXCorrectionFloat* correction, const ATX::ATXFiretimesFloat* firetimes,
    const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
    const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_4_7_impl<<<packet_num, per_points_num>>>(points, correction, firetimes,
      point_data, packet_data, valid_points, raw_distance_unit, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}





__global__ void compute_xyzs_6_1_impl(LidarPointXYZDAE *xyzs, const float* correction_azi, const float* correction_ele, 
    const float* firetimes, const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  auto pa_data = packet_data[packet_index];
  float azimuth = p_data.azimuth / HALF_CIRCLE * M_PI;
  float theta = correction_azi[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float phi = correction_ele[p_data.channel_index] / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;
  if (fParam.firetimes_flag) {
    azimuth += (fParam.rotation_flag > 0 ? 1 : -1) * (firetimes[p_data.channel_index] * pa_data.spin_speed * 6E-6) / HALF_CIRCLE * M_PI;
  }

  if(rho > 0.09 && fParam.distance_correction_flag) {
    float tx = cos(phi) * sin(theta);
    float ty = cos(phi) * cos(theta);
    float tz = sin(phi);
    float B = 2 * tx * optical_center.x + 2 * ty * optical_center.y + 2 * tz * optical_center.z;
    float C = optical_center.x * optical_center.x + optical_center.y * optical_center.y + optical_center.z * optical_center.z - rho * rho;
    float d_opitcal = sqrt(B * B / 4 - C) - B / 2;
    float x = d_opitcal * tx + optical_center.x;
    float y = d_opitcal * ty + optical_center.y;
    float z = d_opitcal * tz + optical_center.z;
    theta = azimuth + atan2(x, y);
    phi = asin(z / rho);
  } else {
    theta += azimuth;
  }

  if (fParam.xt_spot_correction && (rho >= 0.25 && rho < 4.25)) 
  {
    int index = int((rho - 0.25) / 0.5);
    index = index > 7 ? 7 : index;
    float spot_correction_angle[8] = {3, 3, 25, 25, 20, 15, 8, 6};
    theta -= spot_correction_angle[index] / kResolutionFloat / HALF_CIRCLE * M_PI;
  }

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_6_1_cuda(LidarPointXYZDAE* points, const float* correction_azi, const float* correction_ele, const float* firetimes, 
  const PointDecodeData* point_data, const PacketDecodeData* packet_data, const uint32_t* valid_points, const double raw_distance_unit, 
  const LidarOpticalCenter optical_center, const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_6_1_impl<<<packet_num, per_points_num>>>(points, correction_azi, correction_ele, firetimes, 
      point_data, packet_data, valid_points, raw_distance_unit, optical_center, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}

__global__ void compute_xyzs_7_2_impl(LidarPointXYZDAE *xyzs, const PointDecodeData* point_data, const uint32_t* valid_points, 
    const double raw_distance_unit, const FrameDecodeParam fParam, const uint32_t per_points_num) {
  auto packet_index = blockIdx.x;
  auto channel = threadIdx.x;
  if (channel >= valid_points[packet_index]) return;
  int point_index = packet_index * per_points_num + channel;
  auto p_data = point_data[point_index];
  float theta = p_data.azimuth / HALF_CIRCLE * M_PI;
  float phi = p_data.elevation / HALF_CIRCLE * M_PI;
  float rho = p_data.distance * raw_distance_unit;

  float z = rho * sin(phi);
  auto r = rho * cosf(phi);
  float x = r * sin(theta);
  float y = r * cos(theta);
  
  if (fParam.transform.use_flag) {
    auto& transform = fParam.transform;
    float cosa = cos(transform.roll);
    float sina = sin(transform.roll);
    float cosb = cos(transform.pitch);
    float sinb = sin(transform.pitch);
    float cosc = cos(transform.yaw);
    float sinc = sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
                (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
                (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;

    x = x_;
    y = y_;
    z = z_;
  }

  xyzs[point_index].x = x;
  xyzs[point_index].y = y;
  xyzs[point_index].z = z;
  xyzs[point_index].distance = rho;
  xyzs[point_index].azimuthCalib = theta;
  xyzs[point_index].elevationCalib = phi;
}

int compute_7_2_cuda(LidarPointXYZDAE* points, const PointDecodeData* point_data, const uint32_t* valid_points, const double raw_distance_unit, 
  const FrameDecodeParam fParam, const uint32_t packet_num, const uint32_t per_points_num) {
    compute_xyzs_7_2_impl<<<packet_num, per_points_num>>>(points, 
      point_data, valid_points, raw_distance_unit, fParam, per_points_num);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  return 0;
}









}
}
