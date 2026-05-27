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

// DO NOT change this file !!!!!
#ifndef __RETURE_CODE__
#define __RETURE_CODE__

#include <vector>
#include <cinttypes>
#include <memory>
namespace hesai
{
namespace lidar
{
enum class ReturnCode : int {
  Default = 0,
  CorrectionsUnloaded = 1,
  ConfigJsonUnloaded = 2,
  LiDAR2SpeedometerCalibrationUnloaded = 3,
  LiDAR2GroundCalibrationUnloaded = 4,
  CudaMemcpyHostToDeviceError = 5,
  CudaMemcpyDeviceToHostError = 6,
  LiDARPacketInvalid = 7,
  LiDARPacketExpired = 8,
  LiDARPacketNewFrame = 9,
  LiDARPacketPreFrame = 10,
  LiDARPacketPostFrame = 11,
  PerceptionHistoryReset = 12,
  WaitForNextFrame = 13,
  LiDARRawFrameNotReady = 14,
  FrameScansTooFew = 15,
  CudaXYZComputingError = 16,
  CudaMotionCompensationError = 17,
  GravityDirectionHistoryEmpty = 18,
  GravityDirectionHistoryExpired = 19,
  GravityExpired = 20,
  RPYRateHistoryEmpty = 21,
  VelocityHistoryEmpty = 22,
  RPYRateHistoryExpired = 23,
  VelocityHistoryExpired = 24,
  SpeedometerExpired = 25,
  RPYRateStuck = 26,
  VelocityStuck = 27,
  SpeedometerStuck = 28,
  CudaPointsClusteringError = 29,
  CudaProposeObstacleCuboidsError = 30,
  CudaComputeCuboidTimestampError = 31,
  DetectionInferenceError = 32,
  SegmentationInferenceError = 33,
  Total
};
}
}
#endif  // !__AT128Perception_RETURE_CODE__
