/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include <cuda_runtime_api.h>

#include "ScatterBEV_kernels.h"

__global__ void scatterBEV_kernel(const float *pillar_features_data,
          const float *coords_data, const unsigned int *params_data,
          unsigned int featureX, unsigned int featureY,
          float *spatial_feature_data)
{
  int pillar_idx = blockIdx.x * PILLARS_PER_BLOCK + threadIdx.x;
  int valid_pillars_inBlock = PILLARS_PER_BLOCK;

  const int num_pillars = params_data[4];
  if ((blockIdx.x * PILLARS_PER_BLOCK) > num_pillars) return;
  int valid_blocks = (num_pillars+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK;
  if(blockIdx.x >= valid_blocks) return;

  if(blockIdx.x == (valid_blocks-1)) {
    valid_pillars_inBlock = num_pillars % PILLARS_PER_BLOCK;
  }

  valid_pillars_inBlock = (valid_pillars_inBlock==0) ? PILLARS_PER_BLOCK : valid_pillars_inBlock;

  __shared__ float pillarSM[PILLARS_PER_BLOCK][FEATURE_SIZE]; //pillar*64

  for (int i = 0; i < valid_pillars_inBlock; i++)
  {
    pillarSM[i][threadIdx.x] = pillar_features_data[ (blockIdx.x * PILLARS_PER_BLOCK +i)*FEATURE_SIZE + threadIdx.x];
  }

  __syncthreads();

  if(pillar_idx >= num_pillars) return;

  float4 coord = ((const float4 *)coords_data)[pillar_idx];

  int x = (int)coord.w;
  int y = (int)coord.z;

  for (int i = 0; i < FEATURE_SIZE; i++)
  {
    spatial_feature_data[i*featureY*featureX + y*featureX + x] = pillarSM[threadIdx.x][i];
  }

}

cudaError_t scatterBEV_kernel_launcher(const float *pillar_features_data,
        const float *coords_data,
        const unsigned int *params_data,
        unsigned int featureX, unsigned int featureY,
        float *spatial_feature_data,
        cudaStream_t stream)
{

  dim3 blocks( (featureX*featureY+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK);
  dim3 threads(PILLARS_PER_BLOCK);

  scatterBEV_kernel<<<blocks, threads, 0, stream>>>
    (pillar_features_data, coords_data, params_data, featureX, featureY, spatial_feature_data);

  return cudaGetLastError();
}

__global__ void reduceMax_kernel(const float *in,
              float *out, unsigned int pillarCount)
{
  int pillar_idx = (blockIdx.x * blockDim.x + threadIdx.x)/32;
  if(pillar_idx >= pillarCount) return;

  int point_idx = threadIdx.x%32;

  //__shared__ float2 pillarSM[POINTS_PER_PILLAR]; //pillar*64

  unsigned int indexIn = pillar_idx*POINTS_PER_PILLAR*FEATURE_SIZE
                          + 0*FEATURE_SIZE
                          + point_idx * 2;
  float2 outFeature = *((float2*)(in + indexIn));
  for (int i = 1; i < POINTS_PER_PILLAR; i++)
  {
    indexIn = pillar_idx*POINTS_PER_PILLAR*FEATURE_SIZE + i*FEATURE_SIZE + point_idx * 2;
    float2 maxFeature = *((float2*)(in + indexIn));
    outFeature.x = max(outFeature.x, maxFeature.x);
    outFeature.y = max(outFeature.y, maxFeature.y);
  }
  unsigned int indexOut = pillar_idx*FEATURE_SIZE + point_idx*2;
  outFeature.x = max(outFeature.x, 0.0);
  outFeature.y = max(outFeature.y, 0.0);
  *((float2*)(out + indexOut)) = outFeature;

}

cudaError_t reduceMax_kernel_launcher(const float *in,
              float *out, unsigned int pillarCount,
              cudaStream_t stream)
{
  dim3 threads(4*POINTS_PER_PILLAR);
  dim3 blocks((pillarCount*POINTS_PER_PILLAR+threads.x-1) / threads.x);

  reduceMax_kernel<<<blocks, threads, 0, stream>>>
    (in, out, pillarCount);

  return cudaGetLastError();
}