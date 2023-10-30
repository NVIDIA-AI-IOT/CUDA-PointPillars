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

#include "pillarscatter-kernel.hpp"
#include <stdio.h>
#include <stdlib.h>

const int PILLARS_PER_BLOCK = 64;
const int PILLAR_FEATURE_SIZE = 64;

__global__ void pillarScatterHalfkernel(const half *pillar_features_data,
                                        const unsigned int *coords_data, const unsigned int *params_data,
                                        unsigned int featureX, unsigned int featureY,
                                        half *spatial_feature_data)
{
    int pillar_idx = blockIdx.x * PILLARS_PER_BLOCK + threadIdx.x;
    int valid_pillars_inBlock = PILLARS_PER_BLOCK;
    const int num_pillars = params_data[0];
    int valid_blocks = (num_pillars+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK;
    if(blockIdx.x >= valid_blocks) return;
    if(blockIdx.x == (valid_blocks-1)) {
        valid_pillars_inBlock = num_pillars % PILLARS_PER_BLOCK;
    }
    valid_pillars_inBlock = (valid_pillars_inBlock==0) ? PILLARS_PER_BLOCK : valid_pillars_inBlock;
    __shared__ half pillarSM[PILLARS_PER_BLOCK][PILLAR_FEATURE_SIZE]; //pillar*64
    for (int i = 0; i < valid_pillars_inBlock; i++)
    {
        pillarSM[i][threadIdx.x] = pillar_features_data[ (blockIdx.x * PILLARS_PER_BLOCK +i)*PILLAR_FEATURE_SIZE + threadIdx.x];
    }
    __syncthreads();
    if(pillar_idx >= num_pillars) return;
    uint4 coord = ((const uint4 *)coords_data)[pillar_idx];
    unsigned int x = coord.w;
    unsigned int y = coord.z;

    // Output tensor format : kHWC8, [N][H][W][(C+7)/8*8]
    int C_stride = (PILLAR_FEATURE_SIZE+7)/8*8;
    for (int i = 0; i < PILLAR_FEATURE_SIZE; i++)
    {
        spatial_feature_data[y*featureX*C_stride + x*C_stride + i] = pillarSM[threadIdx.x][i];
    }
}

__global__ void pillarScatterFloatkernel(const float *pillar_features_data,
                                         const unsigned int *coords_data, const unsigned int *params_data,
                                         unsigned int featureX, unsigned int featureY,
                                         float *spatial_feature_data)
{
    int pillar_idx = blockIdx.x * PILLARS_PER_BLOCK + threadIdx.x;
    int valid_pillars_inBlock = PILLARS_PER_BLOCK;
    const int num_pillars = params_data[0];
    int valid_blocks = (num_pillars+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK;
    if(blockIdx.x >= valid_blocks) return;
    if(blockIdx.x == (valid_blocks-1)) {
        valid_pillars_inBlock = num_pillars % PILLARS_PER_BLOCK;
    }
    valid_pillars_inBlock = (valid_pillars_inBlock==0) ? PILLARS_PER_BLOCK : valid_pillars_inBlock;
    __shared__ float pillarSM[PILLARS_PER_BLOCK][PILLAR_FEATURE_SIZE]; //pillar*64
    for (int i = 0; i < valid_pillars_inBlock; i++)
    {
        pillarSM[i][threadIdx.x] = pillar_features_data[ (blockIdx.x * PILLARS_PER_BLOCK +i)*PILLAR_FEATURE_SIZE + threadIdx.x];
    }
    __syncthreads();
    if(pillar_idx >= num_pillars) return;
    uint4 coord = ((const uint4 *)coords_data)[pillar_idx];
    unsigned int x = coord.w;
    unsigned int y = coord.z;

    for (int i = 0; i < PILLAR_FEATURE_SIZE; i++)
    {
        spatial_feature_data[i*featureY*featureX + y*featureX + x] = pillarSM[threadIdx.x][i];
    }
}

int pillarScatterHalfKernelLaunch(const half *pillar_features_data,
                                  const unsigned int *coords_data,
                                  const unsigned int *params_data,
                                  unsigned int featureX, unsigned int featureY,
                                  half *spatial_feature_data,
                                  cudaStream_t stream)
{
    dim3 blocks((featureX*featureY+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK);
    dim3 threads(PILLARS_PER_BLOCK);

    pillarScatterHalfkernel<<<blocks, threads, 0, stream>>>(pillar_features_data, coords_data, params_data, featureX, featureY, spatial_feature_data);
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int pillarScatterFloatKernelLaunch(const float *pillar_features_data,
                                   const unsigned int *coords_data,
                                   const unsigned int *params_data,
                                   unsigned int featureX, unsigned int featureY,
                                   float *spatial_feature_data,
                                   cudaStream_t stream)
{
    dim3 blocks((featureX*featureY+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK);
    dim3 threads(PILLARS_PER_BLOCK);

    pillarScatterFloatkernel<<<blocks, threads, 0, stream>>>(pillar_features_data, coords_data, params_data, featureX, featureY, spatial_feature_data);
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
