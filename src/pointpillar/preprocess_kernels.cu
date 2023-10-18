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

#include "kernel.h"

__global__ void generateVoxels_random_kernel(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= points_size) return;

  float4 point = ((float4*)points)[point_idx];

  if(point.x<min_x_range||point.x>=max_x_range
    || point.y<min_y_range||point.y>=max_y_range
    || point.z<min_z_range||point.z>=max_z_range) return;

  int voxel_idx = floorf((point.x - min_x_range)/pillar_x_size);
  int voxel_idy = floorf((point.y - min_y_range)/pillar_y_size);
  unsigned int voxel_index = voxel_idy * grid_x_size
                            + voxel_idx;

  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);

  if(point_id >= POINTS_PER_VOXEL) return;
  float *address = voxels + (voxel_index*POINTS_PER_VOXEL + point_id)*4;
  atomicExch(address+0, point.x);
  atomicExch(address+1, point.y);
  atomicExch(address+2, point.z);
  atomicExch(address+3, point.w);
}

cudaError_t generateVoxels_random_launch(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((points_size+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateVoxels_random_kernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask, voxels);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void generateVoxelsList_kernel(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, int *voxelsList)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= points_size) return;

  float4 point = ((float4*)points)[point_idx];

  if(point.x<min_x_range||point.x>=max_x_range
    || point.y<min_y_range||point.y>=max_y_range
    || point.z<min_z_range||point.z>=max_z_range)
  {
    voxelsList[point_idx] = -1;
    return;
  }

  int voxel_idx = floorf((point.x - min_x_range)/pillar_x_size);
  int voxel_idy = floorf((point.y - min_y_range)/pillar_y_size);
  unsigned int voxel_index = voxel_idy * grid_x_size
                            + voxel_idx;

  atomicAdd(&(mask[voxel_index]), 1);
  voxelsList[point_idx] = voxel_index;

}

__global__ void generateVoxels_kernel(float *points, size_t points_size,
        int *voxelsList,
        unsigned int *mask, float *voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= points_size) return;

  int voxel_index = voxelsList[point_idx];

  if (voxel_index == -1) return;
  int point_id = atomicAdd(&(mask[voxel_index]), 1);

  if(point_id >= POINTS_PER_VOXEL) return;
  float *address = voxels + (voxel_index*POINTS_PER_VOXEL + point_id)*4;
  float4 point = ((float4*)points)[point_idx];
  atomicExch(address+0, point.x);
  atomicExch(address+1, point.y);
  atomicExch(address+2, point.z);
  atomicExch(address+3, point.w);
}

__global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs)
{
  unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(voxel_idx >= grid_x_size ||voxel_idy >= grid_y_size) return;

  unsigned int voxel_index = voxel_idy * grid_x_size
                           + voxel_idx;
  unsigned int count = mask[voxel_index];
  if( !(count>0) ) return;
  count = count<POINTS_PER_VOXEL?count:POINTS_PER_VOXEL;

  unsigned int current_pillarId = 0;
  current_pillarId = atomicAdd(pillar_num, 1);

  voxel_num[current_pillarId] = count;

  uint4 idx = {0, 0, voxel_idy, voxel_idx};
  ((uint4*)voxel_idxs)[current_pillarId] = idx;

  for (int i=0; i<count; i++){
    int inIndex = voxel_index*POINTS_PER_VOXEL + i;
    int outIndex = current_pillarId*POINTS_PER_VOXEL + i;
    ((float4*)voxel_features)[outIndex] = ((float4*)voxels)[inIndex];
  }

  // clear buffer for next infer
  atomicExch(mask + voxel_index, 0);
}

// create 4 channels
cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs,
        cudaStream_t stream)
{
  dim3 threads = {32,32};
  dim3 blocks = {(grid_x_size + threads.x -1)/threads.x,
                 (grid_y_size + threads.y -1)/threads.y};

  generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>
      (mask, voxels, grid_y_size, grid_x_size,
       pillar_num,
       voxel_features,
       voxel_num,
       voxel_idxs);
  cudaError_t err = cudaGetLastError();
  return err;
}

// 4 channels -> 10 channels
__global__ void generateFeatures_kernel(float* voxel_features,
    unsigned int* voxel_num, unsigned int* voxel_idxs, unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features)
{
    int pillar_idx = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x/WARP_SIZE;
    int point_idx = threadIdx.x % WARP_SIZE;

    int pillar_idx_inBlock = threadIdx.x/32;
    unsigned int num_pillars = params[0];

    if (pillar_idx >= num_pillars) return;

    __shared__ float4 pillarSM[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ float4 pillarSumSM[WARPS_PER_BLOCK];
    __shared__ uint4 idxsSM[WARPS_PER_BLOCK];
    __shared__ int pointsNumSM[WARPS_PER_BLOCK];
    __shared__ float pillarOutSM[WARPS_PER_BLOCK][WARP_SIZE][FEATURES_SIZE];

    if (threadIdx.x < WARPS_PER_BLOCK) {
      pointsNumSM[threadIdx.x] = voxel_num[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
      idxsSM[threadIdx.x] = ((uint4*)voxel_idxs)[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
      pillarSumSM[threadIdx.x] = {0,0,0,0};
    }

    pillarSM[pillar_idx_inBlock][point_idx] = ((float4*)voxel_features)[pillar_idx*WARP_SIZE + point_idx];
    __syncthreads();

    //calculate sm in a pillar
    if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].x),  pillarSM[pillar_idx_inBlock][point_idx].x);
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].y),  pillarSM[pillar_idx_inBlock][point_idx].y);
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].z),  pillarSM[pillar_idx_inBlock][point_idx].z);
    }
    __syncthreads();

    //feature-mean
    float4 mean;
    float validPoints = pointsNumSM[pillar_idx_inBlock];
    mean.x = pillarSumSM[pillar_idx_inBlock].x / validPoints;
    mean.y = pillarSumSM[pillar_idx_inBlock].y / validPoints;
    mean.z = pillarSumSM[pillar_idx_inBlock].z / validPoints;

    mean.x  = pillarSM[pillar_idx_inBlock][point_idx].x - mean.x;
    mean.y  = pillarSM[pillar_idx_inBlock][point_idx].y - mean.y;
    mean.z  = pillarSM[pillar_idx_inBlock][point_idx].z - mean.z;


    //calculate offset
    float x_offset = voxel_x / 2 + idxsSM[pillar_idx_inBlock].w * voxel_x + range_min_x;
    float y_offset = voxel_y / 2 + idxsSM[pillar_idx_inBlock].z * voxel_y + range_min_y;
    float z_offset = voxel_z / 2 + idxsSM[pillar_idx_inBlock].y * voxel_z + range_min_z;

    //feature-offset
    float4 center;
    center.x  = pillarSM[pillar_idx_inBlock][point_idx].x - x_offset;
    center.y  = pillarSM[pillar_idx_inBlock][point_idx].y - y_offset;
    center.z  = pillarSM[pillar_idx_inBlock][point_idx].z - z_offset;

    //store output
    if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
      pillarOutSM[pillar_idx_inBlock][point_idx][0] = pillarSM[pillar_idx_inBlock][point_idx].x;
      pillarOutSM[pillar_idx_inBlock][point_idx][1] = pillarSM[pillar_idx_inBlock][point_idx].y;
      pillarOutSM[pillar_idx_inBlock][point_idx][2] = pillarSM[pillar_idx_inBlock][point_idx].z;
      pillarOutSM[pillar_idx_inBlock][point_idx][3] = pillarSM[pillar_idx_inBlock][point_idx].w;

      pillarOutSM[pillar_idx_inBlock][point_idx][4] = mean.x;
      pillarOutSM[pillar_idx_inBlock][point_idx][5] = mean.y;
      pillarOutSM[pillar_idx_inBlock][point_idx][6] = mean.z;

      pillarOutSM[pillar_idx_inBlock][point_idx][7] = center.x;
      pillarOutSM[pillar_idx_inBlock][point_idx][8] = center.y;
      pillarOutSM[pillar_idx_inBlock][point_idx][9] = center.z;

    } else {
      pillarOutSM[pillar_idx_inBlock][point_idx][0] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][1] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][2] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][3] = 0;

      pillarOutSM[pillar_idx_inBlock][point_idx][4] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][5] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][6] = 0;

      pillarOutSM[pillar_idx_inBlock][point_idx][7] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][8] = 0;
      pillarOutSM[pillar_idx_inBlock][point_idx][9] = 0;
    }

    __syncthreads();

    for(int i = 0; i < FEATURES_SIZE; i ++) {
      int outputSMId = pillar_idx_inBlock*WARP_SIZE*FEATURES_SIZE + i* WARP_SIZE + point_idx;
      int outputId = pillar_idx*WARP_SIZE*FEATURES_SIZE + i* WARP_SIZE + point_idx;
      features[outputId] = ((float*)pillarOutSM)[outputSMId] ;
    }

}

cudaError_t generateFeatures_launch(float* voxel_features,
    unsigned int * voxel_num,
    unsigned int* voxel_idxs,
    unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features,
    cudaStream_t stream)
{
    dim3 blocks( (MAX_VOXELS+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK);
    dim3 threads(WARPS_PER_BLOCK*WARP_SIZE);

    generateFeatures_kernel<<<blocks, threads, 0, stream>>>
      (voxel_features,
      voxel_num,
      voxel_idxs,
      params,
      voxel_x, voxel_y, voxel_z,
      range_min_x, range_min_y, range_min_z,
      features);

    cudaError_t err = cudaGetLastError();
    return err;
}
