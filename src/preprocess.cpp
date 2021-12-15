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

#include <assert.h>
#include <iostream>
#include <math.h>

#include <cuda_runtime_api.h>
#include "preprocess.h"
#include "preprocess_kernels.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}
//<< convert points cloud into voxexls by CPU
// transform_points_to_voxels
int PreProcessCuda::clearCacheCPU(void)
{
  int (*coor_to_voxelidx) [params_.grid_y_size][params_.grid_x_size] = nullptr;
  coor_to_voxelidx = (int (*) [params_.grid_y_size][params_.grid_x_size])coor_to_voxelidx_;
  for(int i =0;i< params_.grid_z_size;i++)
    for(int j =0;j< params_.grid_y_size;j++)
      for(int k =0;k< params_.grid_x_size;k++)
        coor_to_voxelidx[i][j][k] = -1;
  return 0;
}

void PreProcessCuda::generateVoxels_cpu(float* points, size_t points_size,
    unsigned int* pillarCount,
    float* voxel_features,
    float* voxel_num_points,
    float* coords)
{
  const static float point_cloud_range[6] = {0., -39.68, -3., 69.12, 39.68, 1.};
  const static float voxel_size[3] = {0.16, 0.16, 4.};
  const int max_num_points = 32;
  const int max_voxels = 40000;

  float* voxels = voxel_features;
  float* num_points_per_voxel = voxel_num_points;
  float* coors = coords;

  int N = points_size;
  const int num_features = 4;

  unsigned int voxel_num = 0;
  int indexX, indexY, indexZ;

  int voxel_idx, num;

  int (*coor_to_voxelidx) [params_.grid_y_size][params_.grid_x_size] = nullptr;
  coor_to_voxelidx = (int (*) [params_.grid_y_size][params_.grid_x_size])coor_to_voxelidx_;
/*
  int n_dim = 3;
  int grid_size[n_dim];
  #pragma unroll(3)
  for(int i=0;i<n_dim;i++) {
      grid_size[i] = round((point_cloud_range[n_dim + i] - point_cloud_range[i]) / voxel_size[i]);
  }
*/

  static int voxel_num_pervious = 0;
  for(int i =0;i< voxel_num_pervious;i++)
  {
    num_points_per_voxel[i] = 0.0f;
  }

  for(int i=0; i<N; i++) {
    if( !(points[i*num_features]>=point_cloud_range[0] && points[i*num_features]<point_cloud_range[3]
        && points[i*num_features+1]>=point_cloud_range[1] && points[i*num_features+1]<point_cloud_range[4]
        && points[i*num_features+2]>=point_cloud_range[2] && points[i*num_features+2]<point_cloud_range[5]
        ) ) {
      continue;
    }

    indexX = floor((points[i*num_features + 0] - point_cloud_range[0]) / voxel_size[0]);
    indexY = floor((points[i*num_features + 1] - point_cloud_range[1]) / voxel_size[1]);
    indexZ = floor((points[i*num_features + 2] - point_cloud_range[2]) / voxel_size[2]);

    voxel_idx = coor_to_voxelidx[indexZ][indexY][indexX];
    if(voxel_idx == -1) {
      if (voxel_num >= max_voxels) {
          continue;
      }
      voxel_idx = voxel_num;
      voxel_num += 1;
      coor_to_voxelidx[indexZ][indexY][indexX] = voxel_idx;
      // coors type: 0,z,y,z
      coors[voxel_idx*num_features+0+1] = indexZ;
      coors[voxel_idx*num_features+1+1] = indexY;
      coors[voxel_idx*num_features+2+1] = indexX;
    }
    num = (int)(num_points_per_voxel[voxel_idx]);
    if (num < max_num_points) {
      for(int k=0; k<num_features;k++) {
          voxels[voxel_idx*max_num_points*num_features + num*num_features + k]
            = points[i*num_features + k];
      }
      num_points_per_voxel[voxel_idx] += 1.0;
    }
  }

  pillarCount[4] = voxel_num;
  voxel_num_pervious = voxel_num;
  return;
}
//convert points cloud into voxexls by CPU>>

PreProcessCuda::PreProcessCuda(cudaStream_t stream)
{
  stream_ = stream;

  unsigned int mask_size = params_.grid_z_size
              * params_.grid_y_size
              * params_.grid_x_size
              * sizeof(unsigned int);

  unsigned int voxels_size = params_.grid_z_size
              * params_.grid_y_size
              * params_.grid_x_size
              * params_.max_num_points_per_pillar
              * params_.num_point_values
              * sizeof(float);

  checkCudaErrors(cudaMallocManaged((void **)&mask_, mask_size));
  checkCudaErrors(cudaMallocManaged((void **)&voxels_, voxels_size));
  checkCudaErrors(cudaMallocManaged((void **)&voxelsList_, 300000l*sizeof(int)));

  checkCudaErrors(cudaMemsetAsync(mask_, 0, mask_size, stream_));
  checkCudaErrors(cudaMemsetAsync(voxels_, 0, voxels_size, stream_));
  checkCudaErrors(cudaMemsetAsync(voxelsList_, 0, 300000l*sizeof(int), stream_));
  //for cpu
  unsigned int coor_to_voxelidx_size = params_.grid_z_size
              * params_.grid_y_size
              * params_.grid_x_size
              * sizeof(int);
  coor_to_voxelidx_ = (int *) malloc(coor_to_voxelidx_size);
  return;
}

PreProcessCuda::~PreProcessCuda()
{
  checkCudaErrors(cudaFree(mask_));
  checkCudaErrors(cudaFree(voxels_));
  checkCudaErrors(cudaFree(voxelsList_));
  free(coor_to_voxelidx_);
  return;
}

int PreProcessCuda::generateVoxels(float *points, size_t points_size,
        unsigned int *pillar_num,
        float *voxel_features,
        float *voxel_num_points,
        float *coords)
{
  int grid_x_size = params_.grid_x_size;
  int grid_y_size = params_.grid_y_size;
  int grid_z_size = params_.grid_z_size;
  assert(grid_z_size ==1);
  float min_x_range = params_.min_x_range;
  float max_x_range = params_.max_x_range;
  float min_y_range = params_.min_y_range;
  float max_y_range = params_.max_y_range;
  float min_z_range = params_.min_z_range;
  float max_z_range = params_.max_z_range;
  float pillar_x_size = params_.pillar_x_size;
  float pillar_y_size = params_.pillar_y_size;
  float pillar_z_size = params_.pillar_z_size;

#if 0
  checkCudaErrors(generateVoxels_launch(points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask_, voxels_, voxelsList_, stream_));
#else
  checkCudaErrors(generateVoxels_random_launch(points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask_, voxels_, stream_));
#endif
  checkCudaErrors(generateBaseFeatures_launch(mask_, voxels_,
      grid_y_size, grid_x_size,
      pillar_num,
      voxel_features,
      voxel_num_points,
      coords, stream_));

  return 0;
}

int PreProcessCuda::generateFeatures(float* voxel_features,
          float* voxel_num_points,
          float* coords,
          unsigned int *params,
          float* features)
{
  int grid_z_size = params_.grid_z_size;
  assert(grid_z_size ==1);
  float range_min_x = params_.min_x_range;
  float range_min_y = params_.min_y_range;
  float range_min_z = params_.min_z_range;

  float voxel_x = params_.pillar_x_size;
  float voxel_y = params_.pillar_y_size;
  float voxel_z = params_.pillar_z_size;

  checkCudaErrors(generateFeatures_launch(voxel_features,
      voxel_num_points,
      coords,
      params,
      voxel_x, voxel_y, voxel_z,
      range_min_x, range_min_y, range_min_z,
      features, stream_));

  return 0;
}

