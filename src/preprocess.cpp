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

#include "preprocess.h"
#include <assert.h>
#include <iostream>
#include <math.h>

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

  return;
}

PreProcessCuda::~PreProcessCuda()
{
  checkCudaErrors(cudaFree(mask_));
  checkCudaErrors(cudaFree(voxels_));
  checkCudaErrors(cudaFree(voxelsList_));
  return;
}

int PreProcessCuda::generateVoxels(float *points, size_t points_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs)
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

  checkCudaErrors(generateVoxels_random_launch(points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask_, voxels_, stream_));

  checkCudaErrors(generateBaseFeatures_launch(mask_, voxels_,
      grid_y_size, grid_x_size,
      pillar_num,
      voxel_features,
      voxel_num,
      voxel_idxs, stream_));

  return 0;
}

int PreProcessCuda::generateFeatures(float* voxel_features,
          unsigned int* voxel_num,
          unsigned int* voxel_idxs,
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
      voxel_num,
      voxel_idxs,
      params,
      voxel_x, voxel_y, voxel_z,
      range_min_x, range_min_y, range_min_z,
      features, stream_));

  return 0;
}

