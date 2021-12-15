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

#include "params.h"

class PreProcessCuda {
  private:
    Params params_;
    unsigned int *mask_;
    float *voxels_;
    int *voxelsList_;
    float *params_cuda_;
    cudaStream_t stream_ = 0;

    //points cloud -> voxels (BEV) -> feature*4 by CPU
    int *coor_to_voxelidx_ = nullptr;

  public:
    PreProcessCuda(cudaStream_t stream_ = 0);
    ~PreProcessCuda();

    //points cloud -> voxels (BEV) -> feature*4 
    int generateVoxels(float *points, size_t points_size,
        unsigned int *pillar_num,
        float *voxel_features,
        float *voxel_num_points,
        float *coords);

    //feature*4 -> feature * 10 
    int generateFeatures(float* voxel_features,
          float* voxel_num_points,
          float* coords,
          unsigned int *params,
          float* features);

    //points cloud -> voxels (BEV) -> feature*4 by CPU
    int clearCacheCPU(void);
    void generateVoxels_cpu(float* points, size_t points_size,
        unsigned int* pillarCount,
        float* voxel_features,
        float* voxel_num_points,
        float* coords);

};

