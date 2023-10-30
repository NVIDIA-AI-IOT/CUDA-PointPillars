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

#ifndef __LIDAR_VOXELIZATION_HPP__
#define __LIDAR_VOXELIZATION_HPP__

#include <memory>
#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

struct VoxelizationParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Float3 voxel_size;
    nvtype::Int3 grid_size;
    int max_voxels;
    int max_points_per_voxel;
    int max_points;
    int num_feature;

    static nvtype::Int3 compute_grid_size(const nvtype::Float3& max_range, const nvtype::Float3& min_range,
                                        const nvtype::Float3& voxel_size);
};

class Voxelization {
  public:
    // points and voxels must be of half-float device pointer
    virtual void forward(const float *points, int num_points, void *stream = nullptr) = 0;

    virtual const nvtype::half* features() = 0;
    virtual const unsigned int* coords() = 0;
    virtual const unsigned int* params() = 0;
};

std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __LIDAR_VOXELIZATION_HPP__