/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __POINTPILLAR_HPP__
#define __POINTPILLAR_HPP__

#include "lidar-voxelization.hpp"
#include "lidar-backbone.hpp"
#include "lidar-postprocess.hpp"
// #include "nms.hpp"

namespace pointpillar {
namespace lidar {

struct CoreParameter {
    VoxelizationParameter voxelization;
    std::string lidar_model;
    PostProcessParameter lidar_post;
};

class Core {
    public:
        virtual std::vector<BoundingBox> forward(const float *lidar_points, int num_points, void *stream) = 0;

        virtual void print() = 0;
        virtual void set_timer(bool enable) = 0;
};

std::shared_ptr<Core> create_core(const CoreParameter &param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __POINTPILLAR_HPP__
