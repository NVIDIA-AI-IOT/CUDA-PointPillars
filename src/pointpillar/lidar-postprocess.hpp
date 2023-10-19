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

#ifndef __LIDAR_POSTPROCESS_HPP__
#define __LIDAR_POSTPROCESS_HPP__

#include <vector>
#include "kernel.h"

#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

struct BoundingBox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    BoundingBox(){};
    BoundingBox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

class PostProcessCuda {
  private:
    Params params_;
    float *anchors_;
    float *anchor_bottom_heights_;
    int *object_counter_;
    cudaStream_t stream_ = 0;
  public:
    PostProcessCuda(cudaStream_t stream = 0);
    ~PostProcessCuda();

    int doPostprocessCuda(const float *cls_input, float *box_input, const float *dir_cls_input, float *bndbox_output);
};

int nms_cpu(std::vector<BoundingBox> bndboxes, const float nms_thresh, std::vector<BoundingBox> &nms_pred);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __LIDAR_POSTPROCESS_HPP__
