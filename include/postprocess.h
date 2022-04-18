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
#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include <vector>
#include "kernel.h"

/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
*/
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
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

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred);

#endif
