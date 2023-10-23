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

#include "lidar-postprocess.hpp"

#include <algorithm>
#include <math.h>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace pointpillar {
namespace lidar {

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void postprocess_kernal(const float *cls_input,
                                        float *box_input,
                                        const float *dir_cls_input,
                                        float *anchors,
                                        float *anchor_bottom_heights,
                                        float *BoundingBox_output,
                                        int *object_counter,
                                        const float min_x_range,
                                        const float max_x_range,
                                        const float min_y_range,
                                        const float max_y_range,
                                        const int feature_x_size,
                                        const int feature_y_size,
                                        const int num_anchors,
                                        const int num_classes,
                                        const int num_box_values,
                                        const float score_thresh,
                                        const float dir_offset)
{
  int loc_index = blockIdx.x;
  int ith_anchor = threadIdx.x;
  if (ith_anchor >= num_anchors)
  {
      return;
  }
  int col = loc_index % feature_x_size;
  int row = loc_index / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
  int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
  float dev_cls[2] = {-1, 0};

  const float *scores = cls_input + cls_offset;
  float max_score = sigmoid(scores[0]);
  int cls_id = 0;
  for (int i = 1; i < num_classes; i++) {
    float cls_score = sigmoid(scores[i]);
    if (cls_score > max_score) {
      max_score = cls_score;
      cls_id = i;
    }
  }
  dev_cls[0] = static_cast<float>(cls_id);
  dev_cls[1] = max_score;

  if (dev_cls[1] >= score_thresh)
  {
    int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;
    float *anchor_ptr = anchors + ith_anchor * 4;
    float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[ith_anchor / 2];
    float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
    float *box_encodings = box_input + box_offset;

    float xa = anchor[0];
    float ya = anchor[1];
    float za = anchor[2];
    float dxa = anchor[3];
    float dya = anchor[4];
    float dza = anchor[5];
    float ra = anchor[6];
    float diagonal = sqrtf(dxa * dxa + dya * dya);
    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * dza + za;
    box_encodings[3] = expf(box_encodings[3]) * dxa;
    box_encodings[4] = expf(box_encodings[4]) * dya;
    box_encodings[5] = expf(box_encodings[5]) * dza;
    box_encodings[6] = box_encodings[6] + ra;

    float yaw;
    int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
    float period = 2 * M_PI / 2;
    float val = box_input[box_offset + 6] - dir_offset;
    float dir_rot = val - floor(val / (period + 1e-8) + 0.f) * period;
    yaw = dir_rot + dir_offset + period * dir_label;

    int resCount = (int)atomicAdd(object_counter, 1);
    BoundingBox_output[0] = resCount+1;
    float *data = BoundingBox_output + 1 + resCount * 9;
    data[0] = box_input[box_offset];
    data[1] = box_input[box_offset + 1];
    data[2] = box_input[box_offset + 2];
    data[3] = box_input[box_offset + 3];
    data[4] = box_input[box_offset + 4];
    data[5] = box_input[box_offset + 5];
    data[6] = yaw;
    data[7] = dev_cls[0];
    data[8] = dev_cls[1];
  }
}

cudaError_t postprocess_launch(const float *cls_input,
                      float *box_input,
                      const float *dir_cls_input,
                      float *anchors,
                      float *anchor_bottom_heights,
                      float *BoundingBox_output,
                      int *object_counter,
                      const float min_x_range,
                      const float max_x_range,
                      const float min_y_range,
                      const float max_y_range,
                      const int feature_x_size,
                      const int feature_y_size,
                      const int num_anchors,
                      const int num_classes,
                      const int num_box_values,
                      const float score_thresh,
                      const float dir_offset,
                      cudaStream_t stream)
{
  int bev_size = feature_x_size * feature_y_size;
  dim3 threads (num_anchors);
  dim3 blocks (bev_size);

  postprocess_kernal<<<blocks, threads, 0, stream>>>
                (cls_input,
                 box_input,
                 dir_cls_input,
                 anchors,
                 anchor_bottom_heights,
                 BoundingBox_output,
                 object_counter,
                 min_x_range,
                 max_x_range,
                 min_y_range,
                 max_y_range,
                 feature_x_size,
                 feature_y_size,
                 num_anchors,
                 num_classes,
                 num_box_values,
                 score_thresh,
                 dir_offset);
  return cudaGetLastError();
}

class PostProcessImplement : public PostProcess {
public:
    virtual ~PostProcessImplement() {
        if (bndbox_) checkRuntime(cudaFree(bndbox_));
        if (anchors_) checkRuntime(cudaFree(anchors_));
        if (anchor_bottom_heights_) checkRuntime(cudaFree(anchor_bottom_heights_));
        if (object_counter_) checkRuntime(cudaFree(object_counter_));
    }

    virtual bool init(const PostProcessParameter& param) {
        param_ = param;

        bndbox_size_ = (param_.feature_size.x * param_.feature_size.y * param_.num_anchors * 9 + 1) * sizeof(float);
        checkRuntime(cudaMallocManaged((void **)&bndbox_, bndbox_size_));
    
        checkRuntime(cudaMalloc((void **)&anchors_, param_.num_anchors * param_.len_per_anchor * sizeof(float)));
        checkRuntime(cudaMalloc((void **)&anchor_bottom_heights_, param_.num_classes * sizeof(float)));
        checkRuntime(cudaMalloc((void **)&object_counter_, sizeof(int)));

        checkRuntime(cudaMemcpy(anchors_, param_.anchors, param_.num_anchors * param_.len_per_anchor * sizeof(float), cudaMemcpyDefault));
        checkRuntime(cudaMemcpy(anchor_bottom_heights_, &param_.anchor_bottom_heights, param_.num_classes * sizeof(float), cudaMemcpyDefault));

        return true;
    }

    virtual void forward(const float* cls, const float* box, const float* dir, void* stream) override {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);

        checkRuntime(cudaMemsetAsync(object_counter_, 0, sizeof(int), _stream));
        checkRuntime(postprocess_launch((float *)cls,
                                        (float *)box,
                                        (float *)dir,
                                        anchors_,
                                        anchor_bottom_heights_,
                                        bndbox_,
                                        object_counter_,
                                        param_.min_range.x,
                                        param_.max_range.x,
                                        param_.min_range.y,
                                        param_.max_range.y,
                                        param_.feature_size.x,
                                        param_.feature_size.y,
                                        param_.num_anchors,
                                        param_.num_classes,
                                        param_.num_box_values,
                                        param_.score_thresh,
                                        param_.dir_offset,
                                        _stream
                                        ));
    }

    virtual const float *bndbox() override { return bndbox_; }

private:
    PostProcessParameter param_;
    float *anchors_;
    float *anchor_bottom_heights_;
    int *object_counter_;

    float *bndbox_ = nullptr;
    unsigned int bndbox_size_ = 0;
};

std::shared_ptr<PostProcess> create_postprocess(const PostProcessParameter& param) {
  std::shared_ptr<PostProcessImplement> instance(new PostProcessImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar