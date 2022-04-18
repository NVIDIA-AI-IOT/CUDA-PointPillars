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

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void postprocess_kernal(const float *cls_input,
                                        float *box_input,
                                        const float *dir_cls_input,
                                        float *anchors,
                                        float *anchor_bottom_heights,
                                        float *bndbox_output,
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
    bndbox_output[0] = resCount+1;
    float *data = bndbox_output + 1 + resCount * 9;
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
                      float *bndbox_output,
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
                 bndbox_output,
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

