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

#include "postprocess.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

const float ThresHold = 1e-8;

inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_box2d(const Bndbox box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box.x;
    float center_y = box.y;
    float angle_cos = cos(-box.rt);
    float angle_sin = sin(-box.rt);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box.w / 2 + MARGIN && fabs(rot_y) < box.l / 2 + MARGIN);
}

bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( std::min(p0.x, p1.x) <= std::max(q0.x, q1.x) &&
          std::min(q0.x, q1.x) <= std::max(p0.x, p1.x) &&
          std::min(p0.y, p1.y) <= std::max(q0.y, q1.y) &&
          std::min(q0.y, q1.y) <= std::max(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > ThresHold) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

inline float box_overlap(const Bndbox &box_a, const Bndbox &box_b) {
    float a_angle = box_a.rt, b_angle = box_b.rt;
    float a_dx_half = box_a.w / 2, b_dx_half = box_b.w / 2, a_dy_half = box_a.l / 2, b_dy_half = box_b.l / 2;
    float a_x1 = box_a.x - a_dx_half, a_y1 = box_a.y - a_dy_half;
    float a_x2 = box_a.x + a_dx_half, a_y2 = box_a.y + a_dy_half;
    float b_x1 = box_b.x - b_dx_half, b_y1 = box_b.y - b_dy_half;
    float b_x2 = box_b.x + b_dx_half, b_y2 = box_b.y + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {box_a.x, box_a.y};
    float2 center_b = float2 {box_b.x, box_b.y};

    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }
    return fabs(area) / 2.0;
}

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred)
{
    std::sort(bndboxes.begin(), bndboxes.end(),
              [](Bndbox boxes1, Bndbox boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(bndboxes.size(), 0);
    for (size_t i = 0; i < bndboxes.size(); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        nms_pred.emplace_back(bndboxes[i]);
        for (size_t j = i + 1; j < bndboxes.size(); j++) {
            if (suppressed[j] == 1) {
                continue;
            }

            float sa = bndboxes[i].w * bndboxes[i].l;
            float sb = bndboxes[j].w * bndboxes[j].l;
            float s_overlap = box_overlap(bndboxes[i], bndboxes[j]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);

            if (iou >= nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return 0;
}

PostProcessCuda::PostProcessCuda(cudaStream_t stream)
{
  stream_ = stream;

  checkCudaErrors(cudaMalloc((void **)&anchors_, params_.num_anchors * params_.len_per_anchor * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&anchor_bottom_heights_, params_.num_classes * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&object_counter_, sizeof(int)));

  checkCudaErrors(cudaMemcpyAsync(anchors_, params_.anchors,
        params_.num_anchors * params_.len_per_anchor * sizeof(float), cudaMemcpyDefault, stream_));
  checkCudaErrors(cudaMemcpyAsync(anchor_bottom_heights_, params_.anchor_bottom_heights,
                     params_.num_classes * sizeof(float), cudaMemcpyDefault, stream_));
  checkCudaErrors(cudaMemsetAsync(object_counter_, 0, sizeof(int), stream_));
  return;
}

PostProcessCuda::~PostProcessCuda()
{
  checkCudaErrors(cudaFree(anchors_));
  checkCudaErrors(cudaFree(anchor_bottom_heights_));
  checkCudaErrors(cudaFree(object_counter_));
  return;
}

int PostProcessCuda::doPostprocessCuda(const float *cls_input, float *box_input, const float *dir_cls_input,
                                        float *bndbox_output)
{
  checkCudaErrors(cudaMemsetAsync(object_counter_, 0, sizeof(int)));
  checkCudaErrors(postprocess_launch(cls_input,
                     box_input,
                     dir_cls_input,
                     anchors_,
                     anchor_bottom_heights_,
                     bndbox_output,
                     object_counter_,
                     params_.min_x_range,
                     params_.max_x_range,
                     params_.min_y_range,
                     params_.max_y_range,
                     params_.feature_x_size,
                     params_.feature_y_size,
                     params_.num_anchors,
                     params_.num_classes,
                     params_.num_box_values,
                     params_.score_thresh,
                     params_.dir_offset,
                     stream_
                     ));
  return 0;
}
