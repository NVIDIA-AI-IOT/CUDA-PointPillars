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

#include "nms.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <math.h>

namespace pointpillar {
namespace lidar {

const float ThresHold = 1e-8;

inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_box2d(const BoundingBox box, const float2 p) {
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

inline float box_overlap(const BoundingBox &box_a, const BoundingBox &box_b) {
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

int nms_cpu(std::vector<BoundingBox> BoundingBoxes, const float nms_thresh, std::vector<BoundingBox> &nms_pred)
{
    std::sort(BoundingBoxes.begin(), BoundingBoxes.end(),
              [](BoundingBox boxes1, BoundingBox boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(BoundingBoxes.size(), 0);
    for (size_t i = 0; i < BoundingBoxes.size(); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        nms_pred.emplace_back(BoundingBoxes[i]);
        for (size_t j = i + 1; j < BoundingBoxes.size(); j++) {
            if (suppressed[j] == 1) {
                continue;
            }

            float sa = BoundingBoxes[i].w * BoundingBoxes[i].l;
            float sb = BoundingBoxes[j].w * BoundingBoxes[j].l;
            float s_overlap = box_overlap(BoundingBoxes[i], BoundingBoxes[j]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);

            if (iou >= nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return 0;
}

};  // namespace lidar
};  // namespace pointpillar