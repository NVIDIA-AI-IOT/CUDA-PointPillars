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

#ifndef PARAMS_H_
#define PARAMS_H_

#define MAX_VOXELS 40000

class Params
{

public:

    // the max number of pillars to use
    const int max_num_pillars = MAX_VOXELS;//40000;

    // the max number of points per pillar
    const int max_num_points_per_pillar = 32;

    // the number of feature maps for pillar scatter
    const int num_feature_scatter = 64;

    // use in preprocess
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;

    // the x size of a pillar
    const float pillar_x_size = 0.16;

    // the y size of a pillar
    const float pillar_y_size = 0.16;

    // the z size of a pillar
    const float pillar_z_size = 4.0;

    // the min detection range in x direction
    const float min_x_range = 0.0;

    // the max detection range in x direction
    const float max_x_range = 69.12f;

    // the min detection range in y direction
    const float min_y_range = -39.68f;

    // the max detection range in y direction
    const float max_y_range = 39.68f;

    // the min detection range in z direction
    const float min_z_range = -3.0f;

    // the max detection range in z direction
    const float max_z_range = 1.0f;

    // x, y, z, intensity
    const int num_point_values = 4;

    // the number of classes to detect
    static const int num_classes = 3;

    const char *class_name[3] = {"Car", "Pedestrian", "Cyclist"};

    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;

    // the number of anchors
    static const int num_anchors = num_classes * 2;

    // the num of direction classes
    const int num_dir_classes = 2;

    // the direction offset
    const float dir_offset = 0.78539f;

    // the direction limited offset
    const float dir_limit_offset = 0.f;

    // the number of direction bins
    const int num_dir_bins = 2;

    // the x input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;

    // the y input size of the 2D backbone network
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;

    // the z input size of the 2D backbone network
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;

    // the x output size of the 2D backbone network
    const int feature_x_size = grid_x_size / 2;

    // the y output size of the 2D backbone network
    const int feature_y_size = grid_y_size / 2;

    // the score threshold
    const float score_thresh = 0.1f;

    // non maximum suppresion threshold
    const float nms_thresh = 0.01f;

    // anchors decode by (x, y, z, dir)
    static const int len_per_anchor = 4;
    const float anchors[num_anchors * len_per_anchor] = {
                                            3.9, 1.6, 1.56, 0,
                                            3.9, 1.6, 1.56, 1.57, 
                                            0.8, 0.6, 1.73, 0, 
                                            0.8, 0.6, 1.73, 1.57,
                                            1.76, 0.6, 1.73, 0, 
                                            1.76, 0.6, 1.73, 1.57
                                          };
    // anchors bottom height
    const float anchors_bottom_height[num_classes] = {-1.78, -0.6, -0.6};

    Params() {};
};

#endif
