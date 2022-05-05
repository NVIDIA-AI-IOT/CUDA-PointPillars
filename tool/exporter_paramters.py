import numpy as np
from pcdet.config import cfg

License = '''/*
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
 */'''

def export_paramters(cfg):
  CLASS_NAMES = []
  CLASS_NUM = 0
  rangMinX = 0
  rangMinY = 0
  rangMinZ = 0
  rangMaxX = 0
  rangMaxY = 0
  rangMaxZ = 0
  VOXEL_SIZE = []
  MAX_POINTS_PER_VOXEL = 0
  MAX_NUMBER_OF_VOXELS = 0
  NUM_POINT_FEATURES = 0
  NUM_BEV_FEATURES = 0
  DIR_OFFSET = 0
  DIR_LIMIT_OFFSET = 0
  NUM_DIR_BINS = 0
  anchor_sizes = []
  anchor_bottom_heights = []
  SCORE_THRESH = 0
  NMS_THRESH = 0

  CLASS_NAMES = cfg.CLASS_NAMES
  CLASS_NUM = len(CLASS_NAMES)

  rangMinX = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0]
  rangMinY = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[1]
  rangMinZ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[2]
  rangMaxX = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3]
  rangMaxY = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[4]
  rangMaxZ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[5]

  for item in cfg.DATA_CONFIG.DATA_PROCESSOR:
    if (item.NAME == "transform_points_to_voxels") :
      VOXEL_SIZE = item.VOXEL_SIZE
      MAX_POINTS_PER_VOXEL = item.MAX_POINTS_PER_VOXEL
      MAX_NUMBER_OF_VOXELS = item.MAX_NUMBER_OF_VOXELS.test

  for item in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST :
    if (item.NAME == "gt_sampling") :
      NUM_POINT_FEATURES = item.NUM_POINT_FEATURES

  NUM_BEV_FEATURES = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
  DIR_OFFSET = cfg.MODEL.DENSE_HEAD.DIR_OFFSET
  DIR_LIMIT_OFFSET = cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET
  NUM_DIR_BINS = cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS

  for item in cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG :
    for anchor in np.array(item.anchor_sizes).flatten() :
      anchor_sizes.append(float(anchor))
    anchor_sizes.append(float(item.anchor_rotations[0]))
    for anchor in np.array(item.anchor_sizes).flatten() :
      anchor_sizes.append(float(anchor))
    anchor_sizes.append(float(item.anchor_rotations[1]))
    for anchor_height in np.array(item.anchor_bottom_heights).flatten() :
      anchor_bottom_heights.append(anchor_height)

  SCORE_THRESH = cfg.MODEL.POST_PROCESSING.SCORE_THRESH
  NMS_THRESH = cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH

  # dump paramters to params.h
  fo = open("params.h","w")
  fo.write(License+"\n")
  fo.write("#ifndef PARAMS_H_\n#define PARAMS_H_\n")
  fo.write("const int MAX_VOXELS = "+str(MAX_NUMBER_OF_VOXELS)+";\n")

  fo.write("class Params\n{\n  public:\n")

  fo.write("    static const int num_classes = "+str(CLASS_NUM)+";\n")
  class_names_list = "    const char *class_name [num_classes] = { "
  for CLASS_NAME in CLASS_NAMES :
    class_names_list = class_names_list + "\""+CLASS_NAME+"\","
  class_names_list = class_names_list + "};\n"
  fo.write(class_names_list)

  fo.write("    const float min_x_range = "+str(float(rangMinX))+";\n")
  fo.write("    const float max_x_range = "+str(float(rangMaxX))+";\n")
  fo.write("    const float min_y_range = "+str(float(rangMinY))+";\n")
  fo.write("    const float max_y_range = "+str(float(rangMaxY))+";\n")
  fo.write("    const float min_z_range = "+str(float(rangMinZ))+";\n")
  fo.write("    const float max_z_range = "+str(float(rangMaxZ))+";\n")

  fo.write("    // the size of a pillar\n")
  fo.write("    const float pillar_x_size = "+str(float(VOXEL_SIZE[0]))+";\n")
  fo.write("    const float pillar_y_size = "+str(float(VOXEL_SIZE[1]))+";\n")
  fo.write("    const float pillar_z_size = "+str(float(VOXEL_SIZE[2]))+";\n")

  fo.write("    const int max_num_points_per_pillar = "+str(MAX_POINTS_PER_VOXEL)+";\n")

  fo.write("    const int num_point_values = "+str(NUM_POINT_FEATURES)+";\n")
  fo.write("    // the number of feature maps for pillar scatter\n")
  fo.write("    const int num_feature_scatter = "+str(NUM_BEV_FEATURES)+";\n")

  fo.write("    const float dir_offset = "+str(float(DIR_OFFSET))+";\n")
  fo.write("    const float dir_limit_offset = "+str(float(DIR_LIMIT_OFFSET))+";\n")

  fo.write("    // the num of direction classes(bins)\n")
  fo.write("    const int num_dir_bins = "+str(NUM_DIR_BINS)+";\n")

  fo.write("    // anchors decode by (x, y, z, dir)\n")
  fo.write("    static const int num_anchors = num_classes * 2;\n")
  fo.write("    static const int len_per_anchor = 4;\n")

  anchor_str = "    const float anchors[num_anchors * len_per_anchor] = {\n"
  anchor_str += "      "
  count = 0
  for item in anchor_sizes :
    anchor_str = anchor_str + str(float(item)) +","
    count +=1
    if((count%4)==0) : anchor_str += "\n      "
  anchor_str = anchor_str + "};\n"
  fo.write(anchor_str)

  anchor_heights = "    const float anchor_bottom_heights[num_classes] = {"
  for item in anchor_bottom_heights :
    anchor_heights = anchor_heights + str(float(item)) +","
  anchor_heights = anchor_heights + "};\n"
  fo.write(anchor_heights)
  fo.write("    // the score threshold for classification\n")
  fo.write("    const float score_thresh = "+str(float(SCORE_THRESH))+";\n")
  fo.write("    const float nms_thresh = "+str(float(NMS_THRESH))+";\n")

  fo.write(
'''    const int max_num_pillars = MAX_VOXELS;
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;
    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;
    // the input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
    // the output size of the 2D backbone network
    const int feature_x_size = grid_x_size / 2;
    const int feature_y_size = grid_y_size / 2;\n'''
  )

  fo.write("    Params() {};\n};\n")
  fo.write("#endif\n")
  fo.close()

if __name__ == '__main__':
  export_paramters(cfg)
