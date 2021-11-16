# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .detector3d_template import Detector3DTemplate
import sys


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, voxel_features, voxel_num_points, coords):
        self.batch_size = 1
        pillar_features = self.module_list[0](voxel_features, voxel_num_points, coords) #"PillarVFE"

        spatial_features = self.module_list[1](pillar_features, coords) #"PointPillarScatter"

        spatial_features_2d = self.module_list[2](spatial_features) #"BaseBEVBackbone"

        cls_preds, box_preds, dir_cls_preds = self.module_list[3](spatial_features_2d, self.batch_size) #"AnchorHeadSingle"

        cls_preds_normalized = False
        return cls_preds, box_preds, dir_cls_preds

