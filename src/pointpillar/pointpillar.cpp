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

#include "pointpillar.hpp"

#include <vector>

#include "common/check.hpp"
#include "common/timer.hpp"

namespace pointpillar {
namespace lidar {

class CoreImplement: public Core {
    public:
        virtual ~CoreImplement() {
            if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
            if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));

            if (cls_output_) checkRuntime(cudaFree(cls_output_));
            if (box_output_) checkRuntime(cudaFree(box_output_));
            if (dir_cls_output_) checkRuntime(cudaFree(dir_cls_output_));
            if (bndbox_output_) checkRuntime(cudaFree(bndbox_output_));
        }

        bool init(const CoreParameter& param) {
            lidar_voxelization_ = lidar::create_voxelization(param.voxelization);
            if (lidar_voxelization_ == nullptr) {
                printf("Failed to create lidar voxelization.\n");
                return false;
            }

            lidar_pfe_ = lidar::create_pfe(param.pfe_model);
                if (lidar_pfe_ == nullptr) {
                printf("Failed to create lidar pfe.\n");
                return false;
            }

            lidar_backbone_head_ = lidar::create_backbone_head_(param.lidar_model);
                if (lidar_backbone_head_ == nullptr) {
                printf("Failed to create lidar backbone & head.\n");
                return false;
            }

            lidar_postprocess_ = lidar::create_postprocess(param.lidar_post);
            if (lidar_postprocess_ == nullptr) {
                printf("Failed to create lidar postprocess.\n");
                return false;
            }

            //output of TRT -- input of post-process
            cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_classes * params_.num_anchors * sizeof(float);
            box_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_box_values * params_.num_anchors * sizeof(float);
            dir_cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_dir_bins * params_.num_anchors * sizeof(float);
            checkRuntime(cudaMallocManaged((void **)&cls_output_, cls_size_));
            checkRuntime(cudaMallocManaged((void **)&box_output_, box_size_));
            checkRuntime(cudaMallocManaged((void **)&dir_cls_output_, dir_cls_size_));

            //output of post-process
            bndbox_size_ = (params_.feature_x_size * params_.feature_y_size * params_.num_anchors * 9 + 1) * sizeof(float);
            checkRuntime(cudaMallocManaged((void **)&bndbox_output_, bndbox_size_));

            res_.reserve(100);
            return;
        }

    private:
        CoreParameter param_;
        nv::EventTimer timer_;
        nvtype::half* lidar_points_device_ = nullptr;
        nvtype::half* lidar_points_host_ = nullptr;
        size_t capacity_points_ = 0;
        size_t bytes_capacity_points_ = 0;

        std::shared_ptr<lidar::Voxelization> lidar_voxelization_;
        std::shared_ptr<lidar::PFE> lidar_pfe_;
        std::shared_ptr<lidar::BackboneHead> lidar_backbone_head_;
        std::shared_ptr<lidar::PostProcess> lidar_postprocess_;

        bool enable_timer_ = false;

        //output of TRT -- input of post-process
        float *cls_output_ = nullptr;
        float *box_output_ = nullptr;
        float *dir_cls_output_ = nullptr;
        unsigned int cls_size_;
        unsigned int box_size_;
        unsigned int dir_cls_size_;

        //output of post-process
        float *bndbox_output_ = nullptr;
        unsigned int bndbox_size_ = 0;

        std::vector<postprocess::BoundingBox> res_;
};

int PointPillar::doinfer(void*points_data, unsigned int points_size, std::vector<Bndbox> &nms_pred)
{
    void *buffers[] = {features_input_, voxel_idxs_, params_input_, cls_output_, box_output_, dir_cls_output_};
    trt_->doinfer(buffers);

    post_->doPostprocessCuda(cls_output_, box_output_, dir_cls_output_,
                            bndbox_output_);
    checkRuntime(cudaDeviceSynchronize());
    float obj_count = bndbox_output_[0];

    int num_obj = static_cast<int>(obj_count);
    auto output = bndbox_output_ + 1;

    for (int i = 0; i < num_obj; i++) {
        auto Bb = Bndbox(output[i * 9],
                        output[i * 9 + 1], output[i * 9 + 2], output[i * 9 + 3],
                        output[i * 9 + 4], output[i * 9 + 5], output[i * 9 + 6],
                        static_cast<int>(output[i * 9 + 7]),
                        output[i * 9 + 8]);
        res_.push_back(Bb);
    }

    nms_cpu(res_, params_.nms_thresh, nms_pred);
    res_.clear();

    return 0;
}

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar
