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

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/tensor.hpp"

namespace pointpillar {
namespace lidar {

class CoreImplement: public Core {
public:
    virtual ~CoreImplement() {
        if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
        if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
    }

    bool init(const CoreParameter& param) {
        lidar_voxelization_ = create_voxelization(param.voxelization);
        if (lidar_voxelization_ == nullptr) {
            printf("Failed to create lidar voxelization.\n");
            return false;
        }

        lidar_backbone_ = create_backbone(param.lidar_model);
            if (lidar_backbone_ == nullptr) {
            printf("Failed to create lidar backbone & head.\n");
            return false;
        }

        lidar_postprocess_ = create_postprocess(param.lidar_post);
        if (lidar_postprocess_ == nullptr) {
            printf("Failed to create lidar postprocess.\n");
            return false;
        }

        res_.reserve(100);

        capacity_points_ = 300000;
        bytes_capacity_points_ = capacity_points_ * param.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
        checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
        param_ = param;
        return true;
    }

    std::vector<BoundingBox> forward_only(const float *lidar_points, int num_points, void *stream) {
        int cappoints = static_cast<int>(capacity_points_);
        if (num_points > cappoints) {
            printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
        }

        num_points = std::min(cappoints, num_points);

        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
        checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

        this->lidar_voxelization_->forward(lidar_points_device_, num_points, _stream);
        this->lidar_backbone_->forward(this->lidar_voxelization_->features(), this->lidar_voxelization_->coords(), this->lidar_voxelization_->params(), _stream);
        this->lidar_postprocess_->forward(this->lidar_backbone_->cls(), this->lidar_backbone_->box(), this->lidar_backbone_->dir(), _stream);

        return this->lidar_postprocess_->bndBoxVec();
    }

    std::vector<BoundingBox> forward_timer(const float *lidar_points, int num_points, void *stream) {
        int cappoints = static_cast<int>(capacity_points_);
        if (num_points > cappoints) {
            printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
        }

        num_points = std::min(cappoints, num_points);

        printf("==================PointPillars===================\n");
        std::vector<float> times;
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        timer_.start(_stream);

        size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
        checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
        timer_.stop("[NoSt] CopyLidar");

        timer_.start(_stream);
        this->lidar_voxelization_->forward(lidar_points_device_, num_points, _stream);
        times.emplace_back(timer_.stop("Lidar Voxelization"));

        timer_.start(_stream);
        this->lidar_backbone_->forward(this->lidar_voxelization_->features(), this->lidar_voxelization_->coords(), this->lidar_voxelization_->params(), _stream);
        times.emplace_back(timer_.stop("Lidar Backbone & Head"));

        timer_.start(_stream);
        this->lidar_postprocess_->forward(this->lidar_backbone_->cls(), this->lidar_backbone_->box(), this->lidar_backbone_->dir(), _stream);
        times.emplace_back(timer_.stop("Lidar Decoder + NMS"));

        float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
        printf("Total: %.3f ms\n", total_time);
        printf("=============================================\n");
        return this->lidar_postprocess_->bndBoxVec();
    }

    virtual std::vector<BoundingBox> forward(const float *lidar_points, int num_points, void *stream) override {
        if (enable_timer_) {
            return this->forward_timer(lidar_points, num_points, stream);
        } else {
            return this->forward_only(lidar_points, num_points, stream);
        }
    }

    virtual void set_timer(bool enable) override { enable_timer_ = enable; }

    virtual void print() override {
        lidar_backbone_->print();
    }

private:
    CoreParameter param_;
    nv::EventTimer timer_;
    float* lidar_points_device_ = nullptr;
    float* lidar_points_host_ = nullptr;
    size_t capacity_points_ = 0;
    size_t bytes_capacity_points_ = 0;

    std::shared_ptr<Voxelization> lidar_voxelization_;
    std::shared_ptr<Backbone> lidar_backbone_;
    std::shared_ptr<PostProcess> lidar_postprocess_;

    bool enable_timer_ = false;

    std::vector<BoundingBox> res_;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar
