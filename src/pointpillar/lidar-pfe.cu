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

#include <cuda_fp16.h>

#include <numeric>

#include "lidar-pfe.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace pointpillar {
namespace lidar {

class PFEImplement : public PFE {
public:
    virtual ~PFEImplement() {
        if (feature_) checkRuntime(cudaFree(feature_));
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        feature_dims_ = engine_->static_dims(1);
        int32_t volumn = std::accumulate(feature_dims_.begin(), feature_dims_.end(), 1, std::multiplies<int32_t>());
        feature_size_ = volumn * sizeof(half);
        checkRuntime(cudaMalloc(&feature_, feature_size_));

        return true;
    }

    virtual void print() override { engine_->print("Lidar PFE"); }

    virtual void forward(const nvtype::half* voxels, void* stream = nullptr) override {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
        engine_->forward({voxels, feature_}, _stream);
    }

    virtual nvtype::half* feature() override { return feature_; }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    nvtype::half *feature_ = nullptr;
    int feature_size_;
    std::vector<int> feature_dims_;
};

std::shared_ptr<PFE> create_pfe(const std::string& model) {
  std::shared_ptr<PFEImplement> instance(new PFEImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar