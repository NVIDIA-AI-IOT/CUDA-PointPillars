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
#include <memory>

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

#include "postprocess.h"
#include "preprocess.h"

#define PERFORMANCE_LOG 1

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
};

class TRT {
  private:
    Params params_;

    cudaEvent_t start_, stop_;

    Logger gLogger_;
    nvinfer1::IExecutionContext *context_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;

    cudaStream_t stream_ = 0;
  public:
    TRT(std::string modelFile, cudaStream_t stream = 0);
    ~TRT(void);

    int doinfer(void**buffers);
};

class PointPillar {
  private:
    Params params_;

    cudaEvent_t start_, stop_;
    cudaStream_t stream_;

    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<TRT> trt_;
    std::shared_ptr<PostProcessCuda> post_;

    //input of pre-process
    float *voxel_features_ = nullptr;
    unsigned int *voxel_num_ = nullptr;
    unsigned int *voxel_idxs_ = nullptr;
    unsigned int *pillar_num_ = nullptr;

    unsigned int voxel_features_size_ = 0;
    unsigned int voxel_num_size_ = 0;
    unsigned int voxel_idxs_size_ = 0;

    //TRT-input
    float *features_input_ = nullptr;
    unsigned int *params_input_ = nullptr;
    unsigned int features_input_size_ = 0;

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

    std::vector<Bndbox> res_;

  public:
    PointPillar(std::string modelFile, cudaStream_t stream = 0);
    ~PointPillar(void);
    int doinfer(void*points, unsigned int point_size, std::vector<Bndbox> &res);
};

