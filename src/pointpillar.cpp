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
#include <iostream>
#include <fstream>
#include <vector>

#include "cuda_runtime.h"

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

#include "pointpillar.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

TRT::~TRT(void)
{
  context->destroy();
  engine->destroy();
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}

TRT::TRT(std::string modelFile, cudaStream_t stream):stream_(stream)
{
  std::string modelCache = modelFile + ".cache";
  std::fstream trtCache(modelCache, std::ifstream::in);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  if (!trtCache.is_open())
  {
	  std::cout << "Building TRT engine."<<std::endl;
    // define builder
    auto builder = (nvinfer1::createInferBuilder(gLogger_));

    // define network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = (builder->createNetworkV2(explicitBatch));

    // define onnxparser
    auto parser = (nvonnxparser::createParser(*network, gLogger_));
    if (!parser->parseFromFile(modelFile.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                  << std::endl;
        exit(-1);
    }

    // define config
    auto networkConfig = builder->createBuilderConfig();
#if defined (__arm64__) || defined (__aarch64__) 
    networkConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "Enable fp16!" << std::endl;
#endif
    // set max batch size
    builder->setMaxBatchSize(1);
    // set max workspace
    networkConfig->setMaxWorkspaceSize(size_t(1) << 30);

    engine = (builder->buildEngineWithConfig(*network, *networkConfig));

    if (engine == nullptr)
    {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto trtModelStream = (engine->serialize());
    std::fstream trtOut(modelCache, std::ifstream::out);
    if (!trtOut.is_open())
    {
       std::cout << "Can't store trt cache.\n";
       exit(-1);
    }

    trtOut.write((char*)trtModelStream->data(), trtModelStream->size());
    trtOut.close();
    trtModelStream->destroy();

    networkConfig->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

  } else {
	  std::cout << "load TRT cache."<<std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trtCache.seekg(0, trtCache.end);
    length = trtCache.tellg();
    trtCache.seekg(0, trtCache.beg);

    data = (char *)malloc(length);
    if (data == NULL ) {
       std::cout << "Can't malloc data.\n";
       exit(-1);
    }

    trtCache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(gLogger_);

    if (runtime == nullptr) {	  std::cout << "load TRT cache0."<<std::endl;
        std::cerr << ": runtime null!" << std::endl;
        exit(-1);
    }
    //plugin_ = nvonnxparser::createPluginFactory(gLogger_);
    engine = (runtime->deserializeCudaEngine(data, length, 0));
    if (engine == nullptr) {
        std::cerr << ": engine null!" << std::endl;
        exit(-1);
    }
    free(data);
    trtCache.close();
  }

  context = engine->createExecutionContext();

}

int TRT::doinfer(void**buffers)
{
  int status;

  status = context->enqueueV2(buffers, stream_, &start);

  if (!status)
  {
      return false;
  }

  return true;
}

PointPillar::PointPillar(std::string modelFile, cudaStream_t stream):stream_(stream)
{

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  pre_.reset(new PreProcessCuda(stream_));
  trt_.reset(new TRT(modelFile, stream_));
  post_.reset(new PostProcessCuda(stream_));

  //input of pre-process
  voxel_features_size = MAX_VOXELS * params_.max_num_points_per_pillar * 4 * sizeof(float);
  voxel_num_points_size = MAX_VOXELS * sizeof(float);
  coords_size = MAX_VOXELS* 4 * sizeof(float);

  checkCudaErrors(cudaMallocManaged((void **)&voxel_features, voxel_features_size));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_num_points, voxel_num_points_size));
  checkCudaErrors(cudaMallocManaged((void **)&coords, MAX_VOXELS* 4 * sizeof(float)));

  checkCudaErrors(cudaMemsetAsync(voxel_features, 0, voxel_features_size, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_num_points, 0, voxel_num_points_size, stream_));
  checkCudaErrors(cudaMemsetAsync(coords, 0, MAX_VOXELS* 4 * sizeof(float), stream_));


  //TRT-input
  features_input_size = MAX_VOXELS * params_.max_num_points_per_pillar * 10 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&features_input, features_input_size));
  checkCudaErrors(cudaMallocManaged((void **)&params_input, 5 * sizeof(unsigned int)));

  checkCudaErrors(cudaMemsetAsync(features_input, 0, features_input_size, stream_));
  checkCudaErrors(cudaMemsetAsync(params_input, 0, 5 * sizeof(float), stream_));


  //output of TRT -- input of post-process
  cls_size = params_.feature_x_size * params_.feature_y_size * params_.num_classes * params_.num_anchors * sizeof(float);
  box_size = params_.feature_x_size * params_.feature_y_size * params_.num_box_values * params_.num_anchors * sizeof(float);
  dir_cls_size = params_.feature_x_size * params_.feature_y_size * params_.num_dir_classes * params_.num_anchors * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&cls_output, cls_size));
  checkCudaErrors(cudaMallocManaged((void **)&box_output, box_size));
  checkCudaErrors(cudaMallocManaged((void **)&dir_cls_output, dir_cls_size));

  //output of post-process
  bndbox_size = (params_.feature_x_size * params_.feature_y_size * 9 + 1) * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&bndbox_output, bndbox_size));

  //res.resize(100);
  res.reserve(100);
}

PointPillar::~PointPillar(void)
{
  pre_.reset();
  trt_.reset();
  post_.reset();

  checkCudaErrors(cudaFree(voxel_features));
  checkCudaErrors(cudaFree(voxel_num_points));
  checkCudaErrors(cudaFree(coords));

  checkCudaErrors(cudaFree(features_input));
  checkCudaErrors(cudaFree(params_input));

  checkCudaErrors(cudaFree(cls_output));
  checkCudaErrors(cudaFree(box_output));
  checkCudaErrors(cudaFree(dir_cls_output));

  checkCudaErrors(cudaFree(bndbox_output));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}

int PointPillar::doinfer(void*points_data, unsigned int points_size, std::vector<Bndbox> &nms_pred)
{
#if PERFORMANCE_LOG
  float generateVoxelsTime = 0.0f;
  cudaEventRecord(start, stream_);
#endif

#if GENERATE_VOXELS_BY_CPU
  pre_->clearCacheCPU();
  pre_->generateVoxels_cpu((float*)points_data, points_size,
        params_input,
        voxel_features, 
        voxel_num_points, 
        coords);
  checkCudaErrors(cudaDeviceSynchronize());
#else
  pre_->generateVoxels((float*)points_data, points_size,
        params_input,
        voxel_features, 
        voxel_num_points, 
        coords);
#endif

#if PERFORMANCE_LOG
  cudaEventRecord(stop, stream_);
  checkCudaErrors(cudaDeviceSynchronize());
  cudaEventElapsedTime(&generateVoxelsTime, start, stop);
  unsigned int params_input_cpu[5];
  checkCudaErrors(cudaMemcpy(params_input_cpu, params_input, 5*sizeof(unsigned int), cudaMemcpyDefault));
  std::cout<<"find pillar_num: "<< params_input_cpu[4] <<std::endl;
#endif
/*
  unsigned int num_pillars = *pillar_num;
  params_input[0] = 1;
  params_input[1] = params_.num_feature_scatter;//featureNum;
  params_input[2] = params_.grid_y_size;//featureY;
  params_input[3] = params_.grid_x_size;//featureX;
  params_input[4] = num_pillars;//num_pillars;
*/
#if PERFORMANCE_LOG
  float generateFeaturesTime = 0.0f;
  cudaEventRecord(start, stream_);
#endif

  pre_->generateFeatures(voxel_features,
      voxel_num_points,
      coords,
      params_input,
      features_input);

#if PERFORMANCE_LOG
  cudaEventRecord(stop, stream_);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&generateFeaturesTime, start, stop);
#endif

#if PERFORMANCE_LOG
  float doinferTime = 0.0f;
  cudaEventRecord(start, stream_);
#endif

  void *buffers[] = {features_input, coords, params_input, cls_output, box_output, dir_cls_output };
  trt_->doinfer(buffers);
  checkCudaErrors(cudaMemsetAsync(params_input, 0, 5 * sizeof(unsigned int), stream_));

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop, stream_));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&doinferTime, start, stop));
#endif

#if PERFORMANCE_LOG
  float doPostprocessCudaTime = 0.0f;
  cudaEventRecord(start, stream_);
#endif

  post_->doPostprocessCuda(cls_output, box_output, dir_cls_output,
                          bndbox_output);

  cudaDeviceSynchronize();
  float obj_count = bndbox_output[0];

  int num_obj = static_cast<int>(obj_count);
  auto output = bndbox_output + 1;

  for (int i = 0; i < num_obj; i++) {
    auto Bb = Bndbox(output[i * 9],
                    output[i * 9 + 1], output[i * 9 + 2], output[i * 9 + 3],
                    output[i * 9 + 4], output[i * 9 + 5], output[i * 9 + 6],
                    static_cast<int>(output[i * 9 + 7]),
                    output[i * 9 + 8]);
    res.push_back(Bb);
  }


  nms_cpu(res, params_.nms_thresh, nms_pred);
  res.clear();

#if PERFORMANCE_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(stop, stream_));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&doPostprocessCudaTime, start, stop));
  std::cout<<"TIME: generateVoxels: "<< generateVoxelsTime <<" ms." <<std::endl;
  std::cout<<"TIME: generateFeatures: "<< generateFeaturesTime <<" ms." <<std::endl;
  std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
  std::cout<<"TIME: doPostprocessCuda: "<< doPostprocessCudaTime <<" ms." <<std::endl;
#endif
}

