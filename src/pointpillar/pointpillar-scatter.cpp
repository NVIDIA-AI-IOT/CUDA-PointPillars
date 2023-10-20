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

#include "pointpillar-scatter.hpp"

#include <iostream>
#include <cassert>
#include <cstring>

#include "pillarscatter-kernel.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::PPScatterPlugin;
using nvinfer1::plugin::PPScatterPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"PPScatterPlugin"};

// Static class fields initialization
PluginFieldCollection PPScatterPluginCreator::mFC{};
std::vector<PluginField> PPScatterPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

PPScatterPlugin::PPScatterPlugin(size_t h, size_t w)
  : feature_y_size_(h), feature_x_size_(w)
{
}

PPScatterPlugin::PPScatterPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    feature_y_size_ = readFromBuffer<size_t>(d);
    feature_x_size_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* PPScatterPlugin::clone() const noexcept
{
    auto* plugin = new PPScatterPlugin(feature_y_size_, feature_x_size_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs PPScatterPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == 0);
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(feature_y_size_);
    output.d[3] = exprBuilder.constant(feature_x_size_);
    return output;
}

bool PPScatterPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(nbInputs == 3);
    assert(nbOutputs == 1);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        return (inOut[0].type == nvinfer1::DataType::kFLOAT && in.type == nvinfer1::DataType::kFLOAT && in.format == TensorFormat::kLINEAR)
            || (inOut[0].type == nvinfer1::DataType::kHALF  && in.type == nvinfer1::DataType::kHALF  && in.format == TensorFormat::kHWC8);
    }
    return false;
}

void PPScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    return;
}

size_t PPScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int PPScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        int numFeatures = inputDesc[0].dims.d[1];
        
        nvinfer1::DataType inputType = inputDesc[0].type;

        auto coords_data = static_cast<const unsigned int *>(inputs[1]);
        auto params_data = static_cast<const unsigned int *>(inputs[2]);

        unsigned int featureY = feature_y_size_;
        unsigned int featureX = feature_x_size_;

        int status = -1;

        if(inputType == nvinfer1::DataType::kHALF){
            auto pillar_features_data = static_cast<const half *>(inputs[0]);
            auto spatial_feature_data = static_cast<half *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, numFeatures*featureY*featureX * sizeof(half), stream);
            status = pillarScatterHalfKernelLaunch(
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            assert(status == 0);
            return status;
        }
        else if(inputType == nvinfer1::DataType::kFLOAT){
            auto pillar_features_data = static_cast<const float *>(inputs[0]);
            auto spatial_feature_data = static_cast<float *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, numFeatures*featureY*featureX * sizeof(float), stream);
            status = pillarScatterFloatKernelLaunch(
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            assert(status == 0);
            return status;
        }
        else{
            assert(status == 0);
            return status;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return -1;
}

nvinfer1::DataType PPScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* PPScatterPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* PPScatterPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int PPScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int PPScatterPlugin::initialize() noexcept
{
    return 0;
}

void PPScatterPlugin::terminate() noexcept
{
}

size_t PPScatterPlugin::getSerializationSize() const noexcept
{
    return 3 * sizeof(size_t);
}

void PPScatterPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<size_t>(d, feature_y_size_);
    writeToBuffer<size_t>(d, feature_x_size_);
}

void PPScatterPlugin::destroy() noexcept
{
    delete this;
}

void PPScatterPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PPScatterPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

PPScatterPluginCreator::PPScatterPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dense_shape", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PPScatterPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* PPScatterPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* PPScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* PPScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int target_h = 0;
    int target_w = 0;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "dense_shape"))
        {
            const int* ts = static_cast<const int*>(fields[i].data);
            target_h = ts[0];
            target_w = ts[1];
        }
    }
    auto* plugin = new PPScatterPlugin(
        target_h,
        target_w
    );
    return plugin;
}

IPluginV2* PPScatterPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    auto* plugin = new PPScatterPlugin(serialData, serialLength);
    return plugin;
}

void PPScatterPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PPScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(PPScatterPluginCreator);
