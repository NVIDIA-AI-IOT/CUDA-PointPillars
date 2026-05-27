#!/bin/bash

# 构建 FP32 主干引擎，输入仍按原模型使用 FP16/Int32，输出保留 FP32
# 适合对比精度：输入保持与原推理一致（voxels 为 FP16），内部计算尽量 FP32

/usr/src/tensorrt/bin/trtexec \
  --onnx=./model.onnx \
  --saveEngine=./model/convnext_fp32.plan \
  --plugins=build/libpointpillar_core.so \
  --inputIOFormats=fp16:chw,int32:chw,int32:chw \
  --layerOutputTypes=cls_preds:fp32,box_preds:fp32,dir_cls_preds:fp32 \
  --noTF32 \
  --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed \
  > model/pointpillar_fp32.log 2>&1


