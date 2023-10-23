#!/bin/bash
/usr/src/tensorrt/bin/trtexec --onnx=./model/pfe.onnx --fp16 --builderOptimizationLevel=4 --saveEngine=./model/pfe.8611.opt4.plan --outputIOFormats=fp16:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > pfe.8611.opt4.log 2>&1
/usr/src/tensorrt/bin/trtexec --onnx=./model/backbone.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/backbone.8611.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > backbone.8611.log 2>&1
