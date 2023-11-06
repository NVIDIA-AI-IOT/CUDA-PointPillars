#!/bin/bash
/usr/src/tensorrt/bin/trtexec --onnx=./model/backbone.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/backbone.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > workspace/backbone.8611.log 2>&1
