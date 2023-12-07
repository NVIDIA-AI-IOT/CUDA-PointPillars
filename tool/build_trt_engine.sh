#!/bin/bash
/usr/src/tensorrt/bin/trtexec --onnx=./model/pointpillar.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/pointpillar.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/pointpillar.8611.log 2>&1
