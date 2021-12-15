# PointPillars inference with TensorRT
This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created by [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and modified by onnx_graphsurgeon.

Inference has four parts:
generateVoxels: convert points cloud into voxels which has 4 channles
generateFeatures: convert voxels into feature maps which has 10 channles
Inference: convert feature maps to raw data of bounding box, class source and direction
Postprocessing: parse bounding box, class source and direction

## Data
The demo use the data from KITTI Dataset and more data can be downloaded following the linker
[GETTING_STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)

## Model
The onnx file can be converted from a model trainned by OpenPCDet with the tool in the demo.

## Build

### Prerequisites
To build the pointpillars inference, **TensorRT** with PillarScatter layer and **CUDA** are needed. PillarScatter layer plugin is already implemented as a plugin for TRT in the demo.

- Jetpack 4.5
- TensorRT v7.1.3
- CUDA-10.2 + cuDNN-8.0.0
- PCL is optinal to store pcd pointcloud file

### Compile
---

```shell
$ cd test
$ mkdir build
$ cd build
$ make -j$(nproc)
```

## Run
```shell
$ ./demo
```
## Enviroments

- Jetpack 4.5
- Cuda10.2 + cuDNN8.0.0 + TensorRT 7.1.3
- Nvidia Jetson AGX Xavier

#### Performance
- FP16
```
|                   | GPU/ms | 
| ----------------- | ------ |
| generateVoxels    | 0.22   |
| generateFeatures  | 0.21   |
| Inference         | 23.86  |
| Postprocessing    | 3.19   |
```
## Note
1. GPU processes all points at the same time and points selected form points cloud for a voxel randomly, so the output of generateVoxels has random value.
Because CPU will select the first 32 points, the output of generateVoxels by CPU has fixed value.

2. The demo will cache the onnx file to improve performance.
If a new onnx will be used, please remove the cache file in "./model"

3. MAX_VOXELS in params.h is used to allocate cache during inference.
Decrease the value to save memory.

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Autoware-AI/core_perception](https://github.com/Autoware-AI/core_perception)

