# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and modified with onnx_graphsurgeon.

Overall inference has four phases:

- Convert points cloud into 4-channle voxels
- Extend 4-channel voxels to 10-channel voxel features
- Run TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data

The demo use the velodyne data from KITTI Dataset.
The onnx file can be converted from [pre-trained model](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) with given script under "./tool".

### Prerequisites

To build the pointpillars inference, **TensorRT** with PillarScatter layer and **CUDA** are needed. PillarScatter layer plugin is already implemented as a plugin for TRT in the demo.

## Environments

- Nvidia Jetson Xavier/Orin + Jetpack 5.0
- CUDA 11.4 + cuDNN 8.3.2 + TensorRT 8.4.0

### Compile && Run

```shell
$ sudo apt-get install git-lfs
$ git lfs install
$ git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git && cd CUDA-PointPillars
$ mkdir build && cd build
$ cmake .. && make -j$(nproc)
$ ./demo
```

#### Performance in FP16

Set Jetson to power mode with "sudo nvpmodel -m 0 && sudo jetson_clocks"

```
| Function(unit:ms) | Xavier | Orin   |
| ----------------- | ------ | ------ |
| GenerateVoxels    | 0.29   | 0.14   |
| GenerateFeatures  | 0.31   | 0.15   |
| Inference         | 20.21  | 9.12   |
| Postprocessing    | 3.38   | 1.77   |
| Overall           | 24.19  | 11.18  |
```

3D detection performance of moderate difficulty on the val set of KITTI dataset.

```
|                   | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 
| ----------------- | --------| -------------- | ------------ |
| CUDA-PointPillars | 77.02   | 51.65          | 62.24        |
| OpenPCDet         | 77.28   | 52.29          | 62.68        |
```

## Note

- GenerateVoxels has random output since GPU processes all points simultaneously while points selection for a voxel is random.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".
- MAX_VOXELS in params.h is used to allocate cache during inference. Decrease the value to save memory.

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
