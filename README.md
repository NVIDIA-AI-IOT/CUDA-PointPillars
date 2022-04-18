# PointPillars inference with TensorRT
This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created by [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and modified by onnx_graphsurgeon.

Inference has four phases:
- Convert points cloud into 4 channles voxels
- Extend raw c=4 voxels to c=10 voxel features
- Run TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data
The demo use the velodyne data from KITTI Dataset.
The onnx file can be converted from [pre-trained model](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) with given script.

### Prerequisites
To build the pointpillars inference, **TensorRT** with PillarScatter layer and **CUDA** are needed. PillarScatter layer plugin is already implemented as a plugin for TRT in the demo.

## Enviroments
- Nvidia Jetson AGX Orin
- Jetpack 5.0
- CUDA 11.4
- TensorRT 8.4.0
- cuDNN 8.3.2
- [optional] PCL

### Compile && Run
```shell
$ mkdir build && cd build
$ cmake .. && make -j$(nproc)
$ ./demo
```

#### Performance in FP16
```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| generateVoxels    | 0.14   |
| generateFeatures  | 0.15   |
| Inference         | 9.12   |
| Postprocessing    | 1.77   |
```

## Note
- GenerateVoxels has random output since GPU processes all points simultaneously while points selection for a voxel is random.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".
- MAX_VOXELS in params.h is used to allocate cache during inference. Decrease the value to save memory.

## References
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Autoware-AI/core_perception](https://github.com/Autoware-AI/core_perception)
