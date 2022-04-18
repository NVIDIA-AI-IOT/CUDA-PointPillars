# PointPillars inference changelog

### 1.0(2021-09-06)
Release
#### Performance
```
| FP16              | GPU/ms |
| ----------------- | ------ |
| generateVoxels    | 0.22   |
| generateFeatures  | 0.21   |
| Inference         | 30.75  |
| Postprocessing    | 3.19   |
```

### 1.1.0(2021-12-15)
1. remove reduceMax and rely form onnx and add it into plugin
2. use "tool" to create params.h based on "pointpillar.yaml"
3. Add error checking
4. clean up code
5. provide tow methods for preprocessing
6. Support x86-64

#### Performance
```
| FP16              | GPU/ms |
| ----------------- | ------ |
| generateVoxels    | 0.22   |
| generateFeatures  | 0.21   |
| Inference         | 23.86  |
| Postprocessing    | 3.19   |
```

### 1.2.0(2022-04-18)
1. Add fp16 support for pillarScatter plugin
2. Rename onnx model input name to "voxels" & "voxel_num" & "voxel_idxs"
3. Change "voxel_num" & "voxel_idxs" type to unsigned int
4. Update performance data on Jetson AGX Orin (power mode + boosted clock)

#### Performance in FP16
```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| generateVoxels    | 0.14   |
| generateFeatures  | 0.15   |
| Inference         | 9.12   |
| Postprocessing    | 1.77   |
```
