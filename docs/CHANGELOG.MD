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
6. Support X86-64

#### Performance
```
| FP16              | GPU/ms |
| ----------------- | ------ |
| generateVoxels    | 0.22   |
| generateFeatures  | 0.21   |
| Inference         | 23.86  |
| Postprocessing    | 3.19   |
```
