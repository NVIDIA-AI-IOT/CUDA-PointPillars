# 基于 TensorRT 的 PointPillars 推理

本仓库提供基于 TensorRT 的 [PointPillars](https://arxiv.org/abs/1812.05784) 推理源码与模型。

推理流程概览：

- 将点云体素化为 10 通道特征
- 运行 TensorRT 引擎，得到检测特征
- 解析检测特征并执行 NMS（非极大值抑制）

## 环境准备

### 模型与数据

项目提供 [Dockerfile](docker/Dockerfile) 便于配置环境。安装 nvidia-docker 后，执行以下命令构建镜像：

```shell
cd docker && docker build . -t pointpillar
```

构建完成后，使用以下命令启动容器：

```shell
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```

导出模型时，先克隆 OpenPCDet 并安装自定义 CUDA 扩展：

```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```

将 [预训练权重（PTM）](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) 下载至 `ckpts/`，再执行：

```shell
python tool/export_onnx.py --ckpt ckpts/checkpoint_epoch_80.pth --out_dir model
```

在 KITTI 上评估可执行：

```shell
sh tool/evaluate_kitti_val.sh
```

数据集准备详见 [KITTI 评估说明](tool/eval/README.md)。

### 运行环境

- Nvidia Jetson Orin + CUDA (11.8) + cuDNN (8.9.4.25-1+cuda12.2) + TensorRT (8.6.2.3-1+cuda12.2)

## 构建与运行

```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git
cd CUDA-PointPillars && . tool/environment.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar ../data/ ../data/ --timer
```

## FP16 性能与评测指标

**耗时（FP16，KITTI 训练集，7481 样本，单位：ms）**

| 阶段           | Jetson Orin |
| -------------- | ----------- |
| 体素化         | 0.18        |
| 骨干网络与检测头 | 4.87        |
| 解码与 NMS     | 1.79        |
| 合计           | 6.84        |

**精度（3D Moderate，KITTI 验证集，3769 样本）**

| 方法              | 车辆 @R11 | 行人 @R11 | 骑行者 @R11 |
| ----------------- | --------- | --------- | ----------- |
| CUDA-PointPillars | 77.00     | 52.50     | 62.26       |
| OpenPCDet         | 77.28     | 52.29     | 62.68       |

## 注意事项

- 体素化阶段输出可能略有差异：GPU 并行处理全部点云，单个体素内采样哪些点是随机的。

## 参考资料

- [使用 NVIDIA CUDA-PointPillars 在点云中检测目标](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)（英文）
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)（论文，英文）
