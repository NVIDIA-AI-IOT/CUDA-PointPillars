# 使用GPU解析
通过将点云解析任务迁移到GPU上，不仅可以释放CPU的计算资源，在一定程度上还能有效提升点云数据解析的速度与效率。


## 准备
使用GPU解析点云需要安装好显卡驱动及CUDA环境。
1. 终端输入`nvidia-smi`命令可以看到当前GPU信息，若未安装可参考：[NVIDIA-DRIVERS](https://www.nvidia.cn/drivers/)
2. 终端输入`nvcc -V`命令可以看到当前CUDA版本，若未安装可参考：[CUDA-DOWNLOADS](https://developer.nvidia.com/cuda-downloads)

#### 1 SDK配置

请参考 **[编译宏控制](../docs/compile_macro_control_description_CN.md)** 中的操作，将宏 `FIND_CUDA` 配置为生效，运行时添加 `cuda` 参数。

#### 2 解析配置
参考 **[如何在线解析激光雷达数据](../docs/parsing_lidar_data_online_CN.md)** 和 **[如何离线解析PCAP文件数据](../docs/parsing_pcap_file_data_offline_CN.md)**

## 操作
### 1 编译
在HesaiLidar_SDK_2.0文件夹下，启动Terminal终端，执行以下指令。
```bash
cd HesaiLidar_SDK_2.0
mkdir -p build 
cd build
cmake .. -DFIND_CUDA=true
make
```

### 2 运行
成功编译后，在build文件夹下运行生成的可执行文件。如需使用GPU解析，请添加 `cuda` 参数：
```bash
./sample cuda
```


## 更多参考
#### 1 如果安装了多个CUDA版本，可以在[CMakeLists.txt](../CMakeLists.txt) 指定CUDA版本路径:

  ```cpp
   # if install different cuda version, set the cuda path, like cuda-11.4
   # set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4/)
   ```

#### 2 代码中是否开启gpu解析的控制参数

  实际使用 `DriverParam` 中的 `use_gpu` 来控制gpu解析是否开启，在 `test.cc` 的 `main` 函数中，通过赋值 `param.use_gpu` 控制gpu解析是否开启