# Using GPU Acceleration
By migrating point cloud parsing tasks to the GPU, not only can CPU computing resources be freed up, but the speed and efficiency of point cloud data parsing can also be effectively improved to a certain extent.


## Preparation
Using GPU for point cloud parsing requires proper installation of graphics drivers and CUDA environment.
1. Enter `nvidia-smi` command in terminal to see current GPU information. If not installed, refer to: [NVIDIA-DRIVERS](https://www.nvidia.cn/drivers/)
2. Enter `nvcc -V` command in terminal to see current CUDA version. If not installed, refer to: [CUDA-DOWNLOADS](https://developer.nvidia.com/cuda-downloads)

#### 1 SDK Configuration

Please refer to the operations in **[Compile Macro Control](../docs/compile_macro_control_description.md)** to configure the macro `FIND_CUDA` to take effect, and add the `cuda` parameter when running.

#### 2 Parsing Configuration
Refer to **[How to Parse Lidar Data Online](../docs/parsing_lidar_data_online.md)** and **[How to Parse PCAP File Data Offline](../docs/parsing_pcap_file_data_offline.md)**

## Steps
### 1 Compilation
In the HesaiLidar_SDK_2.0 folder, open a terminal and execute the following commands:
```bash
cd HesaiLidar_SDK_2.0
mkdir -p build 
cd build
cmake .. -DFIND_CUDA=true
make
```

### 2 Run
After successful compilation, run the generated executable file in the build folder. To use GPU parsing, please add the `cuda` parameter:
```bash
./sample cuda
```


## Additional References
#### 1 If multiple CUDA versions are installed, you can specify the CUDA version path in [CMakeLists.txt](../CMakeLists.txt):

  ```cpp
   # if install different cuda version, set the cuda path, like cuda-11.4
   # set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4/)
   ```

#### 2 Control parameter for enabling GPU parsing in code

  Actually use `use_gpu` in `DriverParam` to control whether GPU parsing is enabled. In the `main` function of `test.cc`, control whether GPU parsing is enabled by assigning `param.use_gpu`