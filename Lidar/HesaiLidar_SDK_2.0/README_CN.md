# HesaiLidar_SDK_2.0

[👉 English version](README.md)

## 1 检查适用性

### 1.1 雷达型号

| Pandar       | OT    | QT       | XT          | AT       | FT    | JT    |
|:-------------|:------|:---------|:------------|:---------|:------|:------|
| Pandar40P    | OT128 | PandarQT | PandarXT    | AT128E2X | FT120 | JT16  |
| Pandar64     | -     | QT128C2X | PandarXT-16 | AT128P   | -     | -     |
| Pandar128E3X | -     | -        | XT32M2X     | ATX      | -     | -     |

### 1.2 操作系统

- Ubuntu 16/18/20/22.04 
- Windows 10

### 1.3 编译器版本

Ubuntu
- Cmake 3.8.0 及以上
- G++ 7.5 及以上

Windows
- Cmake 3.8.0 及以上
- MSVC 2019 及以上

### 1.4 依赖项

- 如果使用点云可视化功能，需安装 `PCL`
- 如果解析 PCAP 文件，需安装 `libpcap`
- 如果需要使用基于TLS/mTLS的Ptcs通讯（部分雷达支持），需安装 `openssl`

<!-- - 如果解析雷达的点云修正文件，需安装 `libyaml`  // 解析ROS驱动中的config.yaml文件需要 -->

## 2 开始使用

### 2.1 克隆

```bash
git clone --recurse-submodules https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0.git
```

> 在Windows系统中，不建议将存储库下载为ZIP文件，因为它可能会由于符号链接问题而导致编译错误。

### 2.2 编译

#### 2.2.1 Ubuntu下的编译说明
```bash
# 0. 安装依赖项
sudo apt update && sudo apt install -y libpcl-dev libpcap-dev libyaml-cpp-dev openssl

# 1. 导航到源目录
cd HesaiLidar_SDK_2.0

# 2. 创建build目录并导航到build目录
mkdir -p build && cd build

# 3. 使用Cmake配置项目
#    - 添加 -DCMAKE_BUILD_TYPE=Release 用于优化编译
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. 编译SDK
#    - 使用 -j$(nproc) 来利用所有的CPU内核
make -j$(nproc)
```

#### 2.2.2 Windows下的编译说明
请参考 **[如何在Windows下编译SDK](docs/compile_on_windows_CN.md)**.

#### 2.2.3 删除对openssl库的依赖（不使用PTCS通信）
请参考 **[编译宏控制](docs/compile_macro_control_description_CN.md)** 中的操作，将宏 `WITH_PTCS_USE` 配置为失效即可。

## 3 应用指南

### 3.1 在线解析激光雷达数据
 请参考 **[如何在线解析激光雷达数据](docs/parsing_lidar_data_online_CN.md)**。

### 3.2 离线解析PCAP文件数据
请参考 **[如何离线解析PCAP文件数据](docs/parsing_pcap_file_data_offline_CN.md)**。

### 3.3 点云数据的可视化
请参考 **[如何可视化点云数据](docs/visualization_of_point_cloud_data_CN.md)**。

### 3.4 坐标转换
请参考 **[如何进行坐标转换](docs/coordinate_transformation_CN.md)**。

### 3.5 保存点云数据为PCD文件
请参考 **[如何将点云数据保存为PCD文件](docs/save_point_cloud_data_as_a_pcd_file_CN.md)**。

### 3.6 使用GPU加速
请参考 **[如何使用GPU加速优化性能](docs/use_gpu_acceleration_CN.md)**.

### 3.7 调用SDK API命令接口 （PTC通讯）
请参考 **[如何调用SDK API命令接口](docs/invoke_sdk_api_command_interface_CN.md)**.

### 3.8 常见故障处理（WARNING）
请参考 **[常见故障处理（WARNING）](docs/common_error_codes_CN.md)**.

### 3.9 丢包统计
请参考 **[如何进行丢包统计](docs/packet_loss_analysis_CN.md)**.

### 3.10 使用多线程加速解析
请参考 **[功能参数解释](docs/parameter_introduction_CN.md)** 中的`thread_num`配置，将其配置为 >1 的值
> 注意： 最大允许线程数为【CPU最大核数 - 2】，如果配置超出，会修改为该数字。多线程会占用更多CPU资源，请妥善配置。

### 3.11 在线解析多台雷达数据
进入 [multi_test.cc](./test/multi_test.cc) 中

解析参数配置参考 **[如何在线解析激光雷达数据](docs/parsing_lidar_data_online_CN.md)**

> 基本原理为使用多线程的方式启动两个SDK解析数据

### 3.12 PCAP或实时接收中包含多雷达数据，筛选并解析指定雷达数据

请参考 **[功能参数解释](docs/parameter_introduction_CN.md)** 中的 `device_udp_src_port` 和 `device_fault_port` 的说明

通过配置 `device_udp_src_port`(点云报文源端口号) 和`device_ip_address`(点云报文源IP) 开启点云报文筛选功能，只解析该源IP+源端口号的点云报文。

通过配置 `device_fault_port`(故障报文源端口号) 和`device_ip_address`(故障报文源IP) 开启故障报文筛选功能，只解析该源IP+源端口号的故障报文。

### 3.13 Pandar系列雷达，OT128，XT系列雷达，QT系列雷达获取每个点的具体时间戳(点云包时间戳 + 发光通道时间纠正)

请使用LidarPointXYZICRTT结构体声明HesaiLidarSdk，其中uint64_t timeSecond为秒时间部分，uint32_t timeNanosecond为纳秒时间部分。例如 `HesaiLidarSdk<LidarPointXYZICRTT> sample;`

### 3.14 初始化过程中，设置SDK接收点云的超时时间，设置PTC的超时时间

1. 设置SDK接收点云的超时时间

    请参考 **[功能参数解释](docs/parameter_introduction_CN.md)** 中的 `recv_point_cloud_timeout`，该参数默认为 `-1`，代表在初始化过程中，如果没有收到有效点云则会一直阻塞等待。当该参数配置为大于等于0时，则SDK会等待一段时间后初始化失败并退出。

2. 设置PTC超时时间
    
    请参考 **[功能参数解释](docs/parameter_introduction_CN.md)** 中的 `ptc_connect_timeout`, 该参数默认为 `-1`，代表在初始化过程中，如果是 `DATA_FROM_LIDAR` 模式，则会一直阻塞等待PTC连接。当该参数大于等于0时，则SDK会等待一段时间后报错连接超时，并继续初始化。

### 3.15 根据水平角和垂直角进行点云重排
请参考 **[点云重排功能](docs/point_cloud_rearrangement_function_CN.md)**

## 4 功能参数解释
请参考 **[功能参数解释](docs/parameter_introduction_CN.md)**.