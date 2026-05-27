# HesaiLidar_SDK_2.0

[👉 Chinese version](README_CN.md)

## 1 Check Compatibility

### 1.1 Lidar Models

| Pandar       | OT    | QT       | XT          | AT       | FT    | JT    |
|:-------------|:------|:---------|:------------|:---------|:------|:------|
| Pandar40P    | OT128 | PandarQT | PandarXT    | AT128E2X | FT120 | JT16  |
| Pandar64     | -     | QT128C2X | PandarXT-16 | AT128P   | -     | -     |
| Pandar128E3X | -     | -        | XT32M2X     | ATX      | -     | -     |

### 1.2 Operating Systems

- Ubuntu 16/18/20/22.04 
- Windows 10

### 1.3 Compiler Versions

Ubuntu
- Cmake 3.8.0 and above
- G++ 7.5 and above

Windows
- Cmake 3.8.0 and above
- MSVC 2019 and above

### 1.4 Dependencies

- If using point cloud visualization features, `PCL` installation is required
- If parsing PCAP files, `libpcap` installation is required
- If using TLS/mTLS-based Ptcs communication (supported by some lidars), `openssl` installation is required

<!-- - If parsing lidar point cloud correction files, `libyaml` installation is required  // Required for parsing config.yaml files in ROS driver -->

## 2 Getting Started

### 2.1 Clone

```bash
git clone --recurse-submodules https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0.git
```

> On Windows systems, downloading the repository as a ZIP file is not recommended as it may cause compilation errors due to symbolic link issues.

### 2.2 Compilation

#### 2.2.1 Compilation Instructions for Ubuntu
```bash
# 0. Install dependencies
sudo apt update && sudo apt install -y libpcl-dev libpcap-dev libyaml-cpp-dev openssl

# 1. Navigate to source directory
cd HesaiLidar_SDK_2.0

# 2. Create build directory and navigate to build directory
mkdir -p build && cd build

# 3. Configure project with Cmake
#    - Add -DCMAKE_BUILD_TYPE=Release for optimized compilation
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. Compile SDK
#    - Use -j$(nproc) to utilize all CPU cores
make -j$(nproc)
```

#### 2.2.2 Compilation Instructions for Windows
Please refer to **[How to Compile SDK on Windows](docs/compile_on_windows.md)**.

#### 2.2.3 Remove dependency on openssl library (not using PTCS communication)
Please refer to the operations in **[Compile Macro Control](docs/compile_macro_control_description.md)** to configure the macro `WITH_PTCS_USE` to be inactive.

## 3 Application Guide

### 3.1 Parse Lidar Data Online
Please refer to **[How to Parse Lidar Data Online](docs/parsing_lidar_data_online.md)**.

### 3.2 Parse PCAP File Data Offline
Please refer to **[How to Parse PCAP File Data Offline](docs/parsing_pcap_file_data_offline.md)**.

### 3.3 Point Cloud Data Visualization
Please refer to **[How to Visualize Point Cloud Data](docs/visualization_of_point_cloud_data.md)**.

### 3.4 Coordinate Transformation
Please refer to **[How to Perform Coordinate Transformation](docs/coordinate_transformation.md)**.

### 3.5 Save Point Cloud Data as PCD Files
Please refer to **[How to Save Point Cloud Data as PCD Files](docs/save_point_cloud_data_as_a_pcd_file.md)**.

### 3.6 Use GPU Acceleration
Please refer to **[How to Use GPU Acceleration for Performance Optimization](docs/use_gpu_acceleration.md)**.

### 3.7 Invoke SDK API Command Interface (PTC Communication)
Please refer to **[How to Invoke SDK API Command Interface](docs/invoke_sdk_api_command_interface.md)**.

### 3.8 Common Troubleshooting (WARNING)
Please refer to **[Common Troubleshooting (WARNING)](docs/common_error_codes.md)**.

### 3.9 Packet Loss Statistics
Please refer to **[How to Perform Packet Loss Statistics](docs/packet_loss_analysis.md)**.

### 3.10 Use Multi-threading to Accelerate Parsing
Please refer to the `thread_num` configuration in **[Functional Parameter Reference](docs/parameter_introduction.md)** and configure it to a value >1
> Note: The maximum allowed thread count is [CPU maximum cores - 2]. If configured beyond this, it will be modified to this number. Multi-threading will consume more CPU resources, please configure appropriately.

### 3.11 Parse Multiple Lidar Data Online
Navigate to [multi_test.cc](./test/multi_test.cc)

For parsing parameter configuration reference, see **[How to Parse Lidar Data Online](docs/parsing_lidar_data_online.md)**

> The basic principle is to use multi-threading to start two SDKs to parse data

### 3.12 Filter and Parse Specified Lidar Data from PCAP or Real-time Reception Containing Multi-lidar Data

Please refer to the descriptions of `device_udp_src_port` and `device_fault_port` in **[Functional Parameter Reference](docs/parameter_introduction.md)**

Enable point cloud packet filtering by configuring `device_udp_src_port` (point cloud packet source port number) and `device_ip_address` (point cloud packet source IP), parsing only point cloud packets from this source IP + source port number.

Enable fault message filtering by configuring `device_fault_port` (fault message source port number) and `device_ip_address` (fault message source IP), parsing only fault messages from this source IP + source port number.

### 3.13 Get Specific Timestamp for Each Point in Pandar Series, OT128, XT Series, QT Series Lidars (Point Cloud Packet Timestamp + Firing Channel Time Correction)

Please use the LidarPointXYZICRTT structure to declare HesaiLidarSdk, where uint64_t timeSecond is the seconds time part and uint32_t timeNanosecond is the nanoseconds time part. For example: `HesaiLidarSdk<LidarPointXYZICRTT> sample;`

### 3.14 Set SDK Point Cloud Reception Timeout and PTC Timeout During Initialization

1. Set SDK point cloud reception timeout

    Please refer to `recv_point_cloud_timeout` in **[Functional Parameter Reference](docs/parameter_introduction.md)**. This parameter defaults to `-1`, meaning that during initialization, if no valid point cloud is received, it will block and wait indefinitely. When this parameter is configured to >= 0, the SDK will wait for a period of time before initialization fails and exits.

2. Set PTC timeout
    
    Please refer to `ptc_connect_timeout` in **[Functional Parameter Reference](docs/parameter_introduction.md)**. This parameter defaults to `-1`, meaning that during initialization, if in `DATA_FROM_LIDAR` mode, it will block and wait for PTC connection indefinitely. When this parameter is >= 0, the SDK will wait for a period of time before reporting a connection timeout error and continuing initialization.

### 3.15 Point Cloud Rearrangement Based on Horizontal and Vertical Angles
Please refer to **[Point Cloud Rearrangement Function](docs/point_cloud_rearrangement_function.md)**

## 4 Functional Parameter Reference
Please refer to **[Functional Parameter Reference](docs/parameter_introduction.md)**.