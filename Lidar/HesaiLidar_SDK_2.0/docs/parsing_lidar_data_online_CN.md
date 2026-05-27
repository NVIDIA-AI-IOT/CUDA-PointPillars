# 在线解析雷达数据

## 准备

#### 以太网连接式激光雷达
```cpp
// 进入 test.cc 进行网络配置
// 使用宏LIDAR_PARSER_TEST，注释其他三个宏

#define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST
// #define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ...

#ifdef LIDAR_PARSER_TEST
  param.input_param.source_type = DATA_FROM_LIDAR;          // 设置数据来源为在线点云数据
  param.input_param.device_ip_address = "192.168.1.201";    // 激光雷达IP
  param.input_param.ptc_port = 9347;                        // PTC协议端口
  param.input_param.udp_port = 2368;                        // 点云数据目的端口
  param.input_param.multicast_ip_address = "";              // 可选项：如果目的IP为组播IP

  param.input_param.ptc_mode = PtcMode::tcp;                // PTC通信类型，如果不使用基于TLS/mTLS的PTCS，可忽略
  param.input_param.use_ptc_connected = true;               // 是否使用PTC连接
  param.input_param.correction_file_path = "/path/to/correction.csv";   // 角度修正文件（建议使用雷达自身的角度修正文件）
  param.input_param.firetimes_path = "path/to/firetimes.csv";           // 可选项：通道发光时序（发光时刻修正文件）

  param.input_param.use_someip = false;                     // 忽略
  param.input_param.host_ip_address = "";                   // 雷达ip地址，如果与点云源IP相同，则忽略
  param.input_param.fault_message_port = 0;                 // fault message目的端口，如果与udp_port相同，则忽略

  // PtcMode::tcp_ssl use                                   PTCS的一些配置，如果不使用则忽略
  param.input_param.certFile = "";
  param.input_param.privateKeyFile = "";
  param.input_param.caFile = "";
#endif
```

#### 串口连接式激光雷达 (JT16)
```cpp
// 进入 test.cc 进行串口配置
// 使用宏SERIAL_PARSER_TEST，注释其他三个宏

// #define LIDAR_PARSER_TEST
#define SERIAL_PARSER_TEST
// #define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ...

#ifdef SERIAL_PARSER_TEST
  param.input_param.source_type = DATA_FROM_SERIAL;         // 设置数据来源为串口数据
  param.input_param.rs485_com = "/dev/ttyUSB0";             // RS485 点云串口 (在Windows系统中是COM0)，串口号以实际为准
  param.input_param.rs232_com = "/dev/ttyUSB1";             // RS232 控制串口 (在Windows系统中是COM1)，串口号以实际为准
  param.input_param.point_cloud_baudrate = 3125000;         // 点云波特率
  param.input_param.correction_save_path = "";              // 雷达内部角度校准文件保存路径，如果设置232端口且成功从雷达获取角度校准文件
  param.input_param.correction_file_path = "/path/to/correction.csv"; // 角度修正文件（建议使用雷达自身的角度修正文件）
#endif
```

## 操作
```bash
# 1. 构建可执行示例程序 (从build文件夹下)：成功编译后，生成可执行程序
make -j$(nproc)

# 2. 执行示例程序：开始解析数据
./sample
```