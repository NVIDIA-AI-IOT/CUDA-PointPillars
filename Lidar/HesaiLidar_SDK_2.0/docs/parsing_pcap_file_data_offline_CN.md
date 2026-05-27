# 离线解析PCAP点云数据

## 准备
```cpp
// 进入 test.cc 进行 PCAP 相关配置
// 使用宏PCAP_PARSER_TEST，注释其他三个宏

// #define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST
#define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ...

#ifdef PCAP_PARSER_TEST
  param.input_param.source_type = DATA_FROM_PCAP;                       // 设置数据来源为离线PCAP点云数据
  param.input_param.pcap_path = "path/to/pcap";                         // 离线PCAP点云数据路径
  param.input_param.correction_file_path = "/path/to/correction.csv";   // 校准文件（角度修正文件），建议使用雷达自身的校准文件
  param.input_param.firetimes_path = "path/to/firetimes.csv";           // 可选项：通道发光时序（发光时刻修正文件）

  param.decoder_param.pcap_play_synchronization = true;                 // 根据点云时间戳同步解析，模拟雷达实际频率
  param.decoder_param.pcap_play_in_loop = false;                        // 循环解析PCAP
#endif
```

## 操作
```bash
# 1. 构建可执行示例程序 (从build文件夹下)：成功编译后，生成可执行程序
make -j$(nproc)

# 2. 执行示例程序：开始解析数据
./sample
```