# 可视化点云数据
本文档展示如何利用PCL可视化点云

## 准备
进入 [pcl_tool.cc](../tool/pcl_tool.cc) 

### 1 启用可视化查看工具
```cpp
/* ------------Select the required file format ------------ */
// #define SAVE_PCD_FILE_ASCII
// #define SAVE_PCD_FILE_BIN
// #define SAVE_PCD_FILE_BIN_COMPRESSED
// #define SAVE_PLY_FILE
#define ENABLE_VIEWER  // 确保该行代码解注释
```

### 2 解析配置参考 
请参考 **[如何在线解析激光雷达数据](../docs/parsing_lidar_data_online_CN.md)** 和 **[如何离线解析PCAP文件数据](../docs/parsing_pcap_file_data_offline_CN.md)**

以PCAP解析为例

``` cpp
/* -------------------Select the test mode ------------------- */
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
### 1 编译
在HesaiLidar_SDK_2.0文件夹下，启动Terminal终端，执行以下指令。
```bash
cd HesaiLidar_SDK_2.0/tool
mkdir build
cd build
cmake ..
make
```

### 2 运行
成功编译后，在build文件夹下运行生成的pcl_tool可执行文件，系统会有可视化窗口。
```bash
./pcl_tool
```

