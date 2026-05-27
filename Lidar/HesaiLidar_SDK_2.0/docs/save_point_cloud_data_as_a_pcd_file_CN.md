# PCAP转PCD
HesaiLidar_SDK_2.0提供了将PCAP格式文件转换为PCD格式点云的示例代码。

## 准备
进入 [pcl_tools.cc](../tool/pcl_tool.cc) 

#### 1 选择需要保存的PCD格式

示例代码中包含四个被注释掉的宏定义（`#define`），作用是控制程序转存不同类型PCD点云格式功能的启用或禁用

选择需要的格式类型对应的宏，解注释即可 （可多选）

```cpp
/* ------------Select the required file format ------------ */
#define SAVE_PCD_FILE_ASCII
// #define SAVE_PCD_FILE_BIN
// #define SAVE_PCD_FILE_BIN_COMPRESSED
// #define SAVE_PLY_FILE
```

1. **`SAVE_PCD_FILE_ASCII`**  
   启用这个宏定义，程序会以ASCII格式保存PCD文件。ASCII格式是一种文本格式，数据以可读的形式存储，便于调试和查看，但文件体积通常较大，读取和写入速度较慢。

2. **`SAVE_PCD_FILE_BIN`**  
   启用这个宏定义，程序会以二进制格式保存PCD文件。二进制格式相比ASCII格式更高效，文件体积更小，读写速度更快，但数据不可直接阅读。

3. **`SAVE_PCD_FILE_BIN_COMPRESSED`**  
   启用这个宏定义，程序可能会以压缩的二进制格式保存PCD文件。这种格式在二进制的基础上进一步压缩数据，进一步减小文件体积。

4. **`SAVE_PLY_FILE`**  
   启用这个宏定义后，程序可能会保存PLY文件（Polygon File Format或Stanford Triangle Format）。PLY文件是一种常用于存储三维数据的文件格式。

#### 2 选择需要保存的成员变量

当前除默认保存的x y z之外，共支持六种成员变量，通过以下六个宏控制，选择需要的成员变量解注释即可

> 注意：部分成员变量需要先确认雷达是否支持，否则为全0。详见后续更多参考

```cpp
/* ------------Select the fields to be exported ------------ */
#define ENABLE_TIMESTAMP
#define ENABLE_RING
#define ENABLE_INTENSITY
// #define ENABLE_CONFIDENCE
// #define ENABLE_WEIGHT_FACTOR
// #define ENABLE_ENV_LIGHT
```
   1. **`ENABLE_TIMESTAMP`** ： 点云时间戳
   2. **`ENABLE_RING`** ： 通道号
   3. **`ENABLE_INTENSITY`** ： 反射强度
   4. **`ENABLE_CONFIDENCE`** ： 置信度或其他标志位
   5. **`ENABLE_WEIGHT_FACTOR`** ： 权重因子信息
   6. **`ENABLE_ENV_LIGHT`** ： 环境光信息


#### 3 解析配置参考 **[如何在线解析激光雷达数据](../docs/parsing_lidar_data_online_CN.md)** 和 **[如何离线解析PCAP文件数据](../docs/parsing_pcap_file_data_offline_CN.md)**

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
成功编译后，在build文件夹下运行生成的pcl_tool可执行文件，系统会有可视化窗口。且在build文件夹下生成对应PCAP文件的每帧点云。
```bash
./pcl_tool
```

## 更多参考
#### 1 如何定义ASCII格式的PCD文件数据精度
`writeASCII` 是 `pcl::PCDWriter` 类的一个方法，以下是其常见用法：
```cpp
pcl::PCDWriter writer;
writer.writeASCII("output.pcd", cloud);
```
按默认的方法保存的PCD文件数据精度有限，例如时间戳会表示为科学计数法。

如何更改PCD文件数据精度？
```cpp
pcl::PCDWriter writer;
// you can change the value of precision to adjust the precison
int precision = 16;
writer.writeASCII(file_name1, *pcl_pointcloud, precision);
```

#### 2 如何定义PCD文件名的时间戳
- 以帧头第一个点的时间戳作为PCD文件名显示的时间戳
```cpp
  std::string file_name1 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".pcd";
  std::string file_name2 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".bin";
  std::string file_name3 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".ply";
  std::string file_name4 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ "_compress" + ".bin";
```

- 以该帧内最后一个点的时间戳作为PCD文件名显示的时间戳
```cpp
  std::string file_name1 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".pcd";
  std::string file_name2 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".bin";
  std::string file_name3 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".ply";
  std::string file_name4 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ "_compress" + ".bin";
```

#### 3 如何确认保存的成员变量是否支持

- 确认 `struct PointXYZIT` 中需要的成员变量，例如 `weightFactor`，则其赋值函数为 `set_weightFactor`

- 确认您使用雷达的点云UDP版本号，可通过`wireshark`抓包查看，一般情况下，点云UDP包中前两个字节为0xEE 0xFF，后续两个字节即为版本号(转化为十进制)，我们以 `Pandar128E3X` 为例，版本号为 `1.4`

- 进入文件 [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc) 中，搜索 `set_weightFactor` 函数，如果存在代码该雷达支持该字段。

#### 4 如何新增保存的成员变量

以新增成员变量 `addFlag` ，数据类型为 `uint8_t` 为例

- 在本文件中的 `struct PointXYZIT` 中添加需要的成员变量 `uint8_t addFlag;`。在 `POINT_CLOUD_REGISTER_POINT_STRUCT` 中添加 `(std::uint8_t, addFlag, addFlag)`

- 确认您使用雷达的点云UDP版本号，可通过`wireshark`抓包查看，一般情况下，点云UDP包中前两个字节为0xEE 0xFF，后续两个字节即为版本号(转化为十进制)，我们以 `Pandar128E3X` 为例，版本号为 `1.4`

- 进入文件 [general_parser.h](../libhesai/UdpParser/include/general_parser.h) 中，搜索 `DEFINE_MEMBER_CHECKER`，在下方同步添加 `DEFINE_MEMBER_CHECKER(addFlag)` 和 `DEFINE_SET_GET(addFlag, uint8_t)`

- 进入文件 [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc) 中，搜索 `ComputeXYZI` 函数，在其中调用`set_addFlag` 函数实现对 `frame.points[point_index_rerank]` 的赋值