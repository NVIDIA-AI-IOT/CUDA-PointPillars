# PCAP to PCD Conversion
HesaiLidar_SDK_2.0 provides sample code for converting PCAP format files to PCD format point clouds.

## Preparation
Navigate to [pcl_tools.cc](../tool/pcl_tool.cc) 

#### 1 Select the Required PCD Format

The sample code includes four commented macro definitions (`#define`) that control the enabling or disabling of different PCD point cloud format conversion functions.

Select the macro corresponding to the desired format type and uncomment it (multiple selections allowed):

```cpp
/* ------------Select the required file format ------------ */
#define SAVE_PCD_FILE_ASCII
// #define SAVE_PCD_FILE_BIN
// #define SAVE_PCD_FILE_BIN_COMPRESSED
// #define SAVE_PLY_FILE
```

1. **`SAVE_PCD_FILE_ASCII`**  
   Enabling this macro definition will save PCD files in ASCII format. ASCII format is a text format where data is stored in readable form, convenient for debugging and viewing, but usually results in larger file sizes and slower read/write speeds.

2. **`SAVE_PCD_FILE_BIN`**  
   Enabling this macro definition will save PCD files in binary format. Binary format is more efficient than ASCII format, with smaller file sizes and faster read/write speeds, but the data is not directly readable.

3. **`SAVE_PCD_FILE_BIN_COMPRESSED`**  
   Enabling this macro definition may save PCD files in compressed binary format. This format further compresses data based on binary format, further reducing file size.

4. **`SAVE_PLY_FILE`**  
   After enabling this macro definition, the program may save PLY files (Polygon File Format or Stanford Triangle Format). PLY files are a file format commonly used for storing three-dimensional data.

#### 2 Select Member Variables to Save

Currently, besides the default saved x, y, z coordinates, six member variables are supported, controlled by the following six macros. Select the needed member variables and uncomment them:

> Note: Some member variables need to be confirmed whether the lidar supports them, otherwise they will be all zeros. See additional references below for details.

```cpp
/* ------------Select the fields to be exported ------------ */
#define ENABLE_TIMESTAMP
#define ENABLE_RING
#define ENABLE_INTENSITY
// #define ENABLE_CONFIDENCE
// #define ENABLE_WEIGHT_FACTOR
// #define ENABLE_ENV_LIGHT
```
   1. **`ENABLE_TIMESTAMP`**: Point cloud timestamp
   2. **`ENABLE_RING`**: Channel number
   3. **`ENABLE_INTENSITY`**: Reflection intensity
   4. **`ENABLE_CONFIDENCE`**: Confidence or other flag bits
   5. **`ENABLE_WEIGHT_FACTOR`**: Weight factor information
   6. **`ENABLE_ENV_LIGHT`**: Environmental light information

#### 3 For parsing configuration reference, see **[How to Parse Lidar Data Online](../docs/parsing_lidar_data_online.md)** and **[How to Parse PCAP File Data Offline](../docs/parsing_pcap_file_data_offline.md)**

Using PCAP parsing as an example:

``` cpp
/* -------------------Select the test mode ------------------- */
// #define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST
#define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ... 

#ifdef PCAP_PARSER_TEST
  param.input_param.source_type = DATA_FROM_PCAP;                       // Set data source to offline PCAP point cloud data
  param.input_param.pcap_path = "path/to/pcap";                         // Offline PCAP point cloud data path
  param.input_param.correction_file_path = "/path/to/correction.csv";   // Calibration file (angle correction file), recommend using the lidar's own calibration file
  param.input_param.firetimes_path = "path/to/firetimes.csv";           // Optional: Channel firing timing (firing moment correction file)

  param.decoder_param.pcap_play_synchronization = true;                 // Synchronize parsing according to point cloud timestamp, simulating actual lidar frequency
  param.decoder_param.pcap_play_in_loop = false;                        // Loop parsing PCAP
#endif
```

## Steps
### 1 Compilation
In the HesaiLidar_SDK_2.0 folder, open a terminal and execute the following commands:
```bash
cd HesaiLidar_SDK_2.0/tool
mkdir build
cd build
cmake ..
make
```

### 2 Run
After successful compilation, run the generated pcl_tool executable file in the build folder. The system will have a visualization window and generate corresponding point clouds for each frame of the PCAP file in the build folder.
```bash
./pcl_tool
```

## Additional References
#### 1 How to Define Data Precision for ASCII Format PCD Files
`writeASCII` is a method of the `pcl::PCDWriter` class. Here's its common usage:
```cpp
pcl::PCDWriter writer;
writer.writeASCII("output.pcd", cloud);
```
PCD files saved using the default method have limited data precision, for example, timestamps will be represented in scientific notation.

How to change PCD file data precision?
```cpp
pcl::PCDWriter writer;
// you can change the value of precision to adjust the precision
int precision = 16;
writer.writeASCII(file_name1, *pcl_pointcloud, precision);
```

#### 2 How to Define Timestamp in PCD Filename
- Use the timestamp of the first point in the frame header as the timestamp displayed in the PCD filename:
```cpp
  std::string file_name1 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".pcd";
  std::string file_name2 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".bin";
  std::string file_name3 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ ".ply";
  std::string file_name4 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_start_timestamp)+ "_compress" + ".bin";
```

- Use the timestamp of the last point in the frame as the timestamp displayed in the PCD filename:
```cpp
  std::string file_name1 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".pcd";
  std::string file_name2 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".bin";
  std::string file_name3 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ ".ply";
  std::string file_name4 = "./PointCloudFrame" + std::to_string(frame.frame_index) + "_" + std::to_string(frame.frame_end_timestamp)+ "_compress" + ".bin";
```

#### 3 How to Confirm Whether Saved Member Variables are Supported

- Confirm the required member variables in `struct PointXYZIT`, for example `weightFactor`, then its assignment function is `set_weightFactor`

- Confirm the point cloud UDP version number of your lidar, which can be viewed through `wireshark` packet capture. Generally, the first two bytes in point cloud UDP packets are 0xEE 0xFF, and the following two bytes are the version number (converted to decimal). Using `Pandar128E3X` as an example, the version number is `1.4`

- Navigate to file [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc), search for the `set_weightFactor` function. If the code exists, the lidar supports this field.

#### 4 How to Add New Saved Member Variables

Taking the addition of member variable `addFlag` with data type `uint8_t` as an example:

- Add the required member variable `uint8_t addFlag;` in `struct PointXYZIT` in this file. Add `(std::uint8_t, addFlag, addFlag)` in `POINT_CLOUD_REGISTER_POINT_STRUCT`

- Confirm the point cloud UDP version number of your lidar, which can be viewed through `wireshark` packet capture. Generally, the first two bytes in point cloud UDP packets are 0xEE 0xFF, and the following two bytes are the version number (converted to decimal). Using `Pandar128E3X` as an example, the version number is `1.4`

- Navigate to file [general_parser.h](../libhesai/UdpParser/include/general_parser.h), search for `DEFINE_MEMBER_CHECKER`, and add `DEFINE_MEMBER_CHECKER(addFlag)` and `DEFINE_SET_GET(addFlag, uint8_t)` below

- Navigate to file [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc), search for the `ComputeXYZI` function, and call the `set_addFlag` function within it to implement assignment to `frame.points[point_index_rerank]` 