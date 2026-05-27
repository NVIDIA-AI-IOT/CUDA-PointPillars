# Point Cloud Data Visualization
This document shows how to visualize point clouds using PCL

## Preparation
Navigate to [pcl_tool.cc](../tool/pcl_tool.cc) 

### 1 Enable Visualization Viewer Tool
```cpp
/* ------------Select the required file format ------------ */
// #define SAVE_PCD_FILE_ASCII
// #define SAVE_PCD_FILE_BIN
// #define SAVE_PCD_FILE_BIN_COMPRESSED
// #define SAVE_PLY_FILE
#define ENABLE_VIEWER  // Make sure this line of code is uncommented
```

### 2 Parsing Configuration Reference 
Please refer to **[How to Parse Lidar Data Online](../docs/parsing_lidar_data_online.md)** and **[How to Parse PCAP File Data Offline](../docs/parsing_pcap_file_data_offline.md)**

Using PCAP parsing as an example

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
After successful compilation, run the generated pcl_tool executable file in the build folder. The system will have a visualization window.
```bash
./pcl_tool
```

