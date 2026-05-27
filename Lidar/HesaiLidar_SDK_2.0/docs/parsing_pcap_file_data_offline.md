# Parsing PCAP Point Cloud Data Offline

## Preparation
```cpp
// Enter test.cc for PCAP related configuration
// Use macro PCAP_PARSER_TEST, comment out the other three macros

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
```bash
# 1. Build executable sample program (from build folder): After successful compilation, generate executable program
make -j$(nproc)

# 2. Execute sample program: Start parsing data
./sample
```