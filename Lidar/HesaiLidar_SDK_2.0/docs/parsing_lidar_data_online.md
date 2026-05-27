# Parsing Lidar Data Online

## Preparation

#### Ethernet-connected Lidar
```cpp
// Enter test.cc for network configuration
// Use macro LIDAR_PARSER_TEST, comment out the other three macros

#define LIDAR_PARSER_TEST
// #define SERIAL_PARSER_TEST
// #define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ...

#ifdef LIDAR_PARSER_TEST
  param.input_param.source_type = DATA_FROM_LIDAR;          // Set data source to online point cloud data
  param.input_param.device_ip_address = "192.168.1.201";    // Lidar IP address
  param.input_param.ptc_port = 9347;                        // PTC protocol port
  param.input_param.udp_port = 2368;                        // Point cloud data destination port
  param.input_param.multicast_ip_address = "";              // Optional: If destination IP is multicast IP

  param.input_param.ptc_mode = PtcMode::tcp;                // PTC communication type, can be ignored if not using TLS/mTLS-based PTCS
  param.input_param.use_ptc_connected = true;               // Whether to use PTC connection
  param.input_param.correction_file_path = "/path/to/correction.csv";   // Angle correction file (recommend using the lidar's own angle correction file)
  param.input_param.firetimes_path = "path/to/firetimes.csv";           // Optional: Channel firing timing (firing moment correction file)

  param.input_param.use_someip = false;                     // Ignore
  param.input_param.host_ip_address = "";                   // Lidar IP address, ignore if same as point cloud source IP
  param.input_param.fault_message_port = 0;                 // Fault message destination port, ignore if same as udp_port

  // PtcMode::tcp_ssl use                                   Some PTCS configurations, ignore if not used
  param.input_param.certFile = "";
  param.input_param.privateKeyFile = "";
  param.input_param.caFile = "";
#endif
```

#### Serial-connected Lidar (JT16)
```cpp
// Enter test.cc for serial port configuration
// Use macro SERIAL_PARSER_TEST, comment out the other three macros

// #define LIDAR_PARSER_TEST
#define SERIAL_PARSER_TEST
// #define PCAP_PARSER_TEST
// #define EXTERNAL_INPUT_PARSER_TEST
... ...

#ifdef SERIAL_PARSER_TEST
  param.input_param.source_type = DATA_FROM_SERIAL;         // Set data source to serial port data
  param.input_param.rs485_com = "/dev/ttyUSB0";             // RS485 point cloud serial port (COM0 in Windows system), actual serial port number may vary
  param.input_param.rs232_com = "/dev/ttyUSB1";             // RS232 control serial port (COM1 in Windows system), actual serial port number may vary
  param.input_param.point_cloud_baudrate = 3125000;         // Point cloud baud rate
  param.input_param.correction_save_path = "";              // Internal lidar angle calibration file save path, if 232 port is set and angle calibration file is successfully obtained from lidar
  param.input_param.correction_file_path = "/path/to/correction.csv"; // Angle correction file (recommend using the lidar's own angle correction file)
#endif
```

## Steps
```bash
# 1. Build executable sample program (from build folder): After successful compilation, generate executable program
make -j$(nproc)

# 2. Execute sample program: Start parsing data
./sample
```