# Functional Parameter Reference

### 1 [DecoderParam](../libhesai/driver_param.h)

#### 1.1 Time Synchronization
1. `pcap_play_synchronization`: Enable real-time playback simulation when parsing PCAP files
    *(Default: true)*

2. `use_timestamp_type`: Select timestamp source  
   - `0`: Point cloud timestamp (default)
   - `1`: System local time

#### 1.2 Frame Processing
3. `frame_start_azimuth`: Frame segmentation start angle (0-360 degrees)
    *(Default: 0.0)*

4. `fov_start`/`fov_end`: Valid field of view range (angle values), point clouds outside this range will be filtered  
   *(Default: 0.0-360.0)*

#### 1.3 Packet Analysis
5. `enable_packet_loss_tool`: Enable sequence number-based packet loss statistics  
   *(Default: false)*

6. `enable_packet_timeloss_tool`: Enable timestamp-based packet loss detection  
   *(Default: false)*

7. `packet_timeloss_tool_continue`: Continuously monitor timestamp continuity
   *(Default: false)*

#### 1.4 Calibration Settings
8. `distance_correction_lidar_flag`: Enable optical center correction for mechanical lidars
   *(Optional for Pandar/QT/OT/XT/JT series)*

9. `xt_spot_correction`: Enable XT16/32 S-shaped layered correction  
   *(Default: false)*

#### 1.5 Performance
10. `system_udpsocket_buffer_size`: UDP socket buffer size (bytes) 
    *(Default: 1500)*

#### 1.6 Coordinate Transformation Related
11. `transform_param`: See definition [TransformParam](../libhesai/Common/include/hs_com.h), implements coordinate transformation

12. `remake_config`: See definition [RemakeConfig](../libhesai/Common/include/hs_com.h), implements data rearrangement according to horizontal and vertical angle positions

#### 1.7 Other Configurations
13. `pcap_play_in_loop`: Allow looping playback of PCAP data
      *(Default: false)*

14. `thread_num`: Multi-threaded parsing, single-threaded parsing when 1, multi-threaded parsing when greater than 1

15. `enable_udp_thread`: Whether to allow receiving point cloud data from lidar
      *(Default: true)*

16. `enable_parser_thread`: Whether to allow parser thread to work
      *(Default: true)*

---

### 2 [InputParam](../libhesai/driver_param.h)

#### 2.1 Connection Basics
1. `source_type`: Data source selection  
   - `DATA_FROM_PCAP`: PCAP file playback
   - `DATA_FROM_LIDAR`: Network connection
   - `DATA_FROM_SERIAL`: Serial port connection

2. `ptc_mode`: Communication protocol mode  
   - `0`: TCP (default)
   - `1`: SSL encryption

#### 2.2 Network Configuration
3. `device_ip_address`: Lidar IP address
   *(Default: 192.168.1.201)*

4. `udp_port`: Point cloud data port  
   *(Default: 2368)*

5. `ptc_port`: Control protocol port  
   *(Default: 9347)*

6. `multicast_ip_address`: Multicast address 
   *(Empty string means no multicast address is set)*

7. `fault_message_port`: Fault message destination port number, configured when different from udp_port

8. `host_ip_address`: Host IP address

9. `device_udp_src_port`: Point cloud source port number, filter point clouds through point cloud source IP address + source port number, valid values [1024, 65535]

10. `device_fault_port`: Fault message source port number, filter fault messages through fault message source IP address + source port number, valid values [1024, 65535]

11. `use_ptc_connected`: Whether to use PTC communication

#### 2.3 File Paths
12. `pcap_path`: PCAP file path  
   *(Required for offline PCAP playback)*

13. `correction_file_path`: Calibration file path
   *(Default requires CSV format, some lidars support DAT format)*

14. `firetimes_path`: Firing timing correction file path

#### 2.4 Serial Port Configuration (JT16 only)
15. `rs485_com`: Point cloud receiving serial port  
    *(Example: COM3 or /dev/ttyUSB0)*

16. `rs232_com`: Command sending serial port

17. `point_cloud_baudrate`: RS485 baud rate  
    *(Default: 3125000)*

18. `rs485_baudrate`: OTA command baud rate
    *(Default: 115200)*

19. `rs232_baudrate`: Normal command baud rate  
    *(Default: 9600)*

20. `correction_save_path`: Local save path for angle calibration file obtained via serial port

#### 2.5 Security Configuration
21. `certFile`: Client certificate path  

22. `privateKeyFile`: Private key path 

23. `caFile`: CA certificate path 

#### 2.6 Lidar Configuration Related
22. `standby_mode`: Configure lidar working mode after successful PTC connection

23. `speed`: Configure lidar rotation speed after successful PTC connection

#### 2.7 Timeout Configuration
24. `recv_point_cloud_timeout`: Timeout for receiving point cloud data during initialization (s), effective when >= 0

25. `ptc_connect_timeout`: Timeout for PTC connection during initialization (s), effective when >= 0

---

### 3 [DriverParam](../libhesai/driver_param.h)

#### 3.1 Log Configuration
1. `log_level`: Log detail level  
   - `0`: Debug
   - `1`: Info (default)
   - `2`: Warning
   - `3`: Error

2. `log_Target`: Output target  
   - `0`: Running terminal (default)
   - `1`: File
   - `2`: Both outputs

3. `log_path`: Log file directory  
   *(Default: ./logs)*

4. `use_gpu`: Whether to use GPU for parsing