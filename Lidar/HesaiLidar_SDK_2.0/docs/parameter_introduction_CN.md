# 功能参数参考

### 1 解码器参数 [DecoderParam](../libhesai/driver_param.h)

#### 1.1 时间同步
1. `pcap_play_synchronization`: 解析PCAP文件时启用实时播放模拟
    *(默认: true)*

2. `use_timestamp_type`: 选择时间戳来源  
   - `0`: 点云时间戳 (默认)
   - `1`: 系统本地时间

#### 1.2 帧处理
3. `frame_start_azimuth`: 帧分割起始角度 (0-360度)
    *(默认: 0.0)*

4. `fov_start`/`fov_end`: 有效视场范围 (角度值)，此范围外的点云将被过滤  
   *(默认: 0.0-360.0)*

#### 1.3 数据包分析
5. `enable_packet_loss_tool`: 启用基于序列号的数据包丢失统计  
   *(默认: false)*

6. `enable_packet_timeloss_tool`: 启用基于时间戳的数据包丢失检测  
   *(默认: false)*

7. `packet_timeloss_tool_continue`: 持续监控时间戳连续性
   *(默认: false)*

#### 1.4 校准设置
8. `distance_correction_lidar_flag`: 为机械式激光雷达启用光心修正
   *(Pandar/QT/OT/XT/JT系列可选)*

9. `xt_spot_correction`:  启用XT16/32 S形分层校正  
   *(默认: false)*

#### 1.5 Performance
10. `system_udpsocket_buffer_size`: UDP套接字缓冲区大小 (字节) 
    *(默认: 1500)*

#### 1.6 坐标变化相关
11. `transform_param`: 详见定义[TransformParam](../libhesai/Common/include/hs_com.h)，实现坐标转换

12. `remake_config`: 详见定义[RemakeConfig](../libhesai/Common/include/hs_com.h)，实现数据按照水平角和垂直角位置进行重排

#### 1.7 其他配置
13. `pcap_play_in_loop`: 允许循环播放PCAP数据
      *(默认: false)*

14. `thread_num`: 多线程解析，为1时为单线程解析，大于1时会开启多线程解析

15. `enable_udp_thread`: 是否允许接收来自雷达的点云数据
      *(默认: true)*

16. `enable_parser_thread`: 是否允许解析线程工作
      *(默认: true)*

---

### 2 输入参数 [InputParam](../libhesai/driver_param.h)

#### 2.1 连接基础
1. `source_type`: 数据源选择  
   - `DATA_FROM_PCAP`: PCAP文件回放
   - `DATA_FROM_LIDAR`: 网络连接
   - `DATA_FROM_SERIAL`: 串口连接

2. `ptc_mode`: 通信协议模式  
   - `0`: TCP (默认)
   - `1`: SSL加密

#### 2.2 网络配置
3. `device_ip_address`: 激光雷达IP地址
   *(默认: 192.168.1.201)*

4. `udp_port`: 点云数据端口  
   *(默认: 2368)*

5. `ptc_port`: 控制协议端口  
   *(默认: 9347)*

6. `multicast_ip_address`: 组播地址 
   *(空字符串表示不设置组播地址)*

7. `fault_message_port`: 故障报文目的端口号，与udp_port不同时配置

8. `host_ip_address`: 主机IP地址

9. `device_udp_src_port`: 点云源端口号，通过点云源IP地址+源端口号筛选点云，有效值[1024， 65535]

10. `device_fault_port`: 故障报文源端口号，通过故障报文源IP地址+源端口号筛选故障报文，有效值[1024， 65535]

11. `use_ptc_connected`: 是否使用PTC通讯

#### 2.3 文件路径
12. `pcap_path`: PCAP文件路径  
   *(离线回放PCAP必需)*

13. `correction_file_path`: 校准文件路径
   *(默认需要CSV格式，部分雷达支持DAT格式)*

14. `firetimes_path`: 发光时刻修正文件路径

#### 2.4 串口配置 (仅JT16)
15. `rs485_com`: 点云接受串口  
    *(示例：COM3 或 /dev/ttyUSB0)*

16. `rs232_com`: 命令发送串口

17. `point_cloud_baudrate`: RS485波特率  
    *(默认: 3125000)*

18. `rs485_baudrate`: OTA命令波特率
    *(默认: 115200)*

19. `rs232_baudrate`: 正常命令波特率  
    *(默认: 9600)*

20. `correction_save_path`: 串口获取角度校准文件后的本地保存路径

#### 2.5 安全配置
21. `certFile`: 客户端证书路径  

22. `privateKeyFile`: 私钥路径 

23. `caFile`: CA证书路径 

#### 2.6 雷达配置相关
22. `standby_mode`: PTC连接成功后，配置雷达工作模式

23. `speed`: PTC连接成功后，配置雷达转速
#### 2.7 配置超时信息
24. `recv_point_cloud_timeout`: 初始化中接收点云数据的超时时间(s)，大于等于0时有效

25. `ptc_connect_timeout`: 初始化时PTC连接的超时时间(s)，大于等于0时有效

---

### 3 驱动参数 [DriverParam](../libhesai/driver_param.h)

#### 3.1 日志配置
1. `log_level`: 日志详细级别  
   - `0`: 调试
   - `1`: 信息 (默认)
   - `2`: 警告
   - `3`: 错误

2. `log_Target`: 输出目标  
   - `0`: 运行终端 (默认)
   - `1`: 文件
   - `2`: 同时输出

3. `log_path`: 日志文件目录  
   *(默认: ./logs)*

4. `use_gpu`: 是否使用gpu解析