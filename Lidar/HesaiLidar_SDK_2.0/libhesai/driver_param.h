/************************************************************************************************
Copyright (C) 2023 Hesai Technology Co., Ltd.
Copyright (C) 2023 Original Authors
All rights reserved.

All code in this repository is released under the terms of the following Modified BSD License. 
Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and 
  the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
  the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************************************************************/

#pragma once
#include <string>
#include "logger.h"  
#include "hs_com.h"
namespace hesai
{
namespace lidar
{

#define NULL_TOPIC  "your topic name"

enum SourceType
{
  DATA_FROM_LIDAR = 1,
  DATA_FROM_PCAP = 2,
  DATA_FROM_ROS_PACKET = 3,
  DATA_FROM_SERIAL = 4,
  DATA_FROM_EXTERNAL_INPUT = 5,
};

enum PtcMode
{
  tcp = 0, 
  tcp_ssl = 1
};

enum UseTimestampType
{
  point_cloud_timestamp = 0,
  sdk_recv_timestamp = 1,
};

///< LiDAR decoder parameter
typedef struct DecoderParam  
{
  ///< Used to transform points
  TransformParam transform_param;    
  int thread_num = 1;
  bool enable_udp_thread = true;
  bool enable_parser_thread = true;
  bool pcap_play_synchronization = true;
  bool pcap_play_in_loop = false;
  //start a new frame when lidar azimuth greater than frame_start_azimuth
  //range:[0-360), set frame_start_azimuth less than 0 if you do want to use it
  float frame_start_azimuth = 0;
  // enable the udp packet loss detection tool
  // it forbiddens parser udp packet while trun on this tool
  bool enable_packet_loss_tool = false;
  bool enable_packet_timeloss_tool = false;
  bool packet_timeloss_tool_continue = false;
  // 0 use point cloud timestamp
  // 1 use sdk receive timestamp
  uint16_t use_timestamp_type = point_cloud_timestamp;
  int fov_start = -1;
  int fov_end = -1;
  bool distance_correction_flag = false;
  bool xt_spot_correction = false;
  RemakeConfig remake_config;
  uint32_t socket_buffer_size = 0;
} DecoderParam;

///< The LiDAR input parameter
typedef struct InputParam  
{
  // PTC mode
  PtcMode ptc_mode = PtcMode::tcp;
  SourceType source_type = DATA_FROM_PCAP;
  bool use_someip = false;
  // Ip of Lidar
  std::string device_ip_address = "";   
  ///< Address of multicast
  std::string multicast_ip_address = "";  
  ///< Address of host
  std::string host_ip_address = "Your host ip"; 
  ///< port filter
  uint16_t device_udp_src_port = 0;
  uint16_t device_fault_port = 0;
  // udp packet destination port number       
  uint16_t udp_port = 2368;   
  uint16_t fault_message_port = 0;
  ///< ptc packet port number     
  bool use_ptc_connected = true;           
  uint16_t ptc_port = 9347;
  ///< serial port and baudrate
  std::string rs485_com = "/dev/ttyUSB0";
  std::string rs232_com = "/dev/ttyUSB1";
  int point_cloud_baudrate = 3125000;
  int rs485_baudrate = 115200;   
  int rs232_baudrate = 9600;          
  std::string pcap_path = "Your pcap file path";  ///< Absolute path of pcap file
  std::string correction_file_path = "Your correction file path";   ///< Path of angle calibration files(angle.csv).Only used for internal debugging.
  std::string firetimes_path = "Your firetime file path";  ///< Path of firetime files(angle.csv).
  std::string correction_save_path = "";
  /// certFile          Represents the path of the user's certificate
  std::string certFile = "";
  /// privateKeyFile    Represents the path of the user's private key
  std::string privateKeyFile = "";
  /// caFile            Represents the path of the root certificate
  std::string caFile = "";
  /// standby_mode    set the standby_mode of lidar
  int standby_mode = -1;
  /// speed             set the rotational speed of lidar
  int speed = -1;
  // timeout
  float recv_point_cloud_timeout = -1; //(s), <0 : not timeout 
  float ptc_connect_timeout = -1; //(s), <0 : not timeout 


  bool send_packet_ros;
  bool send_point_cloud_ros;
  bool send_imu_ros;
  std::string frame_id;

  std::string ros_send_packet_topic = NULL_TOPIC;
  std::string ros_send_point_topic = NULL_TOPIC;
  std::string ros_send_packet_loss_topic = NULL_TOPIC; 
  std::string ros_send_ptp_topic = NULL_TOPIC;
  std::string ros_send_correction_topic = NULL_TOPIC;
  std::string ros_send_firetime_topic = NULL_TOPIC;
  std::string ros_send_imu_topic = NULL_TOPIC;

  std::string ros_recv_correction_topic = NULL_TOPIC;
  std::string ros_recv_packet_topic = NULL_TOPIC;


} InputParam;

///< The LiDAR driver parameter
typedef struct DriverParam  
{
  ///< Input parameter
  InputParam input_param;  
  ///< Decoder parameter        
  DecoderParam decoder_param;  
  ///< The frame id of LiDAR message    
  std::string frame_id = "hesai";  
  bool use_gpu = false;
  uint8_t log_level = HESAI_LOG_INFO | HESAI_LOG_WARNING | HESAI_LOG_ERROR | HESAI_LOG_FATAL; //
  uint8_t log_Target = HESAI_LOG_TARGET_CONSOLE;
  std::string log_path = "./log.log";
} DriverParam;
}  // namespace lidar
}  // namespace hesai


/*
1. 枚举类型
SourceType
定义数据来源类型：

DATA_FROM_LIDAR (1): 直接从LiDAR设备获取数据

DATA_FROM_PCAP (2): 从PCAP文件读取数据

DATA_FROM_ROS_PACKET (3): 从ROS包获取数据

DATA_FROM_SERIAL (4): 从串口获取数据

DATA_FROM_EXTERNAL_INPUT (5): 外部输入数据

PtcMode
定义PTC通信模式：

tcp (0): 普通TCP模式

tcp_ssl (1): TCP SSL加密模式

UseTimestampType
定义时间戳类型：

point_cloud_timestamp (0): 使用点云时间戳

sdk_recv_timestamp (1): 使用SDK接收时间戳

2. 解码参数(DecoderParam)
用于配置LiDAR数据解码的参数：

transform_param: 点云变换参数

thread_num: 线程数量，默认为1

enable_udp_thread: 是否启用UDP线程，默认为true

enable_parser_thread: 是否启用解析线程，默认为true

pcap_play_synchronization: PCAP回放时是否同步时间，默认为true

pcap_play_in_loop: PCAP文件是否循环播放，默认为false

frame_start_azimuth: 帧起始方位角(0-360度)，小于0表示不使用

enable_packet_loss_tool: 启用UDP丢包检测工具，默认为false

enable_packet_timeloss_tool: 启用数据包时间丢失检测工具，默认为false

packet_timeloss_tool_continue: 时间丢失检测工具是否持续运行，默认为false

use_timestamp_type: 使用的时间戳类型(point_cloud_timestamp或sdk_recv_timestamp)

fov_start: 视场角起始角度，默认为-1(不使用)

fov_end: 视场角结束角度，默认为-1(不使用)

distance_correction_flag: 是否启用距离校正，默认为false

xt_spot_correction: 是否启用XT点校正，默认为false

remake_config: 重构配置

socket_buffer_size: 套接字缓冲区大小，默认为0(使用系统默认值)

3. 输入参数(InputParam)
配置LiDAR数据输入源和相关参数：

ptc_mode: PTC通信模式(tcp或tcp_ssl)

source_type: 数据源类型

use_someip: 是否使用SOME/IP协议，默认为false

device_ip_address: LiDAR设备IP地址

multicast_ip_address: 组播IP地址

host_ip_address: 主机IP地址，默认为"Your host ip"

device_udp_src_port: 设备UDP源端口，默认为0

device_fault_port: 设备故障端口，默认为0

udp_port: UDP数据包目标端口，默认为2368

fault_message_port: 故障消息端口，默认为0

use_ptc_connected: 是否使用PTC连接，默认为true

ptc_port: PTC端口号，默认为9347

rs485_com: RS485串口设备路径，默认为"/dev/ttyUSB0"

rs232_com: RS232串口设备路径，默认为"/dev/ttyUSB1"

point_cloud_baudrate: 点云串口波特率，默认为3125000

rs485_baudrate: RS485波特率，默认为115200

rs232_baudrate: RS232波特率，默认为9600

pcap_path: PCAP文件绝对路径，默认为"Your pcap file path"

correction_file_path: 角度校准文件路径(angle.csv)，用于内部调试

firetimes_path: 发射时间文件路径

correction_save_path: 校正文件保存路径

certFile: 用户证书路径(SSL用)

privateKeyFile: 用户私钥路径(SSL用)

caFile: 根证书路径(SSL用)

standby_mode: LiDAR待机模式，-1表示不设置

speed: LiDAR旋转速度(RPM)，-1表示不设置

recv_point_cloud_timeout: 接收点云超时时间(秒)，<0表示不超时

ptc_connect_timeout: PTC连接超时时间(秒)，<0表示不超时

ROS相关参数：

send_packet_ros: 是否发送ROS数据包

send_point_cloud_ros: 是否发送ROS点云

send_imu_ros: 是否发送ROS IMU数据

frame_id: 坐标系ID

各种ROS topic名称(发送和接收)

4. 驱动参数(DriverParam)
整合所有配置参数：

input_param: 输入参数

decoder_param: 解码参数

frame_id: LiDAR消息的帧ID，默认为"hesai"

use_gpu: 是否使用GPU加速，默认为false

log_level: 日志级别(INFO/WARNING/ERROR/FATAL的组合)

log_Target: 日志输出目标(控制台)

log_path: 日志文件路径，默认为"./log.log"

*/