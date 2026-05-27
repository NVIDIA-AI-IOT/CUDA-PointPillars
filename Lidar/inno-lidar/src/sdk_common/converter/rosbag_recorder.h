/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#ifndef CONVERTER_ROSBAG_RECORDER_H_
#define CONVERTER_ROSBAG_RECORDER_H_

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <string>
#include <cstring>
#include <ctime>
#include <memory>
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "sdk_common/converter/recorder_base.h"

namespace innovusion {
// Current header fields
#define INNO_ROS_BAG_RCD_OP_FIELD_NAME        {'o', 'p', '=' }
#define INNO_ROS_BAG_RCD_TOPIC_FIELD_NAME     {'t', 'o', 'p', 'i', 'c', '='}
#define INNO_ROS_BAG_RCD_VER_FIELD_NAME       {'v', 'e', 'r', '='}
#define INNO_ROS_BAG_RCD_COUNT_FIELD_NAME     {'c', 'o', 'u', 'n', 't', '='}
#define INNO_ROS_BAG_RCD_INDEX_POS_FIELD_NAME                   \
  {'i', 'n', 'd', 'e', 'x', '_', 'p', 'o', 's', '='}   // 1.2+
#define INNO_ROS_BAG_RCD_CONNECTION_COUNT_FIELD_NAME            \
  {'c', 'o', 'n', 'n', '_', 'c', 'o', 'u', 'n', 't', '='}
#define INNO_ROS_BAG_RCD_CHUNK_COUNT_FIELD_NAME                 \
  {'c', 'h', 'u', 'n', 'k', '_', 'c', 'o', 'u', 'n', 't', '='}
#define INNO_ROS_BAG_RCD_CONNECTION_FIELD_NAME  \
  {'c', 'o', 'n', 'n', '='}         // 2.0+
#define INNO_ROS_BAG_RCD_COMPRESSION_FIELD_NAME                 \
  {'c', 'o', 'm', 'p', 'r', 'e', 's', 's', 'i', 'o', 'n', '='}
#define INNO_ROS_BAG_RCD_SIZE_FIELD_NAME        \
  {'s', 'i', 'z', 'e', '='}        // 2.0+
#define INNO_ROS_BAG_RCD_TIME_FIELD_NAME        \
  {'t', 'i', 'm', 'e', '='}         // 2.0+
#define INNO_ROS_BAG_RCD_START_TIME_FIELD_NAME                  \
  {'s', 't', 'a', 'r', 't', '_', 't', 'i', 'm', 'e', '='}
#define INNO_ROS_BAG_RCD_END_TIME_FIELD_NAME                    \
  {'e', 'n', 'd', '_', 't', 'i', 'm', 'e', '='}    // 2.0+
#define INNO_ROS_BAG_RCD_CHUNK_POS_FIELD_NAME                           \
  {'c', 'h', 'u', 'n', 'k', '_', 'p', 'o', 's', '='}     // 2.0+
#define INNO_ROS_BAG_RCD_ENCRYPTOR_FIELD_NAME                   \
  {'e', 'n', 'c', 'r', 'y', 'p', 't', 'o', 'r', '='}   // 2.0+
#define INNO_ROS_BAG_RCD_FRAME_ID_INNONVSION            \
  {'i', 'n', 'n', 'o', 'v', 'u', 's', 'i', 'o', 'n'}
// connect data
#define INNO_ROS_BAG_RCD_CALLER_ID              \
  {'c', 'a', 'l', 'l', 'e', 'r', 'i', 'd', '='}
#define INNO_ROS_BAG_RCD_LATCHING               \
  {'l', 'a', 't', 'c', 'h', 'i', 'n', 'g', '='}
#define INNO_ROS_BAG_RCD_TOPIC                  \
  {'t', 'o', 'p', 'i', 'c', '='}
#define INNO_ROS_BAG_RCD_TYPE                   \
  {'t', 'y', 'p', 'e', '='}
#define INNO_ROS_BAG_RCD_MD5SUM                 \
  {'m', 'd', '5', 's', 'u', 'm', '='}
#define INNO_ROS_BAG_RCD_MESSAGE_DEFINITION               {     \
    'm', 'e', 's', 's', 'a', 'g', 'e',  '_',                    \
    'd', 'e', 'f', 'i', 'n', 'i', 't', 'i', 'o', 'n', '='}
#define INNO_ROS_BAG_RCD_MD5SUM_VALUE       {                           \
    '1', '1', '5', '8', 'd', '4', '8', '6', 'd', 'd', '5', '1', 'd',    \
    '6', '8', '3', 'c', 'e', '2', 'f', '1', 'b', 'e', '6', '5', '5',    \
    'c', '3', 'c', '1', '8', '1'}

#define INNO_ROS_BAG_RCD_VERSION                          "2.0"
#define INNO_ROS_BAG_RCD_INNO_POINT_X                     "x"
#define INNO_ROS_BAG_RCD_INNO_POINT_Y                     "y"
#define INNO_ROS_BAG_RCD_INNO_POINT_Z                     "z"
#define INNO_ROS_BAG_RCD_INNO_POINT_ITIMESTAMP            "timestamp"
#define INNO_ROS_BAG_RCD_INNO_POINT_INTENSITY             "intensity"
#define INNO_ROS_BAG_RCD_INNO_POINT_FLAGS                 "flags"
#define INNO_ROS_BAG_RCD_INNO_POINT_SCAN_ID               "scan_id"
#define INNO_ROS_BAG_RCD_INNO_POINT_SCAN_IDX              "scan_idx"
#define INNO_ROS_BAG_RCD_INNO_POINT_IS_2ND_RETURN         "is_2nd_return"

typedef int (*RosbagRecorderCallback)(void *context,
                                      const void *buffer,
                                      size_t buffer_len);

class RosbagRecorder : public RecorderBase {
 private:
  enum OpRosBag {
  OP_MSG_None    = 0,
  OP_MSG_DATA    = 0x02,
  OP_FILE_HEADER,
  OP_INDEX_DATA,
  OP_CHUNK,
  OP_CHUNK_INFO,
  OP_CONNECTION   = 0x07,
};

enum CompressionType {
  Uncompressed = 0,
  BZ2          = 1,
  LZ4          = 2,
};

struct PCL_ADD_POINT4D {
  float             x;
  float             y;
  float             z;
  uint32_t          reserved;
};

typedef struct PointXYZIT {
  PCL_ADD_POINT4D  pcl_point4d;                    // quad-word XYZ
  double           timestamp;
  uint16_t         intensity;
  uint8_t          flags;
  uint8_t          is_2nd_return;
  uint16_t         scan_id;
  uint16_t         scan_idx;
} PointXYZIT;

#pragma pack(push)
#pragma pack(1)
typedef struct  BagHeader {
  uint32_t      chunk_count_len;
  char          chunk_count_name[12];
  uint32_t      chunk_count_value;
  uint32_t      conn_count_len;
  char          conn_count_name[11];
  uint32_t      conn_count_value;
  uint32_t      index_pos_len;
  char          index_pos_name[10];
  uint64_t      index_pos_value;
  uint32_t      op_len;
  char          op_name[3];
  uint8_t       op_value;
} BagHeader;

typedef struct ChunkHeader1 {
  uint32_t      size_len;
  char          size_name[5];
  uint32_t      size_value;
} ChunkHeader1;

typedef struct ChunkHeader2 {
  uint32_t      compression_len;
  char          compression_name[12];
} ChunkHeader2;

typedef struct ChunkHeader3 {
  uint32_t      op_len;
  char          op_name[3];
  uint8_t       op_value;
} ChunkHeader3;

typedef struct ConnectHeader {
  uint32_t     topic_len;
  char         topic_name[6];
  char         topic_value[10];
  uint32_t     conn_len;
  char         conn_name[5];
  uint32_t     conn_value;
  uint32_t     op_len;
  char         op_name[3];
  uint8_t      op_value;
} ConnectHeader;

typedef struct ConnectCallerID {
  uint32_t     callerid_len;
  char         callerid_name[9];
} ConnectCallerID;

typedef struct ConnectLatching {
  uint32_t     latching_len;
  char         latching_name[9];
  char         latching_value;
} ConnectLatching;

typedef struct ConnectMD5Sum {
  uint32_t     md5sum_len;
  char         md5sum_name[7];
  char         md5sum_value[32];
} ConnectMD5Sum;

typedef struct ConnectMessageDefinition {
  uint32_t     message_len;
  char         message_definition[19];
} ConnectMessageDefinition;

typedef struct ConnectTopic {
  uint32_t     topic_len;
  char         topic_name[6];
} ConnectTopic;

typedef struct ConnectType {
  uint32_t     type_len;
  char         type_name[5];
} ConnectType;

typedef struct MessageDataHeader {
  uint32_t     time_len;
  char         time_name[5];
  uint64_t     time_value;
  uint32_t     conn_len;
  char         conn_name[5];
  uint32_t     conn_value;
  uint32_t     op_len;
  char         op_name[3];
  uint8_t      op_value;
} MessageDataHeader;

typedef struct IndexDataHeader {
  uint32_t     count_len;
  char         count_name[6];
  uint32_t     count_value;
  uint32_t     ver_len;
  char         ver_name[4];
  uint32_t     ver_value;
  uint32_t     conn_len;
  char         conn_name[5];
  uint32_t     conn_value;
  uint32_t     op_len;
  char         op_name[3];
  uint8_t      op_value;
} IndexDataHeader;

typedef struct IndexDataData {
  uint64_t     time;
  uint32_t     offset;
} IndexDataData;

typedef struct ChunkInfoHeader {
  uint32_t     count_len;
  char         count_name[6];
  uint32_t     count_value;
  uint32_t     ver_len;
  char         ver_name[4];
  uint32_t     ver_value;
  uint32_t     start_time_len;
  char         start_time_name[11];
  uint64_t     start_time_value;
  uint32_t     chunk_pos_len;
  char         chunk_pos_name[10];
  uint64_t     chunk_pos_value;
  uint32_t     end_time_len;
  char         end_time_name[9];
  uint64_t     end_time_value;
  uint32_t     op_len;
  char         op_name[3];
  uint8_t      op_value;
} ChunkInfoHeader;

typedef struct ChunkInfoData {
  uint32_t     conn;
  uint32_t     count;
} ChunkInfoData;

typedef struct StdMsgsHeader {
  uint32_t     seq;
  uint64_t     time;
  uint32_t     frame_id_len;
  char         frame_id[10];
} StdMsgsHeader;

typedef struct StdMsgsPointField {
  uint32_t     name_len;
  char         name[20];
  uint32_t     offset;
  uint8_t      datatype;
  uint32_t     count;
} StdMsgsPointField;

typedef struct StdMsgs {
  uint32_t     height;
  uint32_t     width;
  uint32_t     pointfield_len;
  bool         is_bigendian;
  uint32_t     point_step;
  uint32_t     row_step;
  bool         is_dense;
} StdMsgs;

#pragma pack(4)
 private:
  static const size_t kBagBufferSize = 1024 * 1024 * 10;
  static const uint32_t kBagHeadPadLen = 0x0fbb;
  static const uint32_t kBagOffset = 0x0991;
  static const uint32_t kUnit32Len = 4;
  static constexpr double kUsInSecondC = 1000000.0;
  static constexpr double kTenUsInSecondC = 100000.0;
  static constexpr char* kUncompressedNone = const_cast<char *>("none");
  static constexpr char* kTopicValue = const_cast<char *>("/iv_points");
  static constexpr char* kTypeValue = const_cast<char *>
                                      ("sensor_msgs/PointCloud2");
  static constexpr char* kCralleridValue = const_cast<char *>
                                           ("/innovusion_nodelet_manager");
  // don't change anything
  static constexpr char* kMesDefValue =
      const_cast<char *>
      (R"(# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its layout described by the
# contents of the "fields" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
Header header

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: sensor_msgs/PointField
# This message holds the description of one point entry in the
# PointCloud2 message format.
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8

string name      # Name of field
uint32 offset    # Offset from start of point struct
uint8  datatype  # Datatype enumeration, see above
uint32 count     # How many elements in the field
)");

 public:
  RosbagRecorder(const char *filename,
                 RosbagRecorderCallback cb,
                 void *cb_ctx,
                 ssize_t size_limit_in_m);
  virtual ~RosbagRecorder();
  virtual int add_block(const InnoDataPacket *pkt);
  uint64_t get_written_size() const {
    return write_file_size_;
  }

 private:
  void start_writing_();
  void write_version_();
  virtual void close_file_();
  virtual int flush_buffer_();
  int write_file_header_record_(uint32_t count, uint64_t postion);
  void write_chunk_header_record_(CompressionType type);
  void process_inno_data_packet_(const InnoDataPacket *pkt);
  void update_ps_value_(uint32_t point_count);
  int start_writing_chunk_(const InnoDataPacket *cframe);
  void write_connection_record_();
  void write_message_data_record_(uint32_t data_len);
  void write_message_data_header_();
  void write_message_data_msg_(const InnoDataPacket *cframe);
  void write_index_record_();
  int stop_writing_chunk_();
  void write_chunk_info_records_();
  int stop_writing_();
  template<typename T>
  void write_value_(T value);
  void write_data_(const void *head, uint32_t len);
  void write_value_at_position_(uint32_t position, uint32_t value);
  void add_xyz_point_(void *ctx,
                      const InnoDataPacket &pkt,
                      const InnoXyzPoint &pt);
  void add_xyz_point_(void *ctx, const InnoDataPacket &pkt, const InnoEnXyzPoint &pt);
  void add_cpoint_(void *ctx,
                   const InnoDataPacket &pkt,
                   const InnoBlock &block,
                   const InnoChannelPoint &pt,
                   const InnoBlockFullAngles &fa,
                   const uint16_t ch,
                   const uint16_t m);
  void add_cpoint_(void *ctx, const InnoDataPacket &pkt, const InnoEnBlock &block, const InnoEnChannelPoint &pt,
                   const InnoBlockFullAngles &fa, const uint16_t ch, const uint16_t m);

 private:
  std::ofstream file_;
  RosbagRecorderCallback write_callback_;
  void *write_callback_ctx_;
  uint64_t file_header_pos_;
  uint64_t file_chunk_begin_pos_;
  uint32_t chunk_count_;
  uint32_t chunk_points_;
  uint32_t current_chunk_size_;
  uint32_t current_point_in_frame_;
  bool bag_full_;
  double max_range_;
  double min_range_;
  uint64_t write_file_size_;
  uint32_t message_data_row_step_positon_;
  uint32_t message_data_width_positon_;
  uint32_t message_data_data_position_;
  uint32_t chunk_data_len_position_;
  uint32_t chunk_size_position_;
  uint32_t message_data_msg_seq_position_;
  uint32_t last_index_postion_;
  char pt_bagbuffer_[kBagBufferSize];
  std::vector<uint64_t> vchunk_pos_;
  std::vector<uint64_t> vstart_time_;
  std::vector<uint64_t> vend_time_;

  int64_t last_packet_ts_us_;
  int64_t last_2nd_packet_ts_us_;
  int64_t packet_offset_ts_us_;
  uint64_t last_time_ns_;

  BagHeader bag_header_ = {
    16, // chunk_count_len
    INNO_ROS_BAG_RCD_CHUNK_COUNT_FIELD_NAME, // chunk_count_name
    0, // chunk_count_value
    15, // conn_count_len
    INNO_ROS_BAG_RCD_CONNECTION_COUNT_FIELD_NAME, // conn_count_name
    1, // conn_count_value
    18, // index_pos_len
    INNO_ROS_BAG_RCD_INDEX_POS_FIELD_NAME, // index_pos_name
    0, // index_pos_value
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_FILE_HEADER, // op_value
  };

  ChunkHeader1 chunk_header1_ = {
    9, // size_len
    INNO_ROS_BAG_RCD_SIZE_FIELD_NAME, // size_name
    0, // size_value
  };

  ChunkHeader2 chunk_header2_ = {
    17, // compression_len
    INNO_ROS_BAG_RCD_COMPRESSION_FIELD_NAME, // compression_name
  };

  ChunkHeader3 chunk_header3_ = {
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_CHUNK, // op_value
  };

  ConnectHeader connect_header_ = {
    0, // topic_len
    INNO_ROS_BAG_RCD_TOPIC_FIELD_NAME, // topic_name
    {0}, // topic_value
    9, // conn_len
    INNO_ROS_BAG_RCD_CONNECTION_FIELD_NAME, // conn_name
    0, // conn_value
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_CONNECTION // op_value
  };

  ConnectCallerID connect_caller_id_ = {
    0, // callerid_len
    INNO_ROS_BAG_RCD_CALLER_ID // callerid_name
  };

  ConnectLatching connect_latching_ = {
    10, // latching_len
    INNO_ROS_BAG_RCD_LATCHING, // latching_name
    '0' // latching_value
  };

  ConnectMD5Sum connect_md5sum_ = {
    39, // md5sum_len,
    INNO_ROS_BAG_RCD_MD5SUM, // md5sum_name
    INNO_ROS_BAG_RCD_MD5SUM_VALUE // md5sum_value
  };

  ConnectMessageDefinition connect_message_definition_ = {
    0, // message_len
    INNO_ROS_BAG_RCD_MESSAGE_DEFINITION // message_definition
  };

  ConnectTopic connect_topic_ = {
    0, // topic_len
    INNO_ROS_BAG_RCD_TOPIC // topic_name
  };

  ConnectType connect_type_ = {
    0, // type_len
    INNO_ROS_BAG_RCD_TYPE // type_name
  };

  MessageDataHeader message_data_header_ = {
    13, // time_len,
    INNO_ROS_BAG_RCD_TIME_FIELD_NAME, // time_name
    0, // time_value,
    9, // conn_len,
    INNO_ROS_BAG_RCD_CONNECTION_FIELD_NAME, // conn_name
    0, // conn_value
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_MSG_DATA // op_value
  };

  IndexDataHeader index_data_header_ = {
    10, // count_len
    INNO_ROS_BAG_RCD_COUNT_FIELD_NAME, // count_name
    1, // count_value
    8, // ver_len
    INNO_ROS_BAG_RCD_VER_FIELD_NAME, // ver_name
    1, // ver_value
    9, // conn_len
    INNO_ROS_BAG_RCD_CONNECTION_FIELD_NAME, // conn_name
    0, // conn_value
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_INDEX_DATA // op_value
  };

  IndexDataData index_data_data_;
  ChunkInfoHeader chunk_info_header_ = {
    10, // count_len
    INNO_ROS_BAG_RCD_COUNT_FIELD_NAME, // count_name
    1, // count_value
    8, // ver_len
    INNO_ROS_BAG_RCD_VER_FIELD_NAME, // ver_name
    1, // ver_value
    19, // start_time_len
    INNO_ROS_BAG_RCD_START_TIME_FIELD_NAME, // start_time_name
    0, // start_time_value
    18, // chunk_pos_len
    INNO_ROS_BAG_RCD_CHUNK_POS_FIELD_NAME, // chunk_pos_name
    0, // chunk_pos_value
    17, // end_time_len
    INNO_ROS_BAG_RCD_END_TIME_FIELD_NAME, // end_time_name
    0, // end_time_value
    4, // op_len
    INNO_ROS_BAG_RCD_OP_FIELD_NAME, // op_name
    OP_CHUNK_INFO // op_value
  };

  StdMsgsHeader std_msgs_header_ = {
    1, // seq,
    0, // time,
    10, // frame_id_len,
    INNO_ROS_BAG_RCD_FRAME_ID_INNONVSION // frame_id,
  };

  StdMsgs std_msgs_ = {
    1, // height
    0, // width
    sizeof(point_field_) / sizeof(point_field_[0]), // pointfield_len
    0, // is_bigendian
    sizeof(PointXYZIT), // point_step
    0, // row_step
    1 // is_dense
  };

  ChunkInfoData check_info_data_ = {
    0, // conn
    1 // count
  };
  // point_field in sensor_msgs/PointCloud2
  StdMsgsPointField point_field_[9] = {
    {1, INNO_ROS_BAG_RCD_INNO_POINT_X,          0,         7, 1},
    {1, INNO_ROS_BAG_RCD_INNO_POINT_Y,          0x04,      7, 1},
    {1, INNO_ROS_BAG_RCD_INNO_POINT_Z,          0x08,      7, 1},
    {9, INNO_ROS_BAG_RCD_INNO_POINT_ITIMESTAMP, 0x10,      8, 1},
    {9, INNO_ROS_BAG_RCD_INNO_POINT_INTENSITY,  0x18,      4, 1},
    {5, INNO_ROS_BAG_RCD_INNO_POINT_FLAGS,      0x1a,      2, 1},
    {13, INNO_ROS_BAG_RCD_INNO_POINT_IS_2ND_RETURN, 0x1b,  2, 1},
    {7, INNO_ROS_BAG_RCD_INNO_POINT_SCAN_ID,    0x1c,      4, 1},
    {8, INNO_ROS_BAG_RCD_INNO_POINT_SCAN_IDX,   0x1e,      4, 1},
  };
};
#pragma pack(pop)

}  // namespace innovusion

#endif  // CONVERTER_ROSBAG_RECORDER_H_
