/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include "sdk_common/converter/rosbag_recorder.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <memory>
#include <string>

#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/inno_lidar_log.h"
#include "utils/utils.h"

namespace innovusion {
RosbagRecorder::RosbagRecorder(const char *filename, RosbagRecorderCallback cb, void *cb_ctx, ssize_t size_limit_in_m)
    : RecorderBase() {
  if (filename && strlen(filename)) {
    filename_ = filename;
    if (filename_.rfind(".") == std::string::npos) {
      filename_ += ".bag";
    }
    file_.open(filename_.c_str(), std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
      inno_log_error_errno("cannot open rosbag recorder %s.", filename_.c_str());
    }
  }
  write_callback_ = cb;
  write_callback_ctx_ = cb_ctx;

  size_limit_ = size_limit_in_m * 1000 * 1000;
  if (0 == size_limit_) {
    bag_full_ = true;
  } else {
    bag_full_ = false;
  }
  write_file_size_ = 0;
  chunk_count_ = 0;
  current_chunk_size_ = 0;
  chunk_points_ = 0;
  current_point_in_frame_ = 0;
  min_range_ = 0.4;
  max_range_ = 2000.0;
  last_index_postion_ = 0;
  file_chunk_begin_pos_ = 0;
  message_data_row_step_positon_ = 0;
  chunk_data_len_position_ = 0;
  message_data_data_position_ = 0;
  message_data_width_positon_ = 0;
  file_header_pos_ = 0;
  chunk_size_position_ = 0;
  last_packet_ts_us_ = -1;
  last_2nd_packet_ts_us_ = -1;
  packet_offset_ts_us_ = 0;
  last_time_ns_ = 0;
}

RosbagRecorder::~RosbagRecorder() {
  close_file_();
  vchunk_pos_.clear();
  vstart_time_.clear();
  vend_time_.clear();
}

void RosbagRecorder::close_file_() {
#if !defined(__MINGW64__) && defined(_WIN32)
#undef close
#endif

  if (file_.is_open()) {
    if (filename_.size()) {
      inno_log_info("write %" PRI_SIZELU " frames, %" PRI_SIZELU " points, %" PRI_SIZEU " bytes to %s", total_frame_,
                    total_points_, write_file_size_, filename_.c_str());
    }
    file_.close();
    inno_log_info("bag file close()");
  }
}

int RosbagRecorder::flush_buffer_() {
  int result = RERCORDER_SUCCESS;
  inno_log_verify(buffer_write_cursor_ <= kBagBufferSize, "buffer overflow %" PRI_SIZELU, buffer_write_cursor_);
  write_file_size_ += buffer_write_cursor_;
  if (buffer_write_cursor_ == 0) {
    return result;
  }
  if (file_.is_open()) {
    file_.write(pt_bagbuffer_, buffer_write_cursor_);
  }

  if (write_callback_) {
    int ret;
    /*
      ret = PcServerWsProcessor::write_chunk_to_ws_socket_full_s(
      &wsconn_,
      pt_bagbuffer_,
      buffer_write_cursor_);
    */
    ret = write_callback_(write_callback_ctx_, pt_bagbuffer_, buffer_write_cursor_);
    if (ret < 0) {
      inno_log_warning("Cannot write rosbag recorder to socket.");
      result = RERCORDER_STREAM_ERROR;
    }
  }

  buffer_write_cursor_ = 0;
  return result;
}

int RosbagRecorder::add_block(const InnoDataPacket *pkt) {
  int result = 0;
  if ((filename_ != "") && (!file_.is_open())) {
    return RERCORDER_FILE_IS_NOT_OPEN;
  }
  if (bag_full_) {
    return RERCORDER_SIZE_LIMIT;
  }
  if (pkt == NULL) {
    if (add_block_called_ == 0) {
      start_writing_chunk_(NULL);
    }
    stop_writing_chunk_();
    stop_writing_();
    bag_full_ = true;
    return RERCORDER_SIZE_LIMIT;
  }
  if (!CHECK_XYZ_POINTCLOUD_DATA(pkt->type) && !CHECK_SPHERE_POINTCLOUD_DATA(pkt->type)) {
    return RERCORDER_TYPE_ERROR;
  }
  if (pkt->idx > 1000000000L) {
    // xxx todo: ask gang to fix
    return RERCORDER_INDEX_ERROR;
  }

  uint64_t time_ns_ = InnoUtils::get_time_ns();

  size_t ts_10us = pkt->common.ts_start_us / 10;
  last_2nd_packet_ts_us_ = last_packet_ts_us_;

  if (last_packet_ts_us_ < 0) {
    last_packet_ts_us_ = pkt->common.ts_start_us;
  } else {
    if (pkt->common.ts_start_us > last_packet_ts_us_ + packet_offset_ts_us_ + 1000000 * 1000L) {
      last_packet_ts_us_ = last_packet_ts_us_ + (time_ns_ - last_time_ns_) / 1000;
    } else {
      last_packet_ts_us_ = pkt->common.ts_start_us - packet_offset_ts_us_;
    }
    packet_offset_ts_us_ = pkt->common.ts_start_us - last_packet_ts_us_;
  }
  last_time_ns_ = time_ns_;
  inno_log_trace("offset=%" PRI_SIZED " ts=%lf", packet_offset_ts_us_ / 1000000, pkt->common.ts_start_us / 1000000);

  if (0 == add_block_called_) {  // first frame
    frame_id_base_ = pkt->idx;
    timestamp_base_10us_ = ts_10us;
    last_frame_id_ = pkt->idx;
    total_frame_ = 1;
    start_writing_();
    result = start_writing_chunk_(pkt);
    if (result > 0) {
      return result;
    }
  }
  add_block_called_++;
  if (last_frame_id_ != pkt->idx) {
    // new frame
    result = stop_writing_chunk_();
    if (result > 0) {
      return result;
    }
    result = start_writing_chunk_(pkt);
    if (result > 0) {
      return result;
    }
    inno_log_info("capture %u points in frame-%" PRI_SIZELU "-%" PRI_SIZELU, current_point_in_frame_, total_frame_,
                  last_frame_id_);

    total_frame_++;
    current_point_in_frame_ = 0;
    last_frame_id_ = pkt->idx;
  }
  size_t pcount = 0;
  current_block_ts_10us_offset_ = ts_10us - timestamp_base_10us_;

  // need one byte is_dense of sensor_msgs
  static const uint32_t kBufferMargin = 32;

  if (filename_.size() == 0 || file_.is_open()) {
    if (buffer_write_cursor_ + sizeof(PointXYZIT) * pkt->item_number + kBufferMargin > sizeof(pt_bagbuffer_)) {
      inno_log_error(
          "pt_bagbuffer_ is too small! "
          "please increase its size. pt=%u",
          pkt->item_number);
    } else {
      if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
        ITERARATE_INNO_DATA_PACKET_CPOINTS(add_cpoint_, NULL, pkt, pcount);
      } else if (CHECK_EN_SPHERE_POINTCLOUD_DATA(pkt->type)) {
        ITERARATE_INNO_DATA_PACKET_EN_CPOINTS(add_cpoint_, NULL, pkt, pcount);
      } else if (CHECK_EN_XYZ_POINTCLOUD_DATA(pkt->type)) {
        ITERARATE_INNO_DATA_PACKET_EN_XYZ_POINTS(add_xyz_point_, NULL, pkt);
      } else {
        ITERARATE_INNO_DATA_PACKET_XYZ_POINTS(add_xyz_point_, NULL, pkt);
      }
    }
  }
  return RERCORDER_SUCCESS;
}

inline void RosbagRecorder::add_xyz_point_(void *ctx, const InnoDataPacket &pkt, const InnoXyzPoint &pt) {
  if (pt.radius == 0) {
    return;
  }
  current_point_in_frame_++;
  double current_ts_start = pkt.common.ts_start_us / kUsInSecondC;
  PointXYZIT *point = reinterpret_cast<PointXYZIT *>(&pt_bagbuffer_[buffer_write_cursor_]);
  point->flags = pt.channel | (pt.in_roi == 3 ? (1 << 2) : 0) | (pt.facet << 3) | (pt.type << 6);
  point->intensity = pt.refl;
  point->timestamp = pt.ts_10us / kTenUsInSecondC + current_ts_start;
  point->pcl_point4d.x = pt.x;  // up
  point->pcl_point4d.y = pt.y;  // right
  point->pcl_point4d.z = pt.z;  // forward
  point->scan_id = pt.scan_id;
  point->scan_idx = pt.scan_idx;
  point->is_2nd_return = pt.is_2nd_return;
  chunk_points_++;
  total_points_++;
  buffer_write_cursor_ += sizeof(PointXYZIT);
}

inline void RosbagRecorder::add_xyz_point_(void *ctx, const InnoDataPacket &pkt, const InnoEnXyzPoint &pt) {
  if (pt.radius == 0) {
    return;
  }
  current_point_in_frame_++;
  double current_ts_start = pkt.common.ts_start_us / kUsInSecondC;
  PointXYZIT *point = reinterpret_cast<PointXYZIT *>(&pt_bagbuffer_[buffer_write_cursor_]);
  point->flags = pt.channel | (pt.in_roi == 3 ? (1 << 2) : 0) | (pt.facet << 3);
  point->intensity = pkt.use_reflectance ? pt.reflectance : pt.intensity;
  point->timestamp = pt.ts_10us / kTenUsInSecondC + current_ts_start;
  point->pcl_point4d.x = pt.x;  // up
  point->pcl_point4d.y = pt.y;  // right
  point->pcl_point4d.z = pt.z;  // forward
  point->scan_id = pt.scan_id;
  point->scan_idx = pt.scan_idx;
  point->is_2nd_return = pt.is_2nd_return;
  chunk_points_++;
  total_points_++;
  buffer_write_cursor_ += sizeof(PointXYZIT);
}

inline void RosbagRecorder::add_cpoint_(void *ctx, const InnoDataPacket &pkt, const InnoBlock &block,
                                        const InnoChannelPoint &pt, const InnoBlockFullAngles &fa, const uint16_t ch,
                                        const uint16_t m) {
  if (pt.radius == 0) {
    return;
  }
  current_point_in_frame_++;
  double current_ts_start = pkt.common.ts_start_us / kUsInSecondC;
  PointXYZIT *point = reinterpret_cast<PointXYZIT *>(&pt_bagbuffer_[buffer_write_cursor_]);
  int roi = block.header.in_roi == 3 ? (1 << 2) : 0;
  // point->scan_id = block.header.scan_id;
  // point->scan_idx = block.header.scan_idx;
  point->flags = ch | roi | (block.header.facet << 3) | (pt.type << 6);
  point->intensity = pt.refl;
  point->timestamp = block.header.ts_10us / kTenUsInSecondC + current_ts_start;
  InnoXyzrD xyzr;
  InnoDataPacketUtils::get_xyzr_meter(fa.angles[ch], pt.radius, ch, &xyzr);
  point->pcl_point4d.x = xyzr.x;  // up
  point->pcl_point4d.y = xyzr.y;  // right
  point->pcl_point4d.z = xyzr.z;  // forward
  point->scan_id = block.header.scan_id;
  point->scan_idx = block.header.scan_idx;
  point->is_2nd_return = pt.is_2nd_return;
  chunk_points_++;
  total_points_++;
  buffer_write_cursor_ += sizeof(PointXYZIT);
  return;
}

inline void RosbagRecorder::add_cpoint_(void *ctx, const InnoDataPacket &pkt, const InnoEnBlock &block,
                                        const InnoEnChannelPoint &pt, const InnoBlockFullAngles &fa, const uint16_t ch,
                                        const uint16_t m) {
  if (pt.radius == 0) {
    return;
  }

  current_point_in_frame_++;
  double current_ts_start = pkt.common.ts_start_us / kUsInSecondC;
  PointXYZIT *point = reinterpret_cast<PointXYZIT *>(&pt_bagbuffer_[buffer_write_cursor_]);
  int roi = block.header.in_roi == 3 ? (1 << 2) : 0;
  // point->scan_id = block.header.scan_id;
  // point->scan_idx = block.header.scan_idx;
  point->flags = ch | roi | (block.header.facet << 3) | (pt.type << 6);
  point->intensity = pkt.use_reflectance ? pt.reflectance : pt.intensity;
  point->timestamp = block.header.ts_10us / kTenUsInSecondC + current_ts_start;
  InnoXyzrD xyzr;
  if (pkt.type == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD) {
    InnoDataPacketUtils::get_xyzr_meter(fa.angles[ch], pt.radius, ch, &xyzr, static_cast<InnoItemType>(pkt.type));
    point->scan_id = block.header.scan_id;
  } else {
    // for robin
    // uint32_t scan_id = block.header.scan_id * 4 + ch;
    DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(pkt.type);
    int index = block.header.scan_id * 4 + ch;
    uint32_t scan_id = channel_mapping[index] + block.header.facet * tdc_channel_number;
    InnoDataPacketUtils::get_xyzr_meter(fa.angles[ch], pt.radius, scan_id, &xyzr, static_cast<InnoItemType>(pkt.type));
    point->scan_id = scan_id;
  }

  point->pcl_point4d.x = xyzr.x;  // up
  point->pcl_point4d.y = xyzr.y;  // right
  point->pcl_point4d.z = xyzr.z;  // forward
  point->scan_idx = block.header.scan_idx;
  point->is_2nd_return = pt.is_2nd_return;
  chunk_points_++;
  total_points_++;
  buffer_write_cursor_ += sizeof(PointXYZIT);
  return;
}

void RosbagRecorder::start_writing_() {
  write_version_();
  file_header_pos_ = buffer_write_cursor_;
  write_file_header_record_(0, 0);
  reset_buffer_();
}

void RosbagRecorder::write_version_() {
  std::string version = std::string("#ROSBAG V") + INNO_ROS_BAG_RCD_VERSION + std::string("\n");
  inno_log_trace(" ROS Bag VERSION is %s", version.c_str());
  write_data_(version.c_str(), version.length());
}

int RosbagRecorder::write_file_header_record_(uint32_t count, uint64_t postion) {
  int result = 0;
  std::string padding;
  uint32_t bag_header_len = sizeof(BagHeader);

  write_value_(bag_header_len);
  bag_header_.chunk_count_value = count;
  bag_header_.index_pos_value = postion;
  write_data_(&bag_header_, sizeof(bag_header_));
  // not first write the file header
  if (count > 0) {
    return flush_buffer_();
  }
  padding.resize(kBagHeadPadLen, ' ');  // pad 0x20
  write_value_(kBagHeadPadLen);
  write_data_(padding.c_str(), kBagHeadPadLen);
  file_chunk_begin_pos_ += buffer_write_cursor_;
  result = flush_buffer_();
  if (result > 0) {
    return result;
  }
  return result;
}

int RosbagRecorder::start_writing_chunk_(const InnoDataPacket *cframe) {
  if (filename_ != "" && !file_.is_open()) {
    return RERCORDER_FILE_IS_NOT_OPEN;
  }
  total_size_ = write_file_size_ + sizeof(IndexDataHeader) + message_data_row_step_positon_ - chunk_data_len_position_ +
                (chunk_count_ + 1) * sizeof(ChunkInfoHeader);
  if (size_limit_ > 0) {
    if (total_size_ > size_limit_) {
      inno_log_trace("reach size limit %" PRI_SIZELD " %" PRI_SIZEU " %u %u %u %" PRI_SIZELD, total_size_,
                     write_file_size_, message_data_row_step_positon_, chunk_data_len_position_, chunk_count_,
                     size_limit_);
      stop_writing_();
      bag_full_ = true;
      return RERCORDER_SIZE_LIMIT;
    }
  }

  struct timespec spec;
  if (last_packet_ts_us_ >= 0) {
    InnoUtils::us_to_timespec(last_packet_ts_us_, &spec);
  } else {
    // clock_gettime(CLOCK_REALTIME, &spec);
    auto now = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    InnoUtils::us_to_timespec(us, &spec);
  }
  uint64_t chunk_start_time = spec.tv_sec + ((uint64_t(spec.tv_nsec)) << 32);
  file_chunk_begin_pos_ = file_chunk_begin_pos_ + current_chunk_size_;
  vchunk_pos_.push_back(file_chunk_begin_pos_);
  vstart_time_.push_back(chunk_start_time);
  chunk_info_header_.start_time_value = chunk_start_time;
  chunk_count_++;
  current_chunk_size_ = 0;
  chunk_points_ = 0;
  reset_buffer_();
  write_chunk_header_record_(Uncompressed);
  if (1 == chunk_count_) {  // only first chunk have connect record
    write_connection_record_();
  }
  write_message_data_header_();
  if (cframe) {
    write_message_data_msg_(cframe);
  }
  return RERCORDER_SUCCESS;
}

template <typename T>
inline void RosbagRecorder::write_value_(T value) {
  memcpy(&pt_bagbuffer_[buffer_write_cursor_], &value, sizeof(T));
  buffer_write_cursor_ += sizeof(T);
}

inline void RosbagRecorder::write_data_(const void *head, uint32_t len) {
  memcpy(&pt_bagbuffer_[buffer_write_cursor_], head, len);
  buffer_write_cursor_ += len;
}

inline void RosbagRecorder::write_value_at_position_(uint32_t position, uint32_t value) {
  memcpy(&pt_bagbuffer_[position], &value, kUnit32Len);
}

void RosbagRecorder::write_chunk_header_record_(CompressionType type) {
  uint32_t chunk_header_len = sizeof(ChunkHeader1) + sizeof(ChunkHeader2) + sizeof(ChunkHeader3);
  const char *str_type = nullptr;
  switch (type) {
    case Uncompressed:
      str_type = kUncompressedNone;  // the default the value
      break;
    case BZ2:
    case LZ4:
    default:
      inno_log_panic("Not support this compressed %d\n", type);
  }
  inno_log_verify(str_type, "get compress type failed!");
  uint32_t type_len = strlen(str_type);
  chunk_header_len = chunk_header_len + type_len;
  write_value_(chunk_header_len);
  chunk_header1_.size_value = 0;
  write_data_(&chunk_header1_, sizeof(ChunkHeader1));
  // record the chunk header size position
  chunk_size_position_ = buffer_write_cursor_ - kUnit32Len;
  chunk_header2_.compression_len = sizeof(chunk_header2_.compression_name) + type_len;
  write_data_(&chunk_header2_, sizeof(ChunkHeader2));
  write_data_(str_type, type_len);
  write_data_(&chunk_header3_, sizeof(ChunkHeader3));
  // write_chunk data len
  chunk_data_len_position_ = buffer_write_cursor_;
  write_value_(chunk_header1_.size_value);
}

void RosbagRecorder::write_connection_record_() {
  uint32_t connect_header_len = sizeof(ConnectHeader);
  uint32_t connect_data_len = 0;
  const char *strtopic = kTopicValue;
  const char *strtype = kTypeValue;
  const char *strcallerid = kCralleridValue;
  const char *strmesdef = kMesDefValue;

  write_value_(connect_header_len);
  // topic value
  connect_header_.topic_len = sizeof(connect_header_.topic_name) + strlen(strtopic);
  inno_log_verify(strlen(strtopic) <= sizeof(connect_header_.topic_value), "topic value size too small!");
  memcpy(connect_header_.topic_value, strtopic, strlen(strtopic));
  write_data_(&connect_header_, sizeof(ConnectHeader));
  connect_data_len = sizeof(ConnectCallerID) + sizeof(ConnectMessageDefinition) + sizeof(ConnectType) +
                     sizeof(ConnectLatching) + sizeof(ConnectMD5Sum) + sizeof(ConnectTopic) + strlen(strtopic) +
                     strlen(strtype) + strlen(strcallerid) + strlen(strmesdef);
  write_value_(connect_data_len);
  // message_definition
  connect_message_definition_.message_len = sizeof(connect_message_definition_.message_definition) + strlen(strmesdef);
  write_data_(&connect_message_definition_, sizeof(connect_message_definition_));
  write_data_(strmesdef, strlen(strmesdef));
  // callerid
  connect_caller_id_.callerid_len = sizeof(connect_caller_id_.callerid_name) + strlen(strcallerid);
  write_data_(&connect_caller_id_, sizeof(ConnectCallerID));
  write_data_(strcallerid, strlen(strcallerid));
  // latching
  write_data_(&connect_latching_, sizeof(connect_latching_));
  // md5sum
  write_data_(&connect_md5sum_, sizeof(connect_md5sum_));
  // topic
  connect_topic_.topic_len = sizeof(connect_topic_.topic_name) + strlen(strtopic);
  write_data_(&connect_topic_, sizeof(connect_topic_));
  write_data_(strtopic, strlen(strtopic));
  // type
  connect_type_.type_len = sizeof(connect_type_.type_name) + strlen(strtype);
  write_data_(&connect_type_, sizeof(connect_type_));
  write_data_(strtype, strlen(strtype));
}

void RosbagRecorder::write_message_data_header_() {
  uint32_t message_data_header_len = sizeof(MessageDataHeader);
  uint32_t message_data_data_len = 0;
  // message data header
  write_value_(message_data_header_len);
  message_data_header_.time_value = chunk_info_header_.start_time_value;
  write_data_(&message_data_header_, message_data_header_len);
  message_data_data_position_ = buffer_write_cursor_;
  write_value_(message_data_data_len);
}

void RosbagRecorder::write_message_data_msg_(const InnoDataPacket *cframe) {
  std_msgs_header_.seq = cframe->idx;

  uint64_t header_time = InnoUtils::get_time_ns();
  std_msgs_header_.time = header_time;
  write_data_(&std_msgs_header_, sizeof(StdMsgsHeader));
  write_value_(std_msgs_.height);
  message_data_width_positon_ = buffer_write_cursor_;
  write_value_(std_msgs_.width);
  write_value_(std_msgs_.pointfield_len);
  for (uint8_t i = 0; i < std_msgs_.pointfield_len; i++) {
    write_value_(point_field_[i].name_len);
    write_data_(&point_field_[i].name, strlen(point_field_[i].name));
    write_value_(point_field_[i].offset);
    write_value_(point_field_[i].datatype);
    write_value_(point_field_[i].count);
  }
  write_value_(std_msgs_.is_bigendian);
  write_value_(std_msgs_.point_step);
  message_data_row_step_positon_ = buffer_write_cursor_;
  write_value_(std_msgs_.row_step);
  // message data len
  write_value_(std_msgs_.row_step);
}

void RosbagRecorder::update_ps_value_(uint32_t point_count) {
  uint32_t width_value = point_count;
  uint32_t point_step = width_value * sizeof(PointXYZIT);
  // len self and one byte is_dense of sensor_msgs
  uint32_t data_position_value =
      point_step + message_data_row_step_positon_ - message_data_data_position_ + kUnit32Len + 1;
  uint32_t chunk_data_ps_value = data_position_value + message_data_data_position_ - chunk_data_len_position_;
  // msgs data len
  write_value_at_position_(message_data_row_step_positon_ + kUnit32Len, point_step);
  write_value_at_position_(message_data_row_step_positon_, point_step);
  write_value_at_position_(message_data_width_positon_, width_value);
  write_value_at_position_(message_data_data_position_, data_position_value);
  write_value_at_position_(chunk_data_len_position_, chunk_data_ps_value);
  write_value_at_position_(chunk_size_position_, chunk_data_ps_value);
}

void RosbagRecorder::write_index_record_() {
  uint32_t index_header_len = sizeof(IndexDataHeader);
  uint32_t index_data_len = sizeof(IndexDataData);
  write_value_(index_header_len);
  write_data_(&index_data_header_, index_header_len);
  write_value_(index_data_len);
  index_data_data_.time = vend_time_[chunk_count_ - 1];
  index_data_data_.offset = chunk_count_ == 1 ? kBagOffset : 0;
  write_data_(&index_data_data_, index_data_len);
}

int RosbagRecorder::stop_writing_chunk_() {
  int result = 0;
  struct timespec spec;
  if (last_2nd_packet_ts_us_ >= 0) {
    InnoUtils::us_to_timespec(last_2nd_packet_ts_us_, &spec);
  } else if (last_packet_ts_us_ >= 0) {
    InnoUtils::us_to_timespec(last_packet_ts_us_, &spec);
  } else {
    // clock_gettime(CLOCK_REALTIME, &spec);
    auto now = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    InnoUtils::us_to_timespec(us, &spec);
  }
  uint64_t chunk_end_time = spec.tv_sec + ((uint64_t(spec.tv_nsec)) << 32);
  chunk_info_header_.end_time_value = chunk_end_time;
  vend_time_.push_back(chunk_end_time);
  update_ps_value_(chunk_points_);
  chunk_points_ = 0;
  write_value_(std_msgs_.is_dense);
  current_chunk_size_ += buffer_write_cursor_;
  result = flush_buffer_();
  if (result > 0) {
    return result;
  }
  write_index_record_();
  current_chunk_size_ += buffer_write_cursor_;
  result = flush_buffer_();
  if (result > 0) {
    return result;
  }
  last_index_postion_ = write_file_size_;
  return result;
}

void RosbagRecorder::write_chunk_info_records_() {
  uint32_t chunk_info_header_len = sizeof(ChunkInfoHeader);
  uint32_t chunk_info_data_len = sizeof(ChunkInfoData);

  for (uint32_t i = 0; i < chunk_count_; i++) {
    write_value_(chunk_info_header_len);
    chunk_info_header_.chunk_pos_value = vchunk_pos_[i];
    chunk_info_header_.start_time_value = vstart_time_[i];
    chunk_info_header_.end_time_value = vend_time_[i];
    write_data_(&chunk_info_header_, chunk_info_header_len);
    write_value_(chunk_info_data_len);
    write_data_(&check_info_data_, chunk_info_data_len);
  }
}

int RosbagRecorder::stop_writing_() {
  int result = 0;
  write_connection_record_();
  write_chunk_info_records_();
  result = flush_buffer_();
  if (result > 0) {
    close_file_();
    return result;
  }
  file_.seekp(file_header_pos_);
  write_file_header_record_(chunk_count_, last_index_postion_);
  result = flush_buffer_();
  if (result > 0) {
    close_file_();
    return result;
  }
  close_file_();
  return result;
}

}  // namespace innovusion
