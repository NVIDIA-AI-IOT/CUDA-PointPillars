/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#ifndef PCS_RECORDER_BASE_H_
#define PCS_RECORDER_BASE_H_

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <string>

#include "sdk_common/inno_lidar_packet.h"

namespace innovusion {

enum RecorderErrorType {
  RERCORDER_SUCCESS = 0,
  RERCORDER_FILE_IS_NOT_OPEN = 1,
  RERCORDER_SIZE_LIMIT = 2,
  RERCORDER_TYPE_ERROR = 3,
  RERCORDER_INDEX_ERROR = 4,
  RERCORDER_STREAM_ERROR = 5,
};

class RecorderBase {
 public:
  RecorderBase() {
    fd_ = -1;  // we have to init fd_!!!
    filename_ = "";
    add_block_called_ = 0;
    buffer_write_cursor_ = 0;
    total_points_ = 0;
    last_frame_id_ = 0;
    total_frame_ = 0;
    size_limit_ = 0;
    frame_id_base_ = 0;
    total_size_ = 0;
    timestamp_base_10us_ = 0;
    current_block_ts_10us_offset_ = 0;
  }
  virtual ~RecorderBase() {}
  virtual bool is_opened() const {
    return fd_ >= 0;
  }
  virtual int add_block(const InnoDataPacket *pkt) = 0;
  virtual void close_file_() = 0;
  virtual void reset_buffer_() {
    buffer_write_cursor_ = 0;
  }
  virtual int flush_buffer_() = 0;

 protected:
  int fd_;
  size_t add_block_called_;
  std::string filename_;
  ssize_t size_limit_;
  size_t buffer_write_cursor_;
  size_t total_points_;
  size_t total_frame_;
  ssize_t total_size_;
  size_t last_frame_id_;
  size_t frame_id_base_;
  size_t timestamp_base_10us_;
  size_t current_block_ts_10us_offset_;
};

}  // namespace innovusion

#endif  // PCS_RECORDER_BASE_H_
