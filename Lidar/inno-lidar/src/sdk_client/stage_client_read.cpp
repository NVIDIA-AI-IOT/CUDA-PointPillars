/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_client/stage_client_read.h"
#include "sdk_client/stage_client_read_pcap.h"

#include <thread>

#include "sdk_client/inno_lidar_packet_v1_adapt.h"
#include "sdk_client/lidar_client.h"
#include "sdk_client/lidar_client_communication.h"
#include "sdk_common/inno_lidar_packet_utils.h"

namespace innovusion {
StageClientRead::StageClientRead(InnoLidarClient *l, LidarClientCommunication *lm, void *ctx) {
  state_ = InnoLidarBase::STATE_INIT;
  lidar_ = l;
  lidar_->add_config(&config_base_);
  config_.copy_from_src(&config_base_);
  lidar_comm_ = lm;
  InputParam *param = reinterpret_cast<InputParam *>(ctx);
  source_ = param->base_param.source_type;
  create_input_(ctx);
  input_->set_misorder_correct_enable(config_.misorder_correct_enable);
}

StageClientRead::~StageClientRead(void) {
  free_cached_packet();
  if (input_) {
    delete input_;
    input_ = NULL;
  }
  lidar_comm_ = NULL;
  {
    std::unique_lock<std::mutex> lk(mutex_);
    inno_log_verify(state_ == InnoLidarBase::STATE_INIT, "invalid state=%d", state_);
  }
}

void StageClientRead::start_reading_(void) {
  std::unique_lock<std::mutex> lk(mutex_);
  inno_log_verify(state_ == InnoLidarBase::STATE_INIT, "%s state=%d", get_name_(), state_);
  state_ = InnoLidarBase::STATE_READING;
  input_->start();
  cond_.notify_all();
}

void StageClientRead::stop(void) {
  input_->stop();
  inno_log_info("%s stop", get_name_());
  std::unique_lock<std::mutex> lk(mutex_);
  if (state_ == InnoLidarBase::STATE_INIT) {
    cond_.wait(lk, [this] {
      inno_log_info("%s wait for state %d", get_name_(), state_);
      return state_ != InnoLidarBase::STATE_INIT;
    });
  }
  inno_log_verify(state_ == InnoLidarBase::STATE_READING || state_ == InnoLidarBase::STATE_STOPPED ||
                      state_ == InnoLidarBase::STATE_STOPPING,
                  "%s state=%d, forget to call start before stop?", get_name_(), state_);
  if (state_ == InnoLidarBase::STATE_READING) {
    state_ = InnoLidarBase::STATE_STOPPING;
    input_->stop();
  }
  cond_.notify_all();
  cond_.wait(lk, [this] {
    inno_log_info("%s wait for state %d", get_name_(), state_);
    return state_ != InnoLidarBase::STATE_STOPPING;
  });
}

void StageClientRead::final_cleanup(void) {
  {
    std::unique_lock<std::mutex> lk(mutex_);
    state_ = InnoLidarBase::STATE_INIT;
  }
  cond_.notify_all();
}

void StageClientRead::print_stats(void) const {
  inno_log_trace("StageRead: %s", "XXX TODO");
}

int StageClientRead::process(void *in_job, void *ctx, bool prefer) {
  // in_job is ignored
  StageClientRead *s = reinterpret_cast<StageClientRead *>(ctx);
  return s->process_job_(in_job);
}

int StageClientRead::process_job_(void *in_job) {
  int ret;
  config_.copy_from_src(&config_base_);
  input_->set_misorder_correct_enable(config_.misorder_correct_enable);
  start_reading_();
  if (source_ == SOURCE_FILE || source_ == SOURCE_PCAP) {
    inno_log_info("read from file");
    while (1) {
      ret = input_->read_data();
      if (ret == -3) {
        // reach_file_end
        lidar_->do_message_callback_fmt(INNO_MESSAGE_LEVEL_INFO, INNO_MESSAGE_CODE_READ_FILE_END, "%s reach file end",
                                        get_name_());
        ret = 0;
        break;
      }
      if (stopping_or_stopped_()) {
        break;
      }
    }
  } else if (source_ == SOURCE_UDP) {
    inno_log_info("read from udp");
    ret = input_->read_data();
  } else if (source_ == SOURCE_TCP) {
    inno_log_info("read from tcp");
    ret = input_->read_data();
  } else {
    inno_log_panic("invalid source %d", source_);
    ret = 1;
  }

  if (ret) {
    inno_log_error("ret is %d", ret);
    send_fatal_message_();
  }

  {
    std::unique_lock<std::mutex> lk(mutex_);
    inno_log_panic_if_not(state_ == InnoLidarBase::STATE_READING || state_ == InnoLidarBase::STATE_STOPPING,
                          "%s state=%d", get_name_(), state_);
    state_ = InnoLidarBase::STATE_STOPPED;
    input_->stop();
    inno_log_info("%s reader new state %d", get_name_(), state_);
  }
  cond_.notify_all();
  return ret;
}

void StageClientRead::send_fatal_message_() {
  lidar_->do_message_callback_fmt(INNO_MESSAGE_LEVEL_CRITICAL, INNO_MESSAGE_CODE_CANNOT_READ,
                                  "cannot read from lidar %s", get_name_());
}

bool StageClientRead::create_input_(const void *ctx) {
  if (source_ == SOURCE_FILE) {
    const InputParam *param = reinterpret_cast<const InputParam *>(ctx);
    lidar_->set_play_rate_(param->file_param.play_rate);
    input_ = new FileInput(lidar_, ctx);
  } else if (source_ == SOURCE_TCP) {
    input_ = new TcpInput(lidar_, lidar_comm_, ctx);
  } else if (source_ == SOURCE_UDP) {
    input_ = new UdpInput(lidar_, ctx);
  } else if (source_ == SOURCE_PCAP) {
    const InputParam *param = reinterpret_cast<const InputParam *>(ctx);
    lidar_->set_play_rate_(param->pcap_param.play_rate);
    input_ = new PcapInput(lidar_, ctx);
  } else {
    inno_log_error("invalid type %d", source_);
    return false;
  }
  inno_log_verify(input_, "input_");
  input_->set_deliver_packet_callback(std::bind(&StageClientRead::add_deliver_packet_, this, std::placeholders::_1));
  input_->set_deliver_packet_after_correct_callback(
      std::bind(&StageClientRead::add_deliver_packet_after_correct_, this, std::placeholders::_1));
  input_->set_fatal_message_callback(std::bind(&StageClientRead::send_fatal_message_, this));
  return true;
}

bool StageClientRead::stopping_or_stopped_() {
  free_cached_packet();
  std::unique_lock<std::mutex> lk(mutex_);
  if (state_ == InnoLidarBase::STATE_STOPPING || state_ == InnoLidarBase::STATE_STOPPED) {
    inno_log_info("stop reading because of stop signal");
    input_->stop();
    return true;
  } else {
    return false;
  }
}

void StageClientRead::add_deliver_packet_(InnoCommonHeader *header) {
  lidar_->stats_update_packet_bytes(ResourceStats::PACKET_TYPE_SRC, 1, header->size);
  //  async recorder
  if (lidar_->recorder_callbacks_[INNO_RECORDER_CALLBACK_TYPE_INNO_PC]) {
    char *data = reinterpret_cast<char *>(lidar_->alloc_buffer_(lidar_->kMaxPacketSize));
    memcpy(data, header, header->size);
    lidar_->add_recorder_job_(reinterpret_cast<void *>(data));
  }
  lidar_->add_deliver_job_(header);
}

void StageClientRead::flush_cached_packet() {
  while (!misorder_que_.empty()) {
    auto candidate = misorder_que_.top();
    inno_log_verify(candidate, "candidate");
    misorder_que_.pop();
    add_deliver_packet_(&candidate->common);
  }
}

void StageClientRead::free_cached_packet() {
  while (!misorder_que_.empty()) {
    auto candidate = misorder_que_.top();
    inno_log_verify(candidate, "candidate");
    misorder_que_.pop();
    lidar_->free_buffer_(candidate);
  }
}

int StageClientRead::add_deliver_packet_after_correct_(InnoDataPacket *data_packet) {
  // fix error timestamp
  if (data_packet->common.ts_start_us < 0) {
    int64_t tmp = static_cast<int64_t>(data_packet->common.ts_start_us);
    tmp = tmp & 0x00000000ffffffff;
    // Extract Fractional part
    double decimals = std::fmod(std::abs(data_packet->common.ts_start_us), 1.0);
    data_packet->common.ts_start_us = static_cast<double>(tmp) + decimals;
  }

  auto next_expected = [&](InnoDataPacket *candidate) {
    if (candidate->is_last_sub_frame) {
      expected_idx_ = candidate->idx + 1;
      expected_sub_idx_ = 0;
    } else {
      expected_idx_ = candidate->idx;
      expected_sub_idx_ = candidate->sub_idx + 1;
    }
  };

  // reset misorder queue, if idx jump step big then 10 frame, need to reset
  if (input_->get_first_step() || abs(static_cast<int64_t>(data_packet->idx) - static_cast<int64_t>(expected_idx_)) >
                                      kResetMisorderQueueFrameIdxDiff) {
    inno_log_info("reset misorder queue, idx: %" PRI_SIZEU " | sub-idx: %u", data_packet->idx, data_packet->sub_idx);
    free_cached_packet();
    input_->set_first_step(true);
  }

  // fix misorder InnoDataPacket
  if (input_->get_first_step() ||
      ((data_packet->idx == expected_idx_) && (data_packet->sub_idx == expected_sub_idx_))) {
    input_->set_first_step(false);
    add_deliver_packet_(&data_packet->common);
    next_expected(data_packet);
    while (misorder_que_.size() > 0) {
      auto candidate = misorder_que_.top();
      if ((candidate->idx == expected_idx_) && (candidate->sub_idx == expected_sub_idx_)) {
        misorder_que_.pop();
        add_deliver_packet_(&candidate->common);
        next_expected(candidate);
      } else {
        break;
      }
    }
  } else {
    // inno_log_info("found misorder InnoDataPacket   ----- idx: %" PRI_SIZEU " | sub-idx: %u ------", data_packet->idx,
    //               data_packet->sub_idx);
    if ((data_packet->idx == expected_idx_ && data_packet->sub_idx > expected_sub_idx_) ||
        data_packet->idx > expected_idx_) {
      misorder_que_.push(data_packet);
    } else {
      inno_log_warning("unexpected InnoDataPacket idx: %" PRI_SIZEU " | sub-idx: %u ", data_packet->idx,
                       data_packet->sub_idx);
      lidar_->free_buffer_(data_packet);
    }

    if (misorder_que_.full()) {
      auto candidate = misorder_que_.top();
      inno_log_verify(candidate, "candidate");
      misorder_que_.pop();
      add_deliver_packet_(&candidate->common);
      next_expected(candidate);
      while (misorder_que_.size() > 0) {
        auto candidate = misorder_que_.top();
        if ((candidate->idx == expected_idx_) && (candidate->sub_idx == expected_sub_idx_)) {
          misorder_que_.pop();
          add_deliver_packet_(&candidate->common);
          next_expected(candidate);
        } else {
          break;
        }
      }
    }
  }

  return 0;
}

const char *StageClientRead::get_name_(void) const {
  return lidar_->get_name();
}

////////////////////////////// FileInput //////////////////////////////
int FileInput::read_fd_(char *buf, size_t len) {
  return innovusion::NetManager::recv_full_buffer(fd_, buf, len, -1);
}

int FileInput::keep_reading_() {
  InnoDataPacket *data_packet = NULL;
  size_t data_len_max = kMaxReadSize;
  size_t data_len;
  InnoDataPacket *message_packet = NULL;
  size_t message_len_max = kMaxReadSize;
  size_t message_len;
  InnoStatusPacket *status_packet = NULL;
  size_t status_len_max = kMaxReadSize;
  size_t status_len;

  size_t read_so_far = 0;
  size_t data_cnt = 0;
  size_t message_cnt = 0;
  size_t status_cnt = 0;

  start_time_us_ = lidar_->get_monotonic_raw_time_us();
  first_data_us_ = 0;
  latest_data_us_ = 0;
  total_byte_received_ = 0;
  const char *name = lidar_->get_name();
  int gap = InnoPacketV1Adapt::kMemorryFrontGap;
  int ret = 0;
  while (1) {
    if (!start_flag_) {
      break;
    } else {
      if (data_packet == NULL) {
        data_packet = reinterpret_cast<InnoDataPacket *>(lidar_->alloc_buffer_(data_len_max));
        inno_log_verify(data_packet, "out of memory");
      }
      data_len = data_len_max;
      if (message_packet == NULL) {
        message_packet = reinterpret_cast<InnoDataPacket *>(lidar_->alloc_buffer_(message_len_max));
        inno_log_verify(message_packet, "out of memory");
      }
      message_len = message_len_max;
      if (status_packet == NULL) {
        status_packet = reinterpret_cast<InnoStatusPacket *>(lidar_->alloc_buffer_(status_len_max));
        inno_log_verify(status_packet, "out of memory");
      }
      status_len = status_len_max;
      int major_version = 0;
      bool is_anglehv_table = false;
      int r = read_packet_(data_packet, &data_len, message_packet, &message_len, status_packet, &status_len,
                           major_version, is_anglehv_table);
      if (r > 0) {
        read_so_far += r;
        if (data_len) {
          data_cnt++;
          char *ptr = reinterpret_cast<char *>(data_packet) + InnoPacketV1Adapt::kMemorryFrontGap;
          if ((major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1 &&
               InnoPacketV1Adapt::check_data_packet_v1_and_convert_packet(&ptr, r, gap)) ||
              InnoDataPacketUtils::check_data_packet(*data_packet, r)) {
            latest_data_us_ = data_packet->common.ts_start_us;
            if (!misorder_correct_enable_ || source_ != SOURCE_PCAP) {
              deliver_packet_callback_(&data_packet->common);
            } else {
              deliver_packet_after_correct_callback_(data_packet);
            }
            data_packet = NULL;
          } else {
            inno_log_warning("receive corruped data_packet");
          }
        } else if (message_len) {
          message_cnt++;
          char *ptr = reinterpret_cast<char *>(message_packet) + InnoPacketV1Adapt::kMemorryFrontGap;
          if ((major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1 &&
               InnoPacketV1Adapt::check_data_packet_v1_and_convert_packet(&ptr, r, gap)) ||
              InnoDataPacketUtils::check_data_packet(*message_packet, r)) {
            deliver_packet_callback_(&message_packet->common);
            message_packet = NULL;
          } else {
            inno_log_warning("receive corruped message_packet");
          }
        } else if (status_len) {
          status_cnt++;
          deliver_packet_callback_(&status_packet->common);
          status_packet = NULL;
        } else if (is_anglehv_table) {
          inno_log_info("anglehv_table");
        } else {
          inno_log_verify(false, "no valid packet");
        }
        read_file_rate_control_(latest_data_us_, r);

        inno_log_trace("%s read from %s return %d", name, filename_, r);
      } else if (source_ == SOURCE_FILE || source_ == SOURCE_PCAP) {
        if (r < 0) {
          inno_log_fatal(
              "%s %s data is corrupted, "
              "ret=%d read_so_far=%" PRI_SIZELU "data_cnt=%" PRI_SIZELU " message_cnt=%" PRI_SIZELU
              " status_cnt=%" PRI_SIZELU,
              name, filename_, r, read_so_far, data_cnt, message_cnt, status_cnt);
          ret = -1;
          break;
        } else {
          inno_log_verify(r == 0, "impossible %d", r);
          inno_log_info(
              "%s reach end of %s, "
              "ret=%d read_so_far=%" PRI_SIZELU "data_cnt=%" PRI_SIZELU " message_cnt=%" PRI_SIZELU
              " status_cnt=%" PRI_SIZELU,
              name, filename_, r, read_so_far, data_cnt, message_cnt, status_cnt);
          ret = -2;
          break;
        }
      } else {
        inno_log_verify(source_ == SOURCE_TCP, "invalid source %d", source_);
        inno_log_warning(
            "%s cannot read from connection, "
            "ret=%d read_so_far=%" PRI_SIZELU "data_cnt=%" PRI_SIZELU " message_cnt=%" PRI_SIZELU
            " status_cnt=%" PRI_SIZELU,
            name, r, read_so_far, data_cnt, message_cnt, status_cnt);
        ret = -1;
        break;
      }
    }  // while loop
  }
  if (data_packet != NULL) {
    lidar_->free_buffer_(data_packet);
  }
  if (message_packet != NULL) {
    lidar_->free_buffer_(message_packet);
  }
  if (status_packet != NULL) {
    lidar_->free_buffer_(status_packet);
  }
  return ret;
}

void FileInput::read_file_rate_control_(InnoTimestampUs last_data_us, int r) {
  int64_t should_elapsed = 0;
  if (play_rate_ > 0) {
    total_byte_received_ += r;
    should_elapsed = total_byte_received_ / play_rate_;
  } else if (play_rate_x_ > 0) {
    if (first_data_us_) {
      // inno_log_trace("last %lu %lu", last_data_us, first_data_us_);
      if (last_data_us > first_data_us_ + 1000000UL * 3600 * 24 * 365 * 1) {
        inno_log_info("rate control reset");
        first_data_us_ = last_data_us;
        start_time_us_ = lidar_->get_monotonic_raw_time_us();
      }
      if (last_data_us > first_data_us_) {
        should_elapsed = (last_data_us - first_data_us_) / play_rate_x_;
      } else {
        return;
      }
    } else {
      first_data_us_ = last_data_us;
      return;
    }
  } else {
    return;
  }
  InnoTimestampUs now_us = lidar_->get_monotonic_raw_time_us();
  int64_t elapsed_us = now_us - start_time_us_;
  if (should_elapsed > elapsed_us && elapsed_us >= 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(should_elapsed - elapsed_us));
  }
  return;
}

int FileInput::read_packet_(InnoDataPacket *data_packet, size_t *data_len, InnoDataPacket *message_packet,
                            size_t *message_len, InnoStatusPacket *status_packet, size_t *status_len,
                            int &major_version, bool &anglehv_table) {
  union {
    InnoCommonHeader header;
    char a_[0];
    int d_[0];
  };
  inno_log_verify(data_packet && data_len, "NULL pointer");
  inno_log_verify(4 && message_len, "NULL pointer");
  inno_log_verify(status_packet && status_len, "NULL pointer");
  inno_log_verify(*data_len >= sizeof(InnoDataPacket), "%" PRI_SIZELU " too small", *data_len);
  inno_log_verify(*message_len >= sizeof(InnoDataPacket), "%" PRI_SIZELU " too small", *message_len);
  inno_log_verify(*status_len >= sizeof(InnoStatusPacket), "%" PRI_SIZELU " too small", *status_len);

  int ret = read_fd_(a_, sizeof(InnoCommonHeader));
  if (ret != ssize_t(sizeof(InnoCommonHeader))) {
    inno_log_info("can not read version header, read return %d", ret);
    return 0;
  }
  major_version = header.version.major_version;
  if (header.version.magic_number == kInnoMagicNumberStatusPacket) {
    status_packet->common = header;
    int to_read = sizeof(InnoStatusPacket) - sizeof(InnoCommonHeader);
    if (status_packet->common.version.major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1) {
      to_read = sizeof(InnoStatusPacketV1) - sizeof(InnoCommonHeader);
    }
    ret = read_fd_(reinterpret_cast<char *>(status_packet) + sizeof(InnoCommonHeader), to_read);
    if (ret < to_read) {
      inno_log_warning("can not read version header, read return %d", ret);
      return -1;
    } else if (status_packet->common.version.major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1 &&
               sizeof(InnoStatusPacketV1) == header.size) {
      *data_len = 0;
      *message_len = 0;
      if (InnoPacketReader::verify_packet_crc32(&status_packet->common)) {
        status_packet->common.size = sizeof(InnoStatusPacket);
        status_packet->common.version.major_version = kInnoMajorVersionStatusPacket;
        status_packet->common.version.minor_version = kInnoMinorVersionStatusPacket;
        *status_len = sizeof(InnoStatusPacket);
        InnoPacketReader::set_packet_crc32(&status_packet->common);
        return status_packet->common.size;
      } else {
        inno_log_warning("status packet checksum error");
        return -2;
      }
    } else if (sizeof(InnoStatusPacket) != header.size) {
      inno_log_warning("bad data header size, read return %d vs %u", ret, header.size);
      return -2;
    } else {
      *data_len = 0;
      *message_len = 0;
      *status_len = sizeof(InnoStatusPacket);
      return header.size;
    }
  } else if (header.version.magic_number == kInnoMagicNumberDataPacket) {
    InnoDataPacket dp;
    dp.common = header;
    int to_read = sizeof(InnoDataPacket) - sizeof(InnoCommonHeader);

    if (dp.common.version.major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1) {
      to_read = sizeof(InnoDataPacketV1) - sizeof(InnoCommonHeader);
    }

    ret = read_fd_(reinterpret_cast<char *>(&dp) + sizeof(InnoCommonHeader), to_read);
    if (ret != to_read) {
      inno_log_warning("can not read data header, read return %d %d", ret, to_read);
      return -1;
    }
    if (header.size < sizeof(InnoDataPacketV1)) {
      inno_log_warning("bad data header size %u vs %" PRI_SIZELU, header.size, sizeof(InnoDataPacketV1));
      /* FIXME array subscript is above array bounds [-Werror=array-bounds]
      inno_log_info("%x %x %x %x %x %x",
                    d_[0], d_[1], d_[2], d_[3], d_[4], d_[5]); */
      return -2;
    }
    char *read_pt = NULL;
    *status_len = 0;
    if (CHECK_SPHERE_POINTCLOUD_DATA(dp.type) || CHECK_XYZ_POINTCLOUD_DATA(dp.type)) {
      read_pt = reinterpret_cast<char *>(data_packet + 1);
      *message_len = 0;
      if (*data_len < header.size) {
        inno_log_warning("not enough buffer message: %" PRI_SIZELU " %u", *data_len, header.size);
        return -3;
      }
      *data_len = header.size;
      *data_packet = dp;
    } else if (dp.type == INNO_ITEM_TYPE_MESSAGE || dp.type == INNO_ITEM_TYPE_MESSAGE_LOG) {
      read_pt = reinterpret_cast<char *>(message_packet + 1);
      *data_len = 0;
      if (*message_len < header.size) {
        inno_log_warning("not enough buffer message: %" PRI_SIZELU " %u", *data_len, header.size);
        return -3;
      }
      *message_len = header.size;
      *message_packet = dp;
    } else if (dp.type == INNO_ROBINE_LITE_TYPE_ANGLEHV_TABLE || dp.type == INNO_ROBINW_ITEM_TYPE_ANGLEHV_TABLE) {
      *lidar_->anglehv_table_ = dp;
      to_read = header.size - sizeof(InnoDataPacket);
      *data_len = 0;
      *message_len = 0;
      *status_len = 0;
      ret = read_fd_(lidar_->anglehv_table_->payload, to_read);
      if (ret != to_read) {
        inno_log_warning("can not read data, read return %d %d", ret, to_read);
        return -1;
      }
      lidar_->anglehv_init_ = true;
      anglehv_table = true;
      return to_read + sizeof(InnoCommonHeader);
    } else {
      inno_log_warning("invalid data type %d", dp.type);
      return -2;
    }
    to_read = header.size - sizeof(InnoDataPacket);
    if (dp.common.version.major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1) {
      char *ptr = read_pt - sizeof(InnoDataPacket);
      memcpy(ptr + InnoPacketV1Adapt::kMemorryFrontGap, &dp, sizeof(InnoDataPacketV1));
      to_read = header.size - sizeof(InnoDataPacketV1);
    }
    ret = read_fd_(read_pt, to_read);
    if (ret < to_read) {
      inno_log_warning("cannot read data, read return %d/%d/%u", ret, to_read, header.size);
      return -1;
    } else {
      return header.size;
    }
  } else {
    inno_log_warning("invalid magic number 0x%hx %u", header.version.magic_number, header.size);
    return -2;
  }
}

int FileInput::read_data() {
  int32_t max_file_rewind = input_param_.file_param.rewind;
  if (max_file_rewind >= 0) {
    if (play_round_ >= max_file_rewind + 1) {
      return -3;
    }
  }
  fd_ = InnoUtils::open_file(filename_, O_RDONLY, 0);
  if (fd_ >= 0) {
    reach_file_end_ = false;
    int ret = keep_reading_();
    close(fd_);
    if (ret == -2) {
      // end of file is not really an error
      reach_file_end_ = true;
      ret = 0;
    }
    play_round_++;
    set_first_step(true);
    inno_log_info("%s rewind file %s %d/%d", lidar_->get_name(), filename_, play_round_, max_file_rewind);
    return ret;
  } else {
    cannot_open_file_ = true;
    return -1;
  }
}

void FileInput::set_file_play_rate() {
  int tmp_rate = 0;
  if (input_param_.base_param.source_type == SOURCE_FILE) {
    tmp_rate = input_param_.file_param.play_rate;
  } else if (input_param_.base_param.source_type == SOURCE_PCAP) {
    tmp_rate = input_param_.pcap_param.play_rate;
  }
  inno_log_info("Received play rate input parameter: %d", tmp_rate);
  if (tmp_rate > 100) {
    play_rate_ = 0;
    play_rate_x_ = tmp_rate / 10000.0;
    inno_log_info("Setting play rate to %fX", play_rate_x_);
  } else if (tmp_rate >= 0) {
    play_rate_ = tmp_rate;
    play_rate_x_ = 0.0;
    inno_log_info("Setting play rate to %d MB/s", play_rate_);
  } else {
    inno_log_verify(false, "Invalid play rate entered %d", tmp_rate);
  }
}

////////////////////////////// UdpInput //////////////////////////////
UdpInput::~UdpInput() {
  if (inno_data_diff3_error_counter_) {
    inno_log_warning(
        "data packet sub_seq interval over 3 counter:%" PRI_SIZEU " delta:%" PRI_SIZEU " repeat counter:%" PRI_SIZEU,
        inno_data_diff3_error_counter_, inno_data_diff3_error_counter_ - last_print_inno_data_diff3_error_counter_,
        inno_data_repeat_error_counter_);
  }
  if (inno_status_diff3_error_counter_) {
    inno_log_warning("status packet idx interval over 3 counter:%" PRI_SIZEU " delta:%" PRI_SIZEU
                     " repeat counter:%" PRI_SIZEU,
                     inno_status_diff3_error_counter_,
                     inno_status_diff3_error_counter_ - last_print_inno_status_diff3_error_counter_,
                     inno_status_repeat_error_counter_);
  }
}

int UdpInput::bind_udp_port_(uint16_t port) {
  std::vector<InnoUdpOpt> opts;
  if (use_mreq_) {
    InnoUdpOpt opt = {IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<const void *>(&mreq_),
                      static_cast<socklen_t>(sizeof(mreq_)), "IP_ADD_MEMBERSHIP"};
    opts.emplace_back(opt);
  }
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 500000;
  InnoUdpOpt opt = {SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const void *>(&tv), static_cast<socklen_t>(sizeof(tv)),
                    "SO_RCVTIMEO"};
  opts.emplace_back(opt);
  int n = 1024 * 1024;
  opt = {SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const void *>(&n), static_cast<socklen_t>(sizeof(n)), "SO_RCVBUF"};
  opts.emplace_back(opt);
  return InnoUdpHelper::bind(port, opts);
}

void UdpInput::wait_until_stopping_() {
  while (start_flag_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    continue;
  }
}

int UdpInput::read_udp_(int32_t port) {
  int fd = bind_udp_port_(port);
  uint32_t timeout_flag = 1;
  uint32_t eagain_count = 1;
  if (fd < 0) {
    inno_log_error("bind_udp_port_ failed, port is %d", port);
    return -1;
  }

  inno_log_info("recvfrom UDP %d", port);

  void *buff = NULL;
  uint32_t specified_ip = lidar_->get_specified_ip();
  int gap = 0;
  while (1) {
    if (!start_flag_) {
      inno_log_info("stop reading because of stop signal");
      break;
    }

    if (buff == NULL) {
      buff = lidar_->alloc_buffer_(kMaxReadSize);
      inno_log_verify(buff, "out of memory");
    }

    struct sockaddr_in cliaddr;
    socklen_t len = sizeof(cliaddr);
    int n;

#if !(defined(__MINGW64__) || defined(_WIN32))
    while (-1 == (n = recvfrom(fd, reinterpret_cast<char *>(buff) + gap, kMaxReadSize - gap, MSG_WAITALL,
                               (struct sockaddr *)&cliaddr, &len)) &&
           errno == EINTR) {
    }
#else
    while (-1 == (n = recvfrom(fd, reinterpret_cast<char *>(buff) + gap, kMaxReadSize - gap, 0,
                               (struct sockaddr *)&cliaddr, &len)) &&
           errno == EINTR) {
    }
#endif
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (eagain_count == timeout_flag) {
          inno_log_info("%s", errno == EAGAIN ?  // EAGAIN means timeout
                                "EAGAIN"
                                            : "EWOULDBLOCK");
          timeout_flag *= 2;
        }
        eagain_count++;
        continue;
      } else {
        inno_log_error_errno("recv port=%d, n=%d, fd=%d", port, n, fd);
      }
    } else {
      if (timeout_flag >= 2) {
        inno_log_info("recover from network timeout");
        timeout_flag = 1;
        eagain_count = 1;
        if (misorder_correct_enable_) {
          set_first_step(true);
        }
      }
      bool source_ip_valid = lidar_->source_ip_check_(specified_ip, cliaddr.sin_addr.s_addr);
      if (!source_ip_valid) {
        continue;
      }
      if (n >= ssize_t(sizeof(InnoCommonHeader))) {
        union {
          const InnoDataPacket *data_hd;
          const InnoStatusPacket *status_hd;
          InnoCommonHeader *hd;
        };
        hd = reinterpret_cast<InnoCommonHeader *>(reinterpret_cast<char *>(buff) + gap);
        bool is_same_data_sub_seq = false;
        bool is_same_status_idx = false;
        if ((hd->version.major_version == InnoPacketV1Adapt::kInnoProtocolMajorV1 &&
             ((hd->version.magic_number == kInnoMagicNumberDataPacket &&
               verify_inno_data_packet_counter(*data_hd, &is_same_data_sub_seq) &&
               (InnoPacketV1Adapt::check_data_packet_v1_and_convert_packet(reinterpret_cast<char **>(&hd), n, gap))) ||
              (hd->version.magic_number == kInnoMagicNumberStatusPacket &&
               InnoPacketV1Adapt::check_status_packet_v1_and_convert_packet(reinterpret_cast<char **>(&hd), n,
                                                                            gap)))) ||
            (hd->version.major_version > InnoPacketV1Adapt::kInnoProtocolMajorV1 &&
             ((hd->version.magic_number == kInnoMagicNumberDataPacket &&
               InnoDataPacketUtils::check_data_packet(*data_hd, n) &&
               verify_inno_data_packet_counter(*data_hd, &is_same_data_sub_seq)) ||
              (hd->version.magic_number == kInnoMagicNumberStatusPacket &&
               InnoDataPacketUtils::check_status_packet(*status_hd, n) &&
               verify_inno_status_packet_counter(*status_hd, &is_same_status_idx))))) {
          if (misorder_correct_enable_ &&
              ((hd->version.magic_number == kInnoMagicNumberDataPacket) &&
               (CHECK_XYZ_POINTCLOUD_DATA(data_hd->type) || CHECK_SPHERE_POINTCLOUD_DATA(data_hd->type)))) {
            deliver_packet_after_correct_callback_(reinterpret_cast<InnoDataPacket *>(hd));
          } else {
            deliver_packet_callback_(hd);
          }
          buff = NULL;
        } else {
          if (is_same_data_sub_seq) {
            inno_log_warning("same data packet - idx:%" PRI_SIZEU " sub_seq: %u", data_hd->idx, data_hd->sub_seq);
          } else if (is_same_status_idx) {
            inno_log_warning("same status packet - idx:%" PRI_SIZEU, status_hd->idx);
          } else {
            inno_log_warning("bad packet magic=0x%x size=%d", hd->version.magic_number, n);
          }
        }
      } else {
        inno_log_warning("size too small %d", n);
      }
    }
  }  // while (1)

  if (buff) {
    lidar_->free_buffer_(buff);
    buff = NULL;
  }
  InnoUtils::close_fd(fd);

  return 0;
}

int UdpInput::read_data() {
  static const size_t kPortsCount = 3;
  static constexpr int32_t kGetUdpPortIntervalMsArray[5] = {500, 500, 1000, 1500, 2000};
  int32_t ports[kPortsCount];
  char ip[64] = {0};
  int ret;
  char my_ip[64] = {0};
  for (int round = 0; round < 2; round++) {
    int get_interval_count = 0;
    while (true) {
      ret =
          lidar_->comm_->get_server_udp_ports_ip(&ports[0], &ports[1], &ports[2], ip, sizeof(ip), my_ip, sizeof(my_ip));
      if (ret != 0) {
        inno_log_error("cannot get server udp ports %d", ret);
        if (get_interval_count == 5) {
          return 1;
        }
        if (!start_flag_) {
          return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kGetUdpPortIntervalMsArray[get_interval_count]));
        get_interval_count += 1;
      } else {
        inno_log_info("read udps: data:%d message:%d status:%d ip=%s my_ip=%s", ports[0], ports[1], ports[2], ip,
                      my_ip);
        break;
      }
    }

    if ((ip[0] == 0 || ip[0] == '0') && udp_port_ == 0) {
      inno_log_error("will not set remote udp port/ip because udp_port is 0 and server udp is off");
      send_fatal_message_callback_();
      return 1;
    }

    if (round == 0) {
      // only set_server_udp_ports_ip in the first round
      if (udp_port_ != 0) {
        inno_log_info("set_server_udp_ports_ip(%hu)", udp_port_);
        lidar_->comm_->set_server_udp_ports_ip(udp_port_);
      }
    } else {
      // server udp is still off
      if (ip[0] == 0 || ip[0] == '0') {
        inno_log_error("cannot set remote udp port/ip");
        send_fatal_message_callback_();
        return 1;
      }
      if (my_ip[0] != 0) {
        mreq_.imr_multiaddr.s_addr = inet_addr(ip);
        mreq_.imr_interface.s_addr = inet_addr(my_ip);
        use_mreq_ = true;
        inno_log_verify(mreq_.imr_multiaddr.s_addr != INADDR_NONE, "bad ip m-addr %s", ip);
        inno_log_verify(mreq_.imr_interface.s_addr != INADDR_NONE, "bad ip i-addr %s", my_ip);
        inno_log_info("use multicast addr %s on interface %s", ip, my_ip);
      }
    }
  }

  // remove duplicated
  for (size_t i = 0; i < kPortsCount; i++) {
    for (size_t j = 0; j < i; j++) {
      if (ports[i] == ports[j]) {
        ports[i] = 0;
      }
    }
  }

  std::vector<std::thread *> threads;

  for (size_t i = 0; i < kPortsCount; i++) {
    if (ports[i]) {
      threads.push_back(new std::thread([this, ports, i]() { read_udp_(ports[i]); }));
    }
  }

  wait_until_stopping_();

  for (auto &th : threads) {
    th->join();
    delete th;
  }
  return 0;
}

bool UdpInput::verify_inno_data_packet_counter(const InnoDataPacket &pkt, bool* same_data_sub_seq) {
  // massgae, return
  if (pkt.type == INNO_ITEM_TYPE_MESSAGE || pkt.type == INNO_ITEM_TYPE_MESSAGE_LOG ||
      pkt.type == INNO_ROBINW_ITEM_TYPE_ANGLEHV_TABLE ||
      pkt.type == INNO_FALCON_RING_ID_TABLE) {
    return true;
  }
  // check first five data packets sub_seq if is always 0, if sub_seq always 0, means old firmware,
  // need not do E2E check
  if (inno_data_first_step_ < 5) {
    inno_data_first_step_++;
    inno_data_last_sub_seq_ = pkt.sub_seq;
    if (pkt.sub_seq == 0) {
      sub_seq_zero_counter_++;
    }
    return true;
  }

  if (sub_seq_zero_counter_ >= 5) {
    return true;
  }
  // same data packet, return
  if (pkt.sub_seq == inno_data_last_sub_seq_) {
    *same_data_sub_seq = true;
    inno_data_repeat_error_counter_++;
    return false;
  }

  if (verify_counter_process(pkt.sub_seq, inno_data_last_sub_seq_)) {
    inno_data_diff3_error_counter_++;
    if (InnoUtils::get_time_ms() - last_print_data_diff_time_ > kPrintPacketDiffIntervalMs) {
      inno_log_warning(
          "data packet sub_seq interval over 3 counter:%" PRI_SIZEU " delta:%" PRI_SIZEU " repeat counter:%" PRI_SIZEU,
          inno_data_diff3_error_counter_, inno_data_diff3_error_counter_ - last_print_inno_data_diff3_error_counter_,
          inno_data_repeat_error_counter_);

      last_print_inno_data_diff3_error_counter_ = inno_data_diff3_error_counter_;
      last_print_data_diff_time_ = InnoUtils::get_time_ms();
    }
  }

  inno_data_last_sub_seq_ = pkt.sub_seq;
  return true;
}

bool UdpInput::verify_inno_status_packet_counter(const InnoStatusPacket &pkt, bool* same_status_sub_seq) {
  // only use 16 bit.
  uint16_t current_idx = pkt.idx;

  // don't check the first packet
  if (inno_status_first_step_ == true) {
    inno_status_first_step_ = false;
    inno_status_last_idx_ = current_idx;
    return true;
  }

  // same data packet, return
  if (current_idx == inno_status_last_idx_) {
    *same_status_sub_seq = true;
    inno_status_repeat_error_counter_++;
    return false;
  }

  if (verify_counter_process(current_idx, inno_status_last_idx_)) {
    inno_status_diff3_error_counter_++;
    if (InnoUtils::get_time_ms() - last_print_status_diff_time_ > kPrintPacketDiffIntervalMs) {
      inno_log_warning("status packet idx interval over 3 counter:%" PRI_SIZEU " delta:%" PRI_SIZEU
                       " repeat counter:%" PRI_SIZEU,
                       inno_status_diff3_error_counter_,
                       inno_status_diff3_error_counter_ - last_print_inno_status_diff3_error_counter_,
                       inno_status_repeat_error_counter_);
      last_print_inno_status_diff3_error_counter_ = inno_status_diff3_error_counter_;
      last_print_status_diff_time_ = InnoUtils::get_time_ms();
    }
  }

  inno_status_last_idx_ = current_idx;
  return true;
}

bool UdpInput::verify_counter_process(int current, int last) {
  // edge value for min
  // -65532 = 0 - 65535 + 3
  int edge_min_value = std::abs(max_no_new_repeate_data_ - max_value_ + max_delta_counter_init_);

  // special calse : 65533 65534 65535 0 1 2
  // special_number_max_start = 65535 - 3 + 1 = 65533
  int special_number_max_start = max_value_ - max_delta_counter_init_ + 1;
  // special_number_max_end = 65535
  int special_number_max_end = max_value_;

  // special_number_min_start = 0
  int special_number_min_start = 0;
  // special_number_min_end = 3 - 1 = 2
  int special_number_min_end = max_delta_counter_init_ - 1;

  bool special_case =
    ((((last <= special_number_max_end) && (last >= special_number_max_start)) ||
    ((last <= special_number_min_end) && (last >= special_number_min_start)))
    &&
    (std::abs(current - last) > edge_min_value));

  bool basic_case = (std::abs(current - last) <= max_delta_counter_init_);
  return !(special_case || basic_case);
}

////////////////////////////// TcpInput //////////////////////////////
int TcpInput::read_fd_(char *buf, size_t len) {
  return innovusion::NetManager::recv_full_buffer(fd_, buf, len, 0);
}

int TcpInput::read_data() {
  fd_ = lidar_comm_->get_connection(0.5, 256 * 1024);
  if (fd_ >= 0) {
    int zero = 0;
    int ret = lidar_comm_->send_command_with_fd(fd_, NULL, &zero, "GET /start/ HTTP/1.0\r\n\r\n");
    if (ret == fd_) {
      ret = keep_reading_();
      InnoUtils::close_fd(fd_);
      return ret;
    } else {
      inno_log_error("cannot send start %d", ret);
      send_fatal_message_callback_();
      return -2;
    }
  } else {
    inno_log_error("cannot get tcp connection to server");
    send_fatal_message_callback_();
    return -1;
  }
}

}  // namespace innovusion
