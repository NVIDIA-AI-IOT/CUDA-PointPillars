/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#ifndef CONVERTER_PNG_RECORDER_H_
#define CONVERTER_PNG_RECORDER_H_

#include <functional>
#include <string>
#include <vector>

#include "sdk_common/inno_lidar_packet.h"

namespace innovusion {
class RangeImage;
class PNGWriter;

class PngRecorder {
  class point_to_image_interface {
   public:
    virtual ~point_to_image_interface() = default;
    virtual void point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_, uint32_t block_so_far_,
                                uint32_t packet_so_far_) = 0;
  };

  class falcon_point_to_image : public point_to_image_interface {
   public:
    void point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_, uint32_t block_so_far_,
                        uint32_t packet_so_far_) override;
  };

  class robin_point_to_image : public point_to_image_interface {
   public:
    void point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_, uint32_t block_so_far_,
                        uint32_t packet_so_far_) override;
  };

  class falcon_2_nt3_point_to_image : public point_to_image_interface {
   public:
    void point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_, uint32_t block_so_far_,
                        uint32_t packet_so_far_) override;
  };

// State Machine
enum class State {
  Invalid,
  Capturing,
  Saving,
  Finish
};

 public:
  explicit PngRecorder(size_t job_duration, size_t max_pack_num,
                       size_t max_block_num, bool ignore_first_frame);
  ~PngRecorder();

  bool capture(const InnoDataPacket *pkt);

  void save(std::vector<char>* buf);
  void save(const std::string& filename);

 private:
  void start_capture_();
  void switch_state_(const InnoDataPacket *pkt);
  void capture_packet_(const InnoDataPacket *pkt);

  void save_point_to_image_(RangeImage* range_image);
  using SaveFunc = std::function<void(const PNGWriter&)>;
  void save_png_(SaveFunc save_func);
  point_to_image_interface *create_point_to_image_instance_(const InnoDataPacket *pkt);

 private:
  State state_ = State::Invalid;

  size_t job_duration_;

  size_t max_pack_num_;
  size_t max_block_num_;

  InnoDataPacket *saved_packet_;
  InnoDataPacket *saved_packet_headers_;

  uint32_t block_so_far_;
  uint32_t packet_so_far_;

  int64_t frame_idx_last_;
  uint64_t frame_captured_;

  uint32_t current_point_in_frame_;
  point_to_image_interface *point_to_image_ = nullptr;
};

}  // namespace innovusion

#endif  // CONVERTER_PNG_RECORDER_H_
