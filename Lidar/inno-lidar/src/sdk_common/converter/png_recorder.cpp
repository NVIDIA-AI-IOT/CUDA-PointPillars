/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "sdk_common/converter/png_recorder.h"

#include "sdk_common/converter/png.h"
#include "sdk_common/converter/png_range_color.h"
#include "sdk_common/converter/png_range_image.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/inno_lidar_log.h"

namespace innovusion {

constexpr int INVALID_FRAME_IDX = -1;

//
//
//
PngRecorder::PngRecorder(size_t job_duration, size_t max_pack_num,
                         size_t max_block_num, bool ignore_first_frame) {
  job_duration_ = job_duration;

  max_pack_num_ = max_pack_num;
  max_block_num_ = max_block_num;

  block_so_far_ = 0;
  packet_so_far_ = 0;

  //
  frame_idx_last_ = INVALID_FRAME_IDX;

  frame_captured_ = 0;
  current_point_in_frame_ = 0;

  //
  size_t len_saved_ =
      sizeof(InnoDataPacket) + sizeof(InnoEnBlock2) * max_block_num;
  saved_packet_ =
      reinterpret_cast<InnoDataPacket *>(calloc(len_saved_, 1));
  inno_log_verify(saved_packet_,
                  "cannot allocate saved_ %" PRI_SIZELU,
                  len_saved_);
  memset(saved_packet_, 0, len_saved_);

  size_t header_len_saved = sizeof(InnoDataPacket) * max_pack_num;
  saved_packet_headers_ =
      reinterpret_cast<InnoDataPacket *>(calloc(header_len_saved, 1));
  inno_log_verify(saved_packet_headers_,
                  "cannot allocate saved_header %" PRI_SIZELU,
                  header_len_saved);
  memset(saved_packet_headers_, 0, header_len_saved);
  if (!ignore_first_frame) {
    start_capture_();
  }
}

//
//
//
PngRecorder::~PngRecorder() {
  inno_log_info("-------- ~PngRecorder --------");

  if (saved_packet_) {
    free(saved_packet_);
  }

  if (saved_packet_headers_) {
    free(saved_packet_headers_);
  }

  if (point_to_image_) {
    delete point_to_image_;
  }
}

void PngRecorder::start_capture_() {
  frame_captured_ = 0;
  current_point_in_frame_ = 0;

  // switch
  this->state_ = State::Capturing;
}

//
//
//
void PngRecorder::switch_state_(const InnoDataPacket* pkt) {
  bool is_first_subframe =
      frame_idx_last_ >= 0 &&
      frame_idx_last_ != ssize_t(pkt->idx);

  switch (this->state_) {
    case State::Invalid:
      if (is_first_subframe) {
        inno_log_info("capture first frame %" PRI_SIZEU,
                      pkt->idx);
        start_capture_();
      }
      break;

    case State::Capturing:
      if (is_first_subframe) {
        inno_log_info("capture %u points in frame-%" PRI_SIZEU
                      "-%" PRI_SIZED " to png",
                      current_point_in_frame_, frame_captured_,
                      frame_idx_last_);

        frame_captured_++;
        current_point_in_frame_ = 0;

        if (job_duration_ > 0 &&
            frame_captured_ >= job_duration_) {
          this->state_ = State::Saving;
        }
      }
      break;

    default:
      break;
  }

  //
  frame_idx_last_ = pkt->idx;
}

//
//
//
void PngRecorder::capture_packet_(const InnoDataPacket *pkt) {
  inno_log_trace("capture frame %" PRI_SIZEU, pkt->idx);
  if (packet_so_far_ >= max_pack_num_) {
    inno_log_warning("reach limit kMaxPacketNum %u", packet_so_far_);
    return;
  }

  uint32_t block_left = max_block_num_ - block_so_far_;
  uint32_t block_to_copy = pkt->item_number;
  if (block_to_copy > block_left) {
    block_to_copy = block_left;
    inno_log_warning("reach limit kMaxBlockNum %" PRI_SIZELU, max_block_num_);
  }

  current_point_in_frame_ += InnoDataPacketUtils::get_points_count(*pkt);
  for (uint32_t i = 0; i < block_to_copy; i++) {
    if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      if (pkt->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        memcpy(reinterpret_cast<InnoBlock2 *>(saved_packet_->payload) + block_so_far_,
               reinterpret_cast<const InnoBlock1 *>(pkt->payload) + i, sizeof(InnoBlock1));
      } else {
        memcpy(reinterpret_cast<InnoBlock2 *>(saved_packet_->payload) + block_so_far_,
               reinterpret_cast<const InnoBlock2 *>(pkt->payload) + i, sizeof(InnoBlock2));
      }
    } else {
      if (pkt->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        memcpy(reinterpret_cast<InnoEnBlock1 *>(saved_packet_->payload) + block_so_far_ ,
               reinterpret_cast<const InnoEnBlock1 *>(pkt->payload) + i, sizeof(InnoEnBlock1));
      } else {
        memcpy(reinterpret_cast<InnoEnBlock *>(saved_packet_->payload) + block_so_far_ ,
               reinterpret_cast<const InnoEnBlock2 *>(pkt->payload) + i, sizeof(InnoEnBlock2));
      }
    }
    block_so_far_++;
  }
  saved_packet_->multi_return_mode = pkt->multi_return_mode;
  saved_packet_->type = pkt->type;

  if (block_to_copy > 0) {
    memcpy(saved_packet_headers_ + packet_so_far_, pkt, sizeof(InnoDataPacket));
    packet_so_far_++;
  }
  return;
}

PngRecorder::point_to_image_interface *PngRecorder::create_point_to_image_instance_(const InnoDataPacket *pkt) {
  if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
    return new falcon_point_to_image();
  } else if (pkt->type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD || pkt->type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD) {
    return new robin_point_to_image();
  } else if (pkt->type == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD) {
    return new falcon_2_nt3_point_to_image();
  }
  return nullptr;
}
//
//
//
bool PngRecorder::capture(const InnoDataPacket *pkt) {
  inno_log_verify(pkt, "pkt");
  switch_state_(pkt);

  if (this->state_ == State::Capturing) {
    capture_packet_(pkt);
  }

  if (!point_to_image_) {
    point_to_image_ = create_point_to_image_instance_(pkt);
  }
  return this->state_ == State::Saving;
}

//
//
//
void PngRecorder::falcon_point_to_image::point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_,
                                                        uint32_t block_so_far_, uint32_t packet_so_far_) {
  size_t current_packet = 0;
  InnoBlock2 *block;
  for (size_t i = 0; i < block_so_far_; i++) {
    inno_log_verify(current_packet < packet_so_far_, "%" PRI_SIZELU " vs %u", current_packet, packet_so_far_);
    InnoBlock2 *inno_block2s = reinterpret_cast<InnoBlock2 *>(saved_packet_->payload);
    block = &inno_block2s[i];

    InnoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header);

    for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {
      for (uint32_t m = 0; m < 2; m++) {
        InnoChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];
        if (pt.radius > 0) {
          InnoXyzrD xyzr;
          InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[ch], pt.radius, ch, &xyzr);

          range_image->insert_point(xyzr.x, xyzr.y, xyzr.z, pt.refl);
        }
      }
    }
  }
}

void PngRecorder::robin_point_to_image::point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_,
                                                       uint32_t block_so_far_, uint32_t packet_so_far_) {
  InnoEnBlock *block;
  size_t current_packet = 0;
  for (size_t i = 0; i < block_so_far_; i++) {
    inno_log_verify(current_packet < packet_so_far_, "%" PRI_SIZELU " vs %u", current_packet, packet_so_far_);
    if (saved_packet_->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
      InnoEnBlock1 *inno_block1s = reinterpret_cast<InnoEnBlock1 *>(saved_packet_->payload) + i;
      block = reinterpret_cast<InnoEnBlock *>(&inno_block1s[i]);
    } else {
      InnoEnBlock *inno_block2s = reinterpret_cast<InnoEnBlock *>(saved_packet_->payload) + i;
      block = &inno_block2s[i];
    }

    InnoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header,
                                               static_cast<InnoItemType>(saved_packet_->type));

    DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(saved_packet_->type);

    for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {
      for (uint32_t m = 0; m < 2; m++) {
        InnoEnChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];
        if (pt.radius > 0) {
          // for robin
          // uint32_t scan_id = block->header.scan_id * 4 + ch;
          int index = block->header.scan_id * 4 + ch;
          uint32_t scan_id = channel_mapping[index] + block->header.facet * tdc_channel_number;
          InnoXyzrD xyzr;
          InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[ch], pt.radius, scan_id, &xyzr,
                                              static_cast<InnoItemType>(saved_packet_->type));
          range_image->insert_point(xyzr.x, xyzr.y, xyzr.z,
                                    saved_packet_->use_reflectance ? pt.reflectance : pt.intensity);
        }
      }
    }
  }
}

void PngRecorder::falcon_2_nt3_point_to_image::point_to_image(RangeImage *range_image, InnoDataPacket *saved_packet_,
                                                       uint32_t block_so_far_, uint32_t packet_so_far_) {
  InnoEnBlock2 *block;
  size_t current_packet = 0;
  for (size_t i = 0; i < block_so_far_; i++) {
    inno_log_verify(current_packet < packet_so_far_, "%" PRI_SIZELU " vs %u", current_packet, packet_so_far_);
    InnoEnBlock2 *inno_block2s = reinterpret_cast<InnoEnBlock2 *>(saved_packet_->payload);
    block = &inno_block2s[i];

    InnoBlockFullAngles full_angles;
    InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header,
                                               static_cast<InnoItemType>(saved_packet_->type));

    for (uint32_t ch = 0; ch < kInnoChannelNumber; ch++) {
      for (uint32_t m = 0; m < 2; m++) {
        InnoEnChannelPoint &pt = block->points[innoblock_get_idx(ch, m)];
        if (pt.radius > 0) {
          InnoXyzrD xyzr;
          InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[ch], pt.radius, ch, &xyzr,
                                              static_cast<InnoItemType>(saved_packet_->type));
          range_image->insert_point(xyzr.x, xyzr.y, xyzr.z,
                                    saved_packet_->use_reflectance ? pt.reflectance : pt.intensity);
        }
      }
    }
  }
}

void PngRecorder::save_point_to_image_(RangeImage* range_image) {
  inno_log_verify(range_image != NULL && point_to_image_ != NULL, "range_image");
  point_to_image_->point_to_image(range_image, saved_packet_, block_so_far_, packet_so_far_);
}

//
//
//
void PngRecorder::save_png_(SaveFunc save_func) {
  inno_log_info("save_to_png, block_so_far_ : %u", block_so_far_);

  //
  float angularResolution_x = static_cast<float>(0.05f * (M_PI / 180.0f));
  float angularResolution_y = static_cast<float>(0.05f * (M_PI / 180.0f));

  float maxAngleWidth = static_cast<float>(120.0f * (M_PI / 180.0f));
  float maxAngleHeight = static_cast<float>(60.0f * (M_PI / 180.0f));

  RangeImage range_image(angularResolution_x, angularResolution_y,
                         maxAngleWidth, maxAngleHeight);

  range_image.start();
  save_point_to_image_(&range_image);
  range_image.stop();

  // save
  RangeColor colorScale(ColorType::BGYR);
  PNGWriter pngWriter(range_image.width, range_image.height,
                      PNGColorType::RGB, 8);

  range_image.get_image(
      [&colorScale](float range, float ref, unsigned char* r,
                    unsigned char* g, unsigned char* b) {
        colorScale.getColor(range, ref, r, g, b);
      },
      [&pngWriter](int x, int y, unsigned char r, unsigned char g,
                    unsigned char b) { pngWriter.encode(x, y, r, g, b); });

  save_func(pngWriter);

  // switch state
  this->state_ = State::Finish;
}

//
//
//
void PngRecorder::save(std::vector<char>* buf) {
  inno_log_verify(buf, "png buf");

  save_png_([buf](const PNGWriter& pngWriter){
    inno_log_info("saveto_buffer");
    pngWriter.saveto_buffer(buf);
  });
}

//
//
//
void PngRecorder::save(const std::string& filename) {
  inno_log_verify(filename.empty() == false, "png filename is empty.");

  std::string filename_ = filename;

  if (!filename_.empty()) {
    if (filename_.rfind(".") == std::string::npos) {
      filename_ += ".png";
    }
  }

  save_png_([&filename_](const PNGWriter& pngWriter){
    inno_log_info("saveto_file");
    pngWriter.saveto_file(filename_);
  });
}

}  // namespace innovusion
