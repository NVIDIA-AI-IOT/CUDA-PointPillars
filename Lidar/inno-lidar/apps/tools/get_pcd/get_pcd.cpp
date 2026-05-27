/**
 *  Copyright (C) 2023 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#if defined(__MINGW64__) || !defined(_WIN32)
#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#else
#include "src/utils/getopt_windows.h"
#define strcasecmp _stricmp
#endif
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "src/sdk_common/converter/cframe_converter.h"
#include "src/sdk_common/converter/png_recorder.h"
#include "src/sdk_common/converter/rosbag_recorder.h"
#include "src/sdk_common/inno_lidar_api.h"
#include "src/sdk_common/inno_lidar_other_api.h"
#include "src/sdk_common/inno_lidar_packet_utils.h"
#include "src/sdk_common/ring_id_converter_interface.h"
#include "src/utils/inno_lidar_log.h"
#include "src/utils/utils.h"

static const double kUsInSecond = 1000000.0;
static const double k10UsInSecond = 100000.0;
static const uint32_t kMaxMsgBuf = 1500;
static int recorder_point_type = 1;  // 0: lite-pcd,1: normal-pcd 2:en-pcd
static bool is_compact_packet = false;
static bool filename_with_timestamp = false;
static int  g_lidar_handle = -1;
static int force_vehicle_coordinate = 0;
/***********************
 * class RingBuffer
 ***********************/
class RingBuffer {
 public:
  struct block_t {
    char *buf = NULL;
    int8_t status = kBufFree;
    int8_t index = -1;
    uint32_t written = 0;
  };

  static const uint32_t kBlockSize = 2 * 1024 * 1024;
  static const int8_t kBufFree = 0;
  static const int8_t kBufUsing = 1;
  static const int8_t kBufUsed = 2;

  explicit RingBuffer(uint32_t num = 10) {
    blocks_.resize(num);
    block_num_ = num;
    for (size_t i = 0; i < blocks_.size(); i++) {
      blocks_[i].buf = new char[kBlockSize];
      blocks_[i].index = i;
      inno_log_verify(blocks_[i].buf, "bufs_[i] - %" PRI_SIZELD, i);
    }
    blocks_[0].status = kBufUsing;
  }

  ~RingBuffer() {
    for (size_t i = 0; i < blocks_.size(); i++) {
      inno_log_verify(blocks_[i].status == kBufFree || (blocks_[i].status == kBufUsing && blocks_[i].written == 0),
                      "buf %" PRI_SIZELD " is not free", i);
      delete blocks_[i].buf;
      blocks_[i].buf = NULL;
    }
  }

  bool append(const char *buf, uint32_t length, bool retry = false) {
    inno_log_verify(length <= kBlockSize, "length %u is too large", length);
    if (!retry && blocks_[index_].written + length <= kBlockSize) {
      if (blocks_[index_].status != kBufUsing) {
        inno_log_panic("block %d is in wrong status", index_);
        return false;
      }
    } else {
      bool r = next_free(retry);
      if (r) {
        blocks_[index_].status = kBufUsing;
      } else {
        // get free block failed, status should be kBufUsed
        if (blocks_[index_].status != kBufUsed) {
          inno_log_panic("block %d is in wrong status %d", index_, blocks_[index_].status);
        }
        return false;
      }
    }
    memcpy(blocks_[index_].buf + blocks_[index_].written, buf, length);
    blocks_[index_].written += length;
    return true;
  }


  block_t cur_used() {
    return blocks_[index_used_];
  }

  block_t cur_using() {
    return blocks_[index_];
  }

  void release(uint8_t idx) {
    blocks_[idx].written = 0;
    blocks_[idx].status = kBufFree;
  }

  // get next used buffer
  int8_t next_used() {
    if (index_used_ >= 0 && blocks_[index_used_].status == kBufUsed) {
      return index_used_;
    }

    if (index_used_ == block_num_ - 1) {
      if (blocks_[0].status == kBufUsed) {
        index_used_ = 0;
        return index_used_;
      } else {
        return -1;
      }
    } else {
      if (blocks_[index_used_ + 1].status == kBufUsed) {
        index_used_ += 1;
        return index_used_;
      } else {
        return -1;
      }
    }
  }

 private:
  RingBuffer() = delete;
  RingBuffer(const RingBuffer &) = delete;
  RingBuffer operator=(const RingBuffer &) = delete;

  bool next_free(bool retry = false) {
    if (blocks_[index_].status == kBufFree) {
      return true;
    } else {
      if (retry) {
        inno_log_error("retry block %d failed, status", index_);
        return false;
      }
    }

    blocks_[index_].status = kBufUsed;
    if (index_ == block_num_ - 1) {
      index_ = 0;
      if (blocks_[index_].status != kBufFree) {
        inno_log_warning("All blocks has been filled, the 0th block of memory has not been released yet");
        return false;
      }
    } else {
      index_ += 1;
      inno_log_verify(blocks_[index_].status == kBufFree, "block %d is in wrong status", index_);
    }
    return true;
  }

  std::vector<block_t> blocks_;
  int8_t index_ = 0;
  int8_t block_num_ = 0;    // block number of the RingBuf
  int8_t index_used_ = -1;  // start from -1
};

enum PointType {
  POINT_TYPE_LITE_PCD,
  POINT_TYPE_PCD,
  POINT_TYPE_EN_PCD,
};

/***********************
 * class FileRecorder
 ***********************/
class FileRecorder {
 public:
  enum FileType {
    FILE_TYPE_PCD,
    FILE_TYPE_PCD_BINARY,
    FILE_TYPE_CSV,
    FILE_TYPE_INNO_PC,
    FILE_TYPE_INNO_PC_XYZ,
    FILE_TYPE_INNO_CFRAME,
    FILE_TYPE_BAG,
    FILE_TYPE_PNG,
    FILE_TYPE_MAX,
  };

  DEFINE_INNO_COMPACT_STRUCT(LitePcdPoint) {
    float x;  
    float y;  
    float z;  
    float intensity; 
    double timestamp;  
    uint16_t ring_id;   
    uint16_t confidence;
  };
  DEFINE_INNO_COMPACT_STRUCT_END

  DEFINE_INNO_COMPACT_STRUCT(PcdPoint) {
    float x;
    float y;
    float z;
    uint16_t intensity;
    uint16_t channel;
    uint16_t roi;
    uint16_t facet;
    uint16_t is_2nd_return;
    uint16_t multi_return;
    uint16_t confid_level;
    uint16_t flags;
    uint16_t elongation;
    double timestamp;
    uint16_t scanline;
    uint16_t scan_idx;
    uint32_t frame_id;
    uint32_t ring_id;
  };
  DEFINE_INNO_COMPACT_STRUCT_END

  DEFINE_INNO_COMPACT_STRUCT(EnPcdPoint) {
    float x;
    float y;
    float z;
    float h_angle;
    float v_angle;
    uint16_t intensity;
    uint16_t channel;
    uint16_t roi;
    uint16_t facet;
    uint16_t is_2nd_return;
    uint16_t multi_return;
    uint16_t confid_level;
    uint16_t flags;
    uint16_t elongation;
    double timestamp;
    uint16_t scanline;
    uint16_t scan_idx;
    uint32_t frame_id;
    uint32_t ring_id;
  };
  DEFINE_INNO_COMPACT_STRUCT_END

 public:
  static const size_t kMaxBufferSize = 256 * 1024;
  static const size_t kHeaderSize = 500;
  static const size_t kMaxCsvSize_ = 180;
  static const size_t kMaxBlockNum = 1000 * 1000;
  static const size_t kMaxPacketNum = 1000 * 100;
  static const uint32_t kShmBufNum = 25;

 public:
  explicit FileRecorder(const std::string &filename, const uint64_t frame_id, bool reflectance, enum FileType file_type)
      : filename_(filename),
        fd_(-1),
        point_count_(0),
        reflectance_(reflectance),
        file_type_(file_type),
        cframe_converter_(NULL),
        rosbag_stream_(NULL),
        png_stream_(NULL) {
    if (can_record_cframe()) {
      cframe_converter_ = new innovusion::CframeConverter();
      inno_log_verify(cframe_converter_, "cframe_converter_");
    } else if (can_record_bag()) {
      rosbag_stream_ = new innovusion::RosbagRecorder(filename.c_str(), NULL, NULL, -1);
      inno_log_verify(rosbag_stream_, "rosbag_stream_");
    } else if (can_record_png()) {
      png_stream_ = new innovusion::PngRecorder(0, kMaxPacketNum, kMaxBlockNum, false);
      inno_log_verify(png_stream_, "png_stream_");
    }

    if (!can_record_bag()) {
      pcd_buf_.resize(kPCDBufSize_);
      ring_buffer_ = new RingBuffer(kShmBufNum);
      inno_log_verify(ring_buffer_, "ring_buffer_");
      write_dummy_header_();
      start_writefile_thread(frame_id);
    }
    return;
  }

  ~FileRecorder() {
    recorder_exit_cframe_();
    recorder_exit_png_();
    recorder_exit_rosbag_();
  }

  void start_writefile_thread(const uint64_t frame_id) {
    std::thread([&]() {
      // delete existing file first
      if (FILE *fp = fopen(filename_.c_str(), "r")) {
        fclose(fp);
        int r = remove(filename_.c_str());
        inno_log_verify(r == 0, "cannot delete %s", filename_.c_str());
      }
      fd_ = innovusion::InnoUtils::open_file(filename_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      inno_log_verify(fd_ >= 0, "cannot open %s", filename_.c_str());
      inno_log_info("open file %s to record frame %" PRI_SIZEU, filename_.c_str(), frame_id);
      if (is_compact_packet && file_type_ == FILE_TYPE_INNO_PC) {
        char *anglehv_table = new char[kInnoAngleHVTableMaxSize];
        if (inno_lidar_get_anglehv_table(g_lidar_handle, reinterpret_cast<InnoDataPacket *>(anglehv_table)) == 0) {
          size_t write_size = reinterpret_cast<InnoDataPacket *>(anglehv_table)->common.size;
          size_t written = 0;
          while (written < write_size) {
            int r = write(fd_, anglehv_table + written, write_size - written);
            inno_log_verify(r >= 0, "cannot write data to %s", filename_.c_str());
            written += r;
          }
        }
        delete[] anglehv_table;
      }
      auto write_file = [&](const RingBuffer::block_t &b) {
        uint32_t written = 0;
        while (written < b.written) {
          int r = write(fd_, b.buf + written, b.written - written);
          inno_log_verify(r >= 0, "cannot write data to %s", filename_.c_str());
          written += r;
        }
        ring_buffer_->release(b.index);
        inno_log_verify(written == b.written, "written not match %d/%d", written, b.written);
      };

      while (!done_ || ring_buffer_->next_used() >= 0) {
        if (ring_buffer_->next_used() < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        const RingBuffer::block_t &b = ring_buffer_->cur_used();
        write_file(b);
      }

      if (ring_buffer_->cur_using().written > 0) {
        const RingBuffer::block_t &b = ring_buffer_->cur_using();
        write_file(b);
      }
      rewrite_header_();
      std::unique_lock<std::mutex> lck(mtx_);
      write_thread_stopped_ = true;
      cv_.notify_all();
    }).detach();
  }

  bool can_record_packet() const {
    return file_type_ == FILE_TYPE_INNO_PC || file_type_ == FILE_TYPE_INNO_PC_XYZ;
  }

  bool can_record_cframe() const {
    return file_type_ == FILE_TYPE_INNO_CFRAME;
  }

  bool can_record_bag() const {
    return file_type_ == FILE_TYPE_BAG;
  }

  bool can_record_png() const {
    return file_type_ == FILE_TYPE_PNG;
  }


  void add_data_packet(const InnoDataPacket &pkt_in) {
    const InnoDataPacket *packet = NULL;
    InnoDataPacket *new_packet = NULL;
    if (file_type_ == FILE_TYPE_INNO_PC) {
      inno_log_verify(CHECK_SPHERE_POINTCLOUD_DATA(pkt_in.type), "cannot convert type=%d to inno_pc",
                      pkt_in.type);
      packet = &pkt_in;
    } else if (file_type_ == FILE_TYPE_INNO_PC_XYZ) {
      if (CHECK_SPHERE_POINTCLOUD_DATA(pkt_in.type)) {
        char *anglehv_table = nullptr;
        if (pkt_in.type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
            pkt_in.type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
          anglehv_table = new char[kInnoAngleHVTableMaxSize];
          inno_lidar_get_anglehv_table(g_lidar_handle, reinterpret_cast<InnoDataPacket *>(anglehv_table));
        }
        new_packet = InnoDataPacketUtils::convert_to_xyz_pointcloud_malloced(
            pkt_in, nullptr,
            reinterpret_cast<char *>(
                reinterpret_cast<InnoAngleHVTable *>((reinterpret_cast<InnoDataPacket *>(anglehv_table)->payload))
                    ->table));
        inno_log_verify(new_packet, "cannot convert");
        packet = new_packet;
        delete[] anglehv_table;
      } else if (CHECK_XYZ_POINTCLOUD_DATA(pkt_in.type)) {
        packet = &pkt_in;
      } else {
        return;
      }
    } else {
      inno_log_panic("invalid file type %d", file_type_);
    }

    inno_log_verify(InnoDataPacketUtils::check_data_packet(*packet, packet->common.size), "corrupted pkt");
    write_buffer_(packet, packet->common.size);

    if (new_packet) {
      free(new_packet);
    }
    return;
  }

  void add_data_packet_to_cframe(const InnoDataPacket &pkt) {
    inno_cframe_header *cframe = cframe_converter_->add_data_packet(&pkt, 1);
    if (cframe) {
      write_buffer_(cframe, cframe->get_size());
    }
    return;
  }

  void add_data_packet_to_bag(const InnoDataPacket &pkt) {
    rosbag_stream_->add_block(&pkt);
    return;
  }

  void add_data_packet_to_png(const InnoDataPacket &pkt) {
    png_stream_->capture(&pkt);
    return;
  }

  void add_points(const uint64_t frame_id, const double x, const double y, const double z, const uint32_t ref,
                  const uint32_t channel, const uint32_t in_roi, const uint32_t facet, const uint32_t m_ret,
                  const uint32_t confidence_level, const uint32_t flags, const uint32_t elongation,
                  const double timestamp_sec, const uint32_t scanline, const uint32_t scan_idx,
                  const uint32_t is_2nd_return, const uint32_t ring_id) {
    int wri = -1;
    const char *row_format;
    switch (file_type_) {
      case FILE_TYPE_CSV:
      case FILE_TYPE_PCD:
        memset(&pcd_buf_[0], 0, kPCDBufSize_);
        row_format = file_type_ == FILE_TYPE_CSV ? "%.3f,%.3f,%.3f,%u,%u,%u,%u,%u,%u,%u,%u,%u,"
                                                   "%.5f,%u,%u,%" PRI_SIZEU ",%u\n "
                                                 : "%.3f %.3f %.3f %u %u %u %u %u %u %u %u %u "
                                                   "%.5f %u %u %" PRI_SIZEU " %u\n";
        wri =
            snprintf(&pcd_buf_[0], kPCDBufSize_, row_format, x, y, z, ref, channel, in_roi, facet, is_2nd_return, m_ret,
                     confidence_level, flags, elongation, timestamp_sec, scanline, scan_idx, frame_id, ring_id);
        write_buffer_(&pcd_buf_[0], wri);
        break;

      case FILE_TYPE_PCD_BINARY: {
        wri = sizeof(PcdPoint);
        PcdPoint *pt = reinterpret_cast<PcdPoint *>(&pcd_buf_[0]);
        pt->x = x;
        pt->y = y;
        pt->z = z;
        pt->intensity = ref;
        pt->channel = channel;
        pt->roi = in_roi;
        pt->facet = facet;
        pt->is_2nd_return = is_2nd_return;
        pt->multi_return = m_ret;
        pt->confid_level = confidence_level;
        pt->flags = flags;
        pt->elongation = elongation;
        pt->timestamp = timestamp_sec;
        pt->scanline = scanline;
        pt->scan_idx = scan_idx;
        pt->frame_id = frame_id;
        pt->ring_id = ring_id;
        write_buffer_(pt, wri);
        break;
      }
      default:
        inno_log_panic("invalid type %d", file_type_);
        break;
    }

    if (wri > 0) {
      point_count_++;
    }
    return;
  }

  void add_lite_points(const double x, const double y, const double z, const uint32_t ref,
                       const uint32_t confidence_level, const double timestamp_sec, const uint32_t ring_id) {
    int wri = -1;
    const char *row_format;
    switch (file_type_) {
      case FILE_TYPE_CSV:
      case FILE_TYPE_PCD:
        memset(&pcd_buf_[0], 0, kPCDBufSize_);
        row_format = "%.3f,%.3f,%.3f,%u,%.5f,%u,%u","\n ";

        row_format = file_type_ == FILE_TYPE_CSV ? "%.3f,%.3f,%.3f,%u,%.5f,%u,%u\n" : 
                                                   "%.3f %.3f %.3f %u %.5f %u %u\n";
        wri =
            snprintf(&pcd_buf_[0], kPCDBufSize_, row_format, x, y, z, ref, timestamp_sec, ring_id, confidence_level);
        write_buffer_(&pcd_buf_[0], wri);
        break;

      case FILE_TYPE_PCD_BINARY: {
        wri = sizeof(LitePcdPoint);
        LitePcdPoint *pt = reinterpret_cast<LitePcdPoint *>(&pcd_buf_[0]);
        pt->x = x;
        pt->y = y;
        pt->z = z;
        pt->intensity = ref;
        pt->confidence = confidence_level;
        pt->timestamp = timestamp_sec;
        pt->ring_id = ring_id;
        write_buffer_(pt, wri);
        break;
      }
      default:
        inno_log_panic("invalid type %d", file_type_);
        break;
    }

    if (wri > 0) {
      point_count_++;
    }
    return;
  }

  void add_en_points(const uint64_t frame_id, const double x, const double y, const double z, const double h_angle,
                     const double v_angle, const uint32_t ref, const uint32_t channel, const uint32_t in_roi,
                     const uint32_t facet, const uint32_t m_ret, const uint32_t confidence_level, const uint32_t flags,
                     const uint32_t elongation, const double timestamp_sec, const uint32_t scanline,
                     const uint32_t scan_idx, const uint32_t is_2nd_return, const uint32_t ring_id) {
    int wri = -1;
    const char *row_format;
    switch (file_type_) {
      case FILE_TYPE_CSV:
      case FILE_TYPE_PCD:
        memset(&pcd_buf_[0], 0, kPCDBufSize_);
        row_format = file_type_ == FILE_TYPE_CSV ? "%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,%u,%u,%u,%u,%u,%u,"
                                                   "%.5f,%u,%u,%" PRI_SIZEU ",%u\n "
                                                 : "%.3f %.3f %.3f %.3f %.3f %u %u %u %u %u %u %u %u %u "
                                                   "%.5f %u %u %" PRI_SIZEU " %u\n";
        wri = snprintf(&pcd_buf_[0], kPCDBufSize_, row_format, x, y, z, h_angle, v_angle, ref, channel, in_roi, facet,
                       is_2nd_return, m_ret, confidence_level, flags, elongation, timestamp_sec, scanline, scan_idx,
                       frame_id, ring_id);
        write_buffer_(&pcd_buf_[0], wri);
        break;

      case FILE_TYPE_PCD_BINARY: {
        wri = sizeof(EnPcdPoint);
        EnPcdPoint *pt = reinterpret_cast<EnPcdPoint *>(&pcd_buf_[0]);
        pt->x = x;
        pt->y = y;
        pt->z = z;
        pt->h_angle = h_angle;
        pt->v_angle = v_angle;
        pt->intensity = ref;
        pt->channel = channel;
        pt->roi = in_roi;
        pt->facet = facet;
        pt->is_2nd_return = is_2nd_return;
        pt->multi_return = m_ret;
        pt->confid_level = confidence_level;
        pt->flags = flags;
        pt->elongation = elongation;
        pt->timestamp = timestamp_sec;
        pt->scanline = scanline;
        pt->scan_idx = scan_idx;
        pt->frame_id = frame_id;
        pt->ring_id = ring_id;
        write_buffer_(pt, wri);
        break;
      }
      default:
        inno_log_panic("invalid type %d", file_type_);
        break;
    }

    if (wri > 0) {
      point_count_++;
    }
    return;
  }

 private:
  void recorder_exit_cframe_() {
    if (can_record_cframe() && cframe_converter_) {
      inno_cframe_header *cframe = cframe_converter_->close_current_frame();
      if (cframe) {
        write_buffer_(cframe, cframe->get_size());
      }
    }

    if (cframe_converter_) {
      delete cframe_converter_;
      cframe_converter_ = NULL;
    }
  }

  void recorder_exit_png_() {
    if (png_stream_) {
      std::vector<char> buf;
      buf.reserve(1024 * 1024 * 6);
      png_stream_->save(&buf);
      uint32_t kBlockSize = ring_buffer_->kBlockSize;
      uint32_t append_times = buf.size() / kBlockSize + 1;
      if (buf.size() % kBlockSize == 0) {
        append_times -= 1;
      }
      for (int i = 0; i < append_times; i++) {
        bool r = false;
        if (i != append_times - 1) {
          r = ring_buffer_->append(buf.data() + i * kBlockSize, kBlockSize);
        } else {
          r = ring_buffer_->append(buf.data() + i * kBlockSize, buf.size() - i * kBlockSize);
        }
        inno_log_verify(r, "cannot write data to ring buffkBlockSizeer,kBlockSize filename is %s", filename_.c_str());
      }
      delete png_stream_;
      png_stream_ = NULL;
    }
  }

  void recorder_exit_rosbag_() {
    if (rosbag_stream_) {
      rosbag_stream_->add_block(NULL);
      delete rosbag_stream_;
      rosbag_stream_ = NULL;
    }
    if (!can_record_bag()) {
      done_ = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::unique_lock<std::mutex> lck(mtx_);
      while (!write_thread_stopped_) {
        cv_.wait(lck);
      }
      delete ring_buffer_;
      ring_buffer_ = NULL;
    } else {
      done_ = true;
    }
  }

  void write_dummy_header_() {
    size_t write_size = 0;
    switch (file_type_) {
      case FILE_TYPE_CSV:
        write_size = kMaxCsvSize_;
        break;
      case FILE_TYPE_PCD:
      case FILE_TYPE_PCD_BINARY:
        write_size = sizeof(header_buffer_);
        break;
      case FILE_TYPE_INNO_PC:
      case FILE_TYPE_INNO_PC_XYZ:
      case FILE_TYPE_INNO_CFRAME:
      case FILE_TYPE_BAG:
      case FILE_TYPE_PNG:
        return;
      default:
        inno_log_panic("invalid type %d", file_type_);
    }
    memset(header_buffer_, ' ', write_size);
    header_buffer_[write_size - 1] = '\n';
    bool r = ring_buffer_->append(header_buffer_, write_size);
    inno_log_verify(r, "cannot write data to ring buffer, filename is %s", filename_.c_str());
  }

  void rewrite_header_() {
    const char *lite_pcd_header =
        "# .PCD v.7 - Point Cloud Data file format\n"
        "FIELDS "
        "x y z %s timestamp ring_id confid_level\n"
        "SIZE 4 4 4 4 8 2 2\n"
        "TYPE F F F F F U U\n"
        "COUNT 1 1 1 1 1 1 1\n"
        "WIDTH %" PRI_SIZEU "\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS %" PRI_SIZEU "\n"
        "DATA %s";

    const char *pcd_header =
        "# .PCD v.7 - Point Cloud Data file format\n"
        "FIELDS "
        "x y z %s channel roi facet is_2nd_return multi_return confid_level "
        "flag elongation timestamp "
        "scanline scan_idx frame_id ring_id\n"
        "SIZE 4 4 4 2 2 2 2 2 2 2 2 2 8 2 2 4 4\n"
        "TYPE F F F U U U U U U U U U F U U U U\n"
        "COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
        "WIDTH %" PRI_SIZEU "\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS %" PRI_SIZEU "\n"
        "DATA %s";

    const char *en_pcd_header =
        "# .PCD v.7 - Point Cloud Data file format\n"
        "FIELDS "
        "x y z h_angle v_angle %s channel roi facet is_2nd_return multi_return confid_level "
        "flag elongation timestamp "
        "scanline scan_idx frame_id ring_id\n"
        "SIZE 4 4 4 4 4 2 2 2 2 2 2 2 2 2 8 2 2 4 4\n"
        "TYPE F F F F F U U U U U U U U U F U U U U\n"
        "COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
        "WIDTH %" PRI_SIZEU "\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS %" PRI_SIZEU "\n"
        "DATA %s";
    const char *csv_header =
        "x,y,z,%s,channel,roi,facet,is_2nd_return,multi_return,confid_level,"
        "flag,elongation,timestamp,"
        "scanline,scan_idx,frame_id,ring_id";

    const char *lite_pcd_csv_header = "x,y,z,%s,timestamp,ring_id,confid_level";
  
    int r;
    uint32_t max_write_size = 0;

    switch (file_type_) {
      case FILE_TYPE_CSV:
        if (recorder_point_type == POINT_TYPE_EN_PCD) {
          r = snprintf(header_buffer_, sizeof(header_buffer_), csv_header,
                       reflectance_ ? "h_angle,v_angle,reflectance" : "h_angle,v_angle,intensity");
        } else if (recorder_point_type == POINT_TYPE_LITE_PCD) {
          r = snprintf(header_buffer_, sizeof(header_buffer_), lite_pcd_csv_header,
                       reflectance_ ? "reflectance" : "intensity");
        } else {
          r = snprintf(header_buffer_, sizeof(header_buffer_), csv_header, reflectance_ ? "reflectance" : "intensity");
        }
        max_write_size = kMaxCsvSize_ - 1;
        break;
      case FILE_TYPE_PCD:
      case FILE_TYPE_PCD_BINARY:
        if (recorder_point_type == POINT_TYPE_EN_PCD) {
          r = snprintf(header_buffer_, sizeof(header_buffer_), en_pcd_header,
                       reflectance_ ? "reflectance" : "intensity", point_count_, point_count_,
                       file_type_ == FILE_TYPE_PCD ? "ascii" : "binary");
        } else if (recorder_point_type == POINT_TYPE_LITE_PCD) {
          r = snprintf(header_buffer_, sizeof(header_buffer_), lite_pcd_header,
                       reflectance_ ? "reflectance" : "intensity", point_count_, point_count_,
                       file_type_ == FILE_TYPE_PCD ? "ascii" : "binary");
        } else {
          r = snprintf(header_buffer_, sizeof(header_buffer_), pcd_header, reflectance_ ? "reflectance" : "intensity",
                       point_count_, point_count_, file_type_ == FILE_TYPE_PCD ? "ascii" : "binary");
        }
        max_write_size = sizeof(header_buffer_) - 1;
        break;
      case FILE_TYPE_INNO_PC:
      case FILE_TYPE_INNO_PC_XYZ:
      case FILE_TYPE_INNO_CFRAME:
      case FILE_TYPE_BAG:
      case FILE_TYPE_PNG:
        return;
      default:
        inno_log_panic("invalid type %d", file_type_);
    }

    inno_log_verify(r < int32_t(sizeof(header_buffer_)), "buffer too small %d vs %" PRI_SIZELU, r,
                    sizeof(header_buffer_));
    inno_log_info("write %" PRI_SIZEU " points to pcd file %s", point_count_, filename_.c_str());
    uint64_t fsize = lseek(fd_, 0, SEEK_END);
    if (point_count_ > 0) {
      inno_log_verify((file_type_ == FILE_TYPE_CSV ? fsize > kMaxCsvSize_ : fsize > kHeaderSize),
                      "the PCD file size is invalid, fsize: %" PRI_SIZEU ", point_count_ is : %" PRI_SIZEU, fsize,
                      point_count_);
    }

    r = lseek(fd_, 0, SEEK_SET);
    inno_log_verify(r == 0, "seek to file begin failed");
    size_t write_size = strlen(header_buffer_);
    inno_log_verify(write_size <= max_write_size, "write_size %" PRI_SIZELU " > header_size %u", write_size,
                    max_write_size);
    uint32_t written = 0;
    while (written < write_size) {
      int r = write(fd_, header_buffer_ + written, write_size - written);
      inno_log_verify(r >= 0, "cannot write data to %s", filename_.c_str());
      written += r;
    }
    inno_log_verify(written == write_size, "written %d,  write_size: %" PRI_SIZELD, written, write_size);
    close(fd_);
    fd_ = -1;
    return;
  }

  void write_buffer_(const void *src, size_t src_len) {
    bool r = false;
    r = ring_buffer_->append(reinterpret_cast<const char *>(src), src_len);
    // retry
    if (!r) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      inno_log_info("retry append");
      r = ring_buffer_->append(reinterpret_cast<const char *>(src), src_len, true);
    }
    inno_log_verify(r, "cannot write data to ring buffer, filename is %s", filename_.c_str());
  }

 private:
  std::string filename_;
  int fd_ = -1;
  uint64_t point_count_ = 0;
  bool reflectance_;
  enum FileType file_type_;
  char header_buffer_[kHeaderSize];
  innovusion::CframeConverter *cframe_converter_ = NULL;
  innovusion::RosbagRecorder *rosbag_stream_ = NULL;
  innovusion::PngRecorder *png_stream_ = NULL;
  RingBuffer *ring_buffer_ = NULL;
  char bytes_buffer_[kMaxBufferSize] = {0};
  bool done_ = false;
  bool write_thread_stopped_ = false;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::vector<char> pcd_buf_;
  static const uint8_t kPCDBufSize_ = 128;
};

/***********************
 * class ExampleProcessor
 ***********************/
class ExampleProcessor {
 public:
  explicit ExampleProcessor(const std::string &filename, const int64_t frame_start, const int64_t frame_number,
                            const int64_t file_number, int use_xyz_point, const std::string &latency_file,
                            int ascii_pcd, int extract_message, uint64_t run_time, int lidar_handle,
                            const std::string anglehv_table_file)
      : filename_(filename),
        frame_start_(frame_start),
        frame_number_(frame_number),
        file_number_(file_number),
        use_xyz_point_(use_xyz_point),
        latency_file_(latency_file),
        current_frame_(-1),
        frame_so_far_(-1),
        file_so_far_(0),
        file_recorder_(NULL),
        file_type_(FileRecorder::FILE_TYPE_PCD),
        done_(false),
        transform_cs_(false),
        yaw_(0.0),
        pitch_(0.0),
        roll_(0.0),
        run_time_(run_time),
        lidar_handle_(lidar_handle) {
    // create status and message file
    size_t dot = filename_.find_last_of(".");
    if (dot != std::string::npos) {
      filename_base_ = filename_.substr(0, dot);
      file_extension_ = filename_.substr(dot);
      if (strcasecmp(file_extension_.c_str(), ".csv") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_CSV;
      } else if (strcasecmp(file_extension_.c_str(), ".inno_pc") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_INNO_PC;
      } else if (strcasecmp(file_extension_.c_str(), ".inno_pc_xyz") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_INNO_PC_XYZ;
      } else if (strcasecmp(file_extension_.c_str(), ".inno_cframe") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_INNO_CFRAME;
      } else if (strcasecmp(file_extension_.c_str(), ".bag") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_BAG;
      } else if (strcasecmp(file_extension_.c_str(), ".png") == 0) {
        file_type_ = FileRecorder::FILE_TYPE_PNG;
        inno_log_info("force to use sphere coordinate to record PNG file");
      } else {
        file_type_ = ascii_pcd ? FileRecorder::FILE_TYPE_PCD : FileRecorder::FILE_TYPE_PCD_BINARY;
      }
    } else {
      filename_base_ = filename_;
      file_type_ = ascii_pcd ? FileRecorder::FILE_TYPE_PCD : FileRecorder::FILE_TYPE_PCD_BINARY;
    }
    if (extract_message) {
      status_filename_ = filename_ + "_status_log.txt";
      status_fd_ = innovusion::InnoUtils::open_file(status_filename_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
      inno_log_verify(status_fd_ >= 0, "cannot open %s", status_filename_.c_str());
      msg_filename_ = filename_ + "_message_log.txt";
      msg_fd_ = innovusion::InnoUtils::open_file(msg_filename_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
      inno_log_verify(msg_fd_ >= 0, "cannot open %s", msg_filename_.c_str());
    } else {
      status_fd_ = -1;
      msg_fd_ = -1;
    }
    if (!latency_file_.empty()) {
      latency_recorder_ = new std::fstream(latency_file_, std::ios::out | std::ios::app);
      inno_log_verify(latency_recorder_ && latency_recorder_->is_open(), "cannot open %s save latency",
                      latency_file_.c_str());
    }

    if (!anglehv_table_file.empty()) {
      anglehv_table_ = new char[kInnoAngleHVTableMaxSize];
      std::ifstream ifs(anglehv_table_file, std::ios::binary);
      if (ifs.is_open()) {
        ifs.read(anglehv_table_, sizeof(InnoDataPacket));
        uint32_t payload_len = reinterpret_cast<InnoDataPacket *>(anglehv_table_)->common.size - sizeof(InnoDataPacket);
        ifs.read(anglehv_table_ + sizeof(InnoDataPacket), payload_len);
        ifs.close();
        inno_log_verify(
            InnoDataPacketUtils::check_data_packet(*reinterpret_cast<InnoDataPacket *>(anglehv_table_),
                                                   reinterpret_cast<InnoDataPacket *>(anglehv_table_)->common.size),
            "verify angletable failed");
      } else {
        inno_log_panic("cannot open %s", anglehv_table_file.c_str());
        delete[] anglehv_table_;
        anglehv_table_ = nullptr;
      }
    }
    galvo_check_fd_ = -1;
    max_distance_fd_ = -1;
    start_check_datacallback_thread();
  }

  ~ExampleProcessor() {
    if (file_recorder_) {
      delete file_recorder_;
      file_recorder_ = NULL;
    }
    if (msg_fd_ >= 0) {
      close(msg_fd_);
      msg_fd_ = -1;
    }
    if (status_fd_ >= 0) {
      close(status_fd_);
      status_fd_ = -1;
    }
    if (galvo_check_fd_ >= 0) {
      close(galvo_check_fd_);
      galvo_check_fd_ = -1;
    }
    if (max_distance_fd_ >= 0) {
      close(max_distance_fd_);
      max_distance_fd_ = -1;
    }
    if (latency_recorder_) {
      latency_recorder_->close();
      delete latency_recorder_;
      latency_recorder_ = NULL;
    }
    if (anglehv_table_) {
      delete[] anglehv_table_;
      anglehv_table_ = nullptr;
    }
    inno_log_info(
        "----------Summary----------\n"
        "frame_counter = %" PRI_SIZEU
        ", "
        "miss_frame_counter = %" PRI_SIZEU
        ", "
        "miss_sub_frame_gap_count = %" PRI_SIZEU
        ", "
        "empty_sub_frame_count = %" PRI_SIZEU
        ", "
        "miss_sub_frame_last_one_counter = %" PRI_SIZEU
        ", "
        "miss_sub_frame_except_last_one_counter = %" PRI_SIZEU "\n",
        summary_package_.get_frame_count(), summary_package_.get_miss_frame_count(),
        summary_package_.get_miss_sub_frame_gap_count(), summary_package_.get_empty_sub_frame_count(),
        summary_package_.get_miss_sub_frame_last_one_count(),
        summary_package_.get_empty_sub_frame_except_last_one_count());
  }

  void set_done() {
    done_ = true;
  }

  void async_set_done(int sleep_sec) {
    std::thread([&]() {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
      done_ = true;
    }).join();
  }

  bool is_done() {
    return done_;
  }

  void set_need_process_data(int src) {
    need_process_data_ = src;
  }

  void set_orientation(double yaw, double pitch, double roll) {
    yaw_ = yaw;
    pitch_ = pitch;
    roll_ = roll;
    if (yaw != 0 || pitch != 0 || roll != 0) {
      transform_cs_ = true;
    }
  }

  void set_ring_id_converter(RingIdConverterInterface *converter) {
    ring_id_converter_ = converter;
  }

  void save_message_to_file(const char *message) {
    if (msg_fd_ >= 0) {
      int write_len = snprintf(msg_buffer_, sizeof(msg_buffer_), "%s\n", message);
      int r = write(msg_fd_, msg_buffer_, write_len);
      if (r < 0) {
        inno_log_warning("cannot write data to %s", msg_filename_.c_str());
      }
    }
  }

  void save_galvo_check_result_to_file(const char *message) {
    if (galvo_check_fd_ >= 0) {
      int write_len = snprintf(galvo_check_buffer_, sizeof(galvo_check_buffer_), "%s\n", message);
      int r = write(galvo_check_fd_, galvo_check_buffer_, write_len);
      if (r < 0) {
        inno_log_warning("cannot write data to %s", galvo_check_filename_.c_str());
      }
    }
  }

  void save_max_distance_result_to_file(const char *message) {
    if (max_distance_fd_ >= 0) {
      int write_len = snprintf(max_distance_buffer_, sizeof(max_distance_buffer_), "%s\n", message);
      int r = write(max_distance_fd_, max_distance_buffer_, write_len);
      if (r < 0) {
        inno_log_warning("cannot write data to %s", max_distance_filename_.c_str());
      }
    }
  }

  void create_galvo_check_result_file() {
    time_t rawtime;
    struct tm *tm_info;
    char temp[80] = {0};
    struct tm result_tm;
    time(&rawtime);
#if !(defined(__MINGW64__) || defined(_WIN32))
    tm_info = localtime_r(&rawtime, &result_tm);
#else
    tm_info = localtime_s(&result_tm, &rawtime) == 0 ? &result_tm : NULL;
#endif
    strftime(temp, sizeof(temp) - 1, "%Y%m%d-%H%M%S", tm_info);
    galvo_check_filename_ = "galvo-check-";
    galvo_check_filename_.append(temp).append(".txt");
    galvo_check_fd_ =
        innovusion::InnoUtils::open_file(galvo_check_filename_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    inno_log_verify(galvo_check_fd_ >= 0, "cannot open %s", galvo_check_filename_.c_str());
    const char *galvo_check_head_ =
        "Frame\tC_Code\tP_Count\t"
        "Ref_x\tRef_y\tRef_z\t"
        "CoeA\tCoeB\tCoeC\tCoeD\t"
        "Rot_x\tRot_y\tRot_z\t"
        "Speed\tSpeed_Acc\tR2\tCos(θ)\t"
        "D_Angle\tMean_Angle\tAngle_R2\t"
        "Fault_Status\tF_Times\t"
        "Valid_Count\tSpell_Time(us)";
    save_galvo_check_result_to_file(galvo_check_head_);
  }

  void create_max_distance_result_file() {
    time_t rawtime;
    struct tm *tm_info;
    char temp[80] = {0};
    struct tm result_tm;
    time(&rawtime);
#if !(defined(__MINGW64__) || defined(_WIN32))
    tm_info = localtime_r(&rawtime, &result_tm);
#else
    tm_info = localtime_s(&result_tm, &rawtime) == 0 ? &result_tm : NULL;
#endif
    strftime(temp, sizeof(temp) - 1, "%Y%m%d-%H%M%S", tm_info);
    max_distance_filename_ = "max-distance-";
    max_distance_filename_.append(temp).append(".txt");
    max_distance_fd_ =
        innovusion::InnoUtils::open_file(max_distance_filename_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    inno_log_verify(max_distance_fd_ >= 0, "cannot open %s", max_distance_filename_.c_str());
  }

  static void message_callback_s(const int lidar_handle, void *ctx, const uint32_t from_remote,
                                 const enum InnoMessageLevel level, const enum InnoMessageCode code,
                                 const char *error_message) {
    inno_log_info("level = %d code = %d\n", level, code);
    switch (level) {
      case INNO_MESSAGE_LEVEL_INFO:
        inno_log_info("content = %s\n", error_message);
        if (code == INNO_MESSAGE_CODE_READ_FILE_END) {
          reinterpret_cast<ExampleProcessor *>(ctx)->async_set_done(1);
        }
        break;
      case INNO_MESSAGE_LEVEL_WARNING:
        inno_log_warning("content = %s\n", error_message);
        break;
      case INNO_MESSAGE_LEVEL_ERROR:
        inno_log_error("content = %s\n", error_message);
        break;
      case INNO_MESSAGE_LEVEL_FATAL:
        inno_log_fatal("content = %s\n", error_message);
        break;
      case INNO_MESSAGE_LEVEL_CRITICAL:
        if (code == INNO_MESSAGE_CODE_CANNOT_READ) {
          inno_log_error("critical error, abort.");
          reinterpret_cast<ExampleProcessor *>(ctx)->set_done();
          break;
        } else {
          inno_log_fatal("content = %s\n", error_message);
          break;
        }
      default:
        inno_log_info("content = %s\n", error_message);
        break;
    }
    if (code == INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT) {
      reinterpret_cast<ExampleProcessor *>(ctx)->save_galvo_check_result_to_file(error_message);
    } else if (code == INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT) {
      reinterpret_cast<ExampleProcessor *>(ctx)->save_max_distance_result_to_file(error_message);
    } else {
      reinterpret_cast<ExampleProcessor *>(ctx)->save_message_to_file(error_message);
    }
    return;
  }

  void save_status_to_file(const InnoStatusPacket *pkt) {
    if (status_fd_ >= 0) {
      int write_len = InnoDataPacketUtils::printf_status_packet(*pkt, status_buffer_, sizeof(status_buffer_));

      if (write_len > 0) {
        int r = write(status_fd_, status_buffer_, write_len);
        if (r < 0) {
          inno_log_warning("cannot write data to %s", status_filename_.c_str());
        }
      }
    }
  }

  static int status_callback_s(const int lidar_handle, void *ctx, const InnoStatusPacket *pkt) {
    inno_log_verify(pkt, "pkt");

    // sanity check
    if (!InnoDataPacketUtils::check_status_packet(*pkt, 0)) {
      inno_log_error("corrupted pkt->idx = %" PRI_SIZEU "", pkt->idx);
      return 0;
    }

    static size_t cnt = 0;
    if (cnt++ % 100 == 1) {
      constexpr size_t buf_size = 2048;
      char buf[buf_size]{0};

      int ret = InnoDataPacketUtils::printf_status_packet(*pkt, buf, buf_size);
      if (ret > 0) {
        inno_log_info("Received status packet #%" PRI_SIZELU ": %s", cnt, buf);
      } else {
        inno_log_warning(
            "Received status packet #"
            "%" PRI_SIZELU ": errorno: %d",
            cnt, ret);
      }
    }

    reinterpret_cast<ExampleProcessor *>(ctx)->save_status_to_file(pkt);
    return 0;
  }

  void get_latency_info(const InnoDataPacket& pkt) {
    auto cur_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    auto latency = cur_time - pkt.common.ts_start_us;
    *latency_recorder_ << "frame_idx=" << pkt.idx << ","
      << "latency=" << latency << ","
      << "ts_start_us=" << static_cast<int64_t>(pkt.common.ts_start_us) << ","
      << "accept_time=" << cur_time << ","
      << "item_count=" << pkt.item_number << "\n";
    inno_log_info("frame_idx=%" PRI_SIZEU ", latency=%lf", pkt.idx, latency);
  }

  static int data_callback_s(const int lidar_handle, void *ctx, const InnoDataPacket *pkt) {
    static size_t cnt = 0;
    if (cnt++ % 1000 == 0) {
      inno_log_info("Received data packet #%" PRI_SIZELU "\n", cnt);
    }
    inno_log_verify(pkt, "pkt");

    // sanity check
    if (!CHECK_XYZ_POINTCLOUD_DATA(pkt->type) && !CHECK_SPHERE_POINTCLOUD_DATA(pkt->type)) {
      inno_log_warning("ignore type %u", pkt->type);
      return 0;
    }

    // sanity check
    if (!InnoDataPacketUtils::check_data_packet(*pkt, 0)) {
      inno_log_warning("corrupted pkt");
      return 0;
    }

    static uint64_t cur_frame_id = 0;
    static uint32_t cur_frame_point = 0;
    if (reinterpret_cast<ExampleProcessor *>(ctx)->get_recorder_(pkt->idx, pkt->use_reflectance,
                                                                 pkt->common.ts_start_us / kUsInSecond) &&
        !reinterpret_cast<ExampleProcessor *>(ctx)->latency_file_.empty()) {
      reinterpret_cast<ExampleProcessor *>(ctx)->get_latency_info(*pkt);
    }
    if (cur_frame_id != pkt->idx) {
      inno_log_info("frame idx:%" PRI_SIZEU ", point_count:%d", cur_frame_id, cur_frame_point);
      cur_frame_id = pkt->idx;
      cur_frame_point = 0;
    }
    cur_frame_point += InnoDataPacketUtils::get_points_count(*pkt);

    reinterpret_cast<ExampleProcessor *>(ctx)->process_data_(*pkt);

    if (reinterpret_cast<ExampleProcessor *>(ctx)->run_time_ > 0) {
      static uint64_t start_time = innovusion::InnoUtils::get_time_ms();
      static uint64_t run_time_check_frame_id = 0;
      if (cur_frame_id % 10 == 0 && cur_frame_id != run_time_check_frame_id) {
        run_time_check_frame_id = cur_frame_id;
        uint64_t now_time = innovusion::InnoUtils::get_time_ms();
        if (now_time - start_time > reinterpret_cast<ExampleProcessor *>(ctx)->run_time_ * 1000) {
          inno_log_info("get_pcd start_time: %" PRI_SIZEU " ms, end_time: %" PRI_SIZEU " ms, run_time:%" PRI_SIZEU
                        " ms",
                        start_time, now_time, reinterpret_cast<ExampleProcessor *>(ctx)->run_time_ * 1000);
          reinterpret_cast<ExampleProcessor *>(ctx)->set_done();
        }
      }
    }

    return 0;
  }

 private:
  inline FileRecorder *get_recorder_(const int64_t frame_id, const bool reflectance, double timestamp = 0) {
    // how many frames we have seen so far?
    if (frame_so_far_ == -1 || current_frame_ != frame_id) {
      current_frame_ = frame_id;
      frame_so_far_++;
    }

    if (filename_.size() == 0) {
      // do nothing
    } else if (frame_number_ == -1) {
      // do nothing
    } else if (frame_so_far_ < frame_start_) {
      // do nothing
    } else if ((frame_so_far_ < frame_start_ + frame_number_ * file_number_) || run_time_ > 0) {
      // in record range
      if ((frame_so_far_ >= frame_start_ + frame_number_ * file_so_far_) && run_time_ == 0) {
        if (file_recorder_) {
          delete file_recorder_;
          file_recorder_ = NULL;
        }
      }
      if (file_recorder_ == NULL) {
        // create FileRecorder
        std::string fn = filename_;
        if (file_number_ != 1 || ::filename_with_timestamp) {
          if (::filename_with_timestamp) {
            int64_t timestamp_sec = static_cast<int64_t>(timestamp);
            std::tm *tm = std::gmtime(&timestamp_sec);
            std::ostringstream oss;
            oss << std::put_time(tm, "_%Y%m%d_%H%M%S");
            std::string tmp = oss.str();
            fn =
                filename_base_ + tmp + "_" + std::to_string(timestamp) + file_extension_;
          } else {
            fn = filename_base_ + "-" + std::to_string(file_so_far_) + file_extension_;
          }
        }
        file_recorder_ = new FileRecorder(fn, frame_id, reflectance, file_type_);
        file_so_far_++;
        inno_log_verify(file_recorder_ != NULL, "cannot create file recorder");
      }
    } else {
      set_done();
      // record done, close the recorder
      if (file_recorder_) {
        delete file_recorder_;
        file_recorder_ = NULL;
      }
    }
    return file_recorder_;
  }

  void process_data_(const InnoDataPacket &pkt) {
    if (done_) {
      static bool print_skip = false;
      if (!print_skip) {
        print_skip = true;
        inno_log_info("data capture process has been done, skip data packet.");
      }
      return;
    }

    if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt.type)) {
      is_compact_packet = true;
    }

    receive_datacallback_ = true;
    // we have two ways of processing the data packet
    if (need_process_data_) {
      if (CHECK_SPHERE_POINTCLOUD_DATA(pkt.type)) {
          process_data_cpoint_(pkt);
      } else if (CHECK_XYZ_POINTCLOUD_DATA(pkt.type)) {
        process_data_xyz_point_(pkt);
      } else {
        inno_log_panic("invalid type %d", pkt.type);
      }
    }

    if (frame_so_far_ < frame_start_) {
      // not in record range yet
    } else if ((filename_.size() == 0) || (frame_number_ == -1) || (run_time_ > 0) ||
               (frame_so_far_ < frame_start_ + frame_number_ * file_number_)) {
      // summary date package frame and sub frame
      summary_package_.summary_data_package(pkt);
    }
  }

  void rotate_coordinate(double angle, double *x, double *y) {
    // https://farside.ph.utexas.edu/teaching/celestial/Celestial/node122.html
    double x1 = cos(angle) * *x + sin(angle) * *y;
    double y1 = -sin(angle) * *x + cos(angle) * *y;
    *x = x1;
    *y = y1;
    return;
  }

  void sensor_to_world_cs(double *new_x, double *new_y, double *new_z, double x, double y, double z, double roll,
                          double pitch, double yaw) {
    // Rotational coordinate system transformations only.  Does not translate
    // the origin.
    rotate_coordinate(yaw, &y, &z);
    rotate_coordinate(pitch, &x, &z);
    rotate_coordinate(roll, &y, &x);
    *new_x = x;
    *new_y = y;
    *new_z = z;
  }


  void recoder_xyz_point(const InnoDataPacket &pkt, double frame_timestamp_sec, FileRecorder *recorder) {
    const InnoXyzPoint *points = reinterpret_cast<const InnoXyzPoint *>(pkt.payload);
    double x, y, z;
    for (uint32_t i = 0; i < pkt.item_number; i++) {
      const InnoXyzPoint &pt = points[i];
      x = pt.x;
      y = pt.y;
      z = pt.z;
      if (transform_cs_) {
        sensor_to_world_cs(&x, &y, &z, pt.x, pt.y, pt.z, roll_, pitch_, yaw_);
      }
      recorder->add_points(pkt.idx, x, y, z, pt.refl, pt.channel, pt.in_roi, pt.facet, pt.multi_return,
                           pkt.confidence_level, pt.type, pt.elongation,
                           frame_timestamp_sec + pt.ts_10us / k10UsInSecond, pt.scan_id, pt.scan_idx, pt.is_2nd_return,
                           pt.ring_id);
    }
  }


  void recoder_en_xyz_point(const InnoDataPacket &pkt, double frame_timestamp_sec, FileRecorder *recorder) {
    const InnoEnXyzPoint *points = reinterpret_cast<const InnoEnXyzPoint *>(pkt.payload);
    double x, y, z;
    for (uint32_t i = 0; i < pkt.item_number; i++) {
      const InnoEnXyzPoint &pt = points[i];
      x = pt.x;
      y = pt.y;
      z = pt.z;
      if (transform_cs_) {
        sensor_to_world_cs(&x, &y, &z, pt.x, pt.y, pt.z, roll_, pitch_, yaw_);
      }

      recorder->add_points(pkt.idx, x, y, z, pkt.use_reflectance ? pt.reflectance : pt.intensity, pt.channel, pt.in_roi,
                           pt.facet, pt.multi_return, pkt.confidence_level, (pt.firing << 2), 0,
                           frame_timestamp_sec + pt.ts_10us / k10UsInSecond, pt.scan_id, pt.scan_idx, pt.is_2nd_return,
                           0);
    }
  }

  void process_data_xyz_point_(const InnoDataPacket &pkt) {
    inno_log_verify(CHECK_XYZ_POINTCLOUD_DATA(pkt.type), "invalid packet %u", pkt.type);

    double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
    // enumerate each xyz_point in the new packet and add it to the recorder
    FileRecorder *recorder = get_recorder_(pkt.idx, pkt.use_reflectance, frame_timestamp_sec);
    if (recorder) {
      if (recorder->can_record_packet()) {
        recorder->add_data_packet(pkt);
      } else if (recorder->can_record_cframe()) {
        recorder->add_data_packet_to_cframe(pkt);
      } else if (recorder->can_record_bag()) {
        recorder->add_data_packet_to_bag(pkt);
      } else if (recorder->can_record_png()) {
        recorder->add_data_packet_to_png(pkt);
      } else {
        if (pkt.type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
          recoder_xyz_point(pkt, frame_timestamp_sec, recorder);
        } else {
          recoder_en_xyz_point(pkt, frame_timestamp_sec, recorder);
        }
      }
    }
    return;
  }


  // convert enblock to enxyz and recorder
  void en_block_to_en_xyz_point(const InnoDataPacket &pkt, FileRecorder *recorder, int unit_size, int return_number) {
    const InnoEnBlock *blocks = reinterpret_cast<const InnoEnBlock *>(pkt.payload);
    double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
    const InnoEnBlock *block = reinterpret_cast<const InnoEnBlock *>(&blocks[0]);

    for (uint32_t i = 0; i < pkt.item_number;
         i++, block = reinterpret_cast<const InnoEnBlock *>(reinterpret_cast<const char *>(block) + unit_size)) {
      // calculate (x,y,z) cartesian coordinate from spherical coordinate
      // for each point in the block
      // 1. use get_full_angles() to restore angle for each channel
      // 2. use get_xyzr_meter() to calculate (x,y,z)
      InnoBlockFullAngles full_angles;
      InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header, static_cast<InnoItemType>(pkt.type));
      double x, y, z, h_angle, v_angle;

      for (uint32_t channel = 0; channel < kInnoChannelNumber; channel++) {
        for (uint32_t m = 0; m < return_number; m++) {
          const InnoEnChannelPoint &pt = block->points[innoblock_get_idx(channel, m)];
          InnoXyzrD xyzr;
          uint32_t scan_id = 0;
          if (pt.radius > 0) {
            if (pkt.type == INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD) {
              scan_id = block->header.scan_id;
              InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, channel, &xyzr,
                                                  static_cast<InnoItemType>(pkt.type));
            } else if (pkt.type == INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD ||
                       pkt.type == INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD) {
              DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(pkt.type);
              int index = block->header.scan_id * 4 + channel;
              scan_id = channel_mapping[index] + block->header.facet * tdc_channel_number;
              InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, scan_id, &xyzr,
                                                  static_cast<InnoItemType>(pkt.type));
            }

            if (force_vehicle_coordinate) {
              x = xyzr.z;
              y = -xyzr.y;
              z = xyzr.x;
            } else {
              x = xyzr.x;
              y = xyzr.y;
              z = xyzr.z;
            }
            h_angle = static_cast<double>(full_angles.angles[channel].h_angle) / kInnoAngleUnitPerDegree;
            v_angle = static_cast<double>(full_angles.angles[channel].v_angle) / kInnoAngleUnitPerDegree;

            if (transform_cs_) {
              sensor_to_world_cs(&x, &y, &z, xyzr.x, xyzr.y, xyzr.z, roll_, pitch_, yaw_);
            }

            if (recorder_point_type == POINT_TYPE_EN_PCD) {
              recorder->add_en_points(
                  pkt.idx, x, y, z, h_angle, v_angle, pkt.use_reflectance ? pt.reflectance : pt.intensity, channel,
                  block->header.in_roi, block->header.facet, m, pkt.confidence_level, pt.type | (pt.firing << 2),
                  pt.elongation, frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id,
                  block->header.scan_idx, pt.is_2nd_return, 0);
            } else if (recorder_point_type == POINT_TYPE_LITE_PCD) {
              recorder->add_lite_points(x, y, z, pkt.use_reflectance ? pt.reflectance : pt.intensity,
                                        pkt.confidence_level,
                                        frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id);
            } else {
              recorder->add_points(pkt.idx, x, y, z, pkt.use_reflectance ? pt.reflectance : pt.intensity, channel,
                                   block->header.in_roi, block->header.facet, m, pkt.confidence_level,
                                   pt.type | (pt.firing << 2), pt.elongation,
                                   frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id,
                                   block->header.scan_idx, pt.is_2nd_return, 0);
            }
          }
        }
      }
    }
  }

  // this functon use robin convert coblock to xyz and recorder
  void co_block_to_en_xyz_point(const InnoDataPacket &pkt, FileRecorder *recorder, int unit_size, int return_number) {
    const InnoCoBlock *blocks = reinterpret_cast<const InnoCoBlock *>(pkt.payload);
    double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
    const InnoCoBlock *block = reinterpret_cast<const InnoCoBlock *>(&blocks[0]);
    int ret = 0;
    if (anglehv_table_ == nullptr) {
      anglehv_table_ = new char[kInnoAngleHVTableMaxSize];
      ret = inno_lidar_get_anglehv_table(lidar_handle_, reinterpret_cast<InnoDataPacket *>(anglehv_table_));
    }

    inno_log_verify(ret == 0, "get anglehv_table failed");
    for (uint32_t i = 0; i < pkt.item_number;
         i++, block = reinterpret_cast<const InnoCoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) {
      // calculate (x,y,z) cartesian coordinate from spherical coordinate
      // for each point in the block
      // 1. use get_full_angles() to restore angle for each channel
      // 2. use get_xyzr_meter() to calculate (x,y,z)
      InnoCoBlockFullAngles full_angles;
      InnoDataPacketUtils::get_block_full_angles(
          &full_angles, block->header, static_cast<InnoItemType>(pkt.type),
          reinterpret_cast<char *>(
              reinterpret_cast<InnoAngleHVTable *>((reinterpret_cast<InnoDataPacket *>(anglehv_table_)->payload))
                  ->table));
      double x, y, z, h_angle, v_angle;

      for (uint32_t channel = 0; channel < kInnoCompactChannelNumber; channel++) {
        for (uint32_t m = 0; m < return_number; m++) {
          const InnoCoChannelPoint &pt = block->points[innocoblock_get_idx(channel, m)];
          InnoXyzrD xyzr;
          uint32_t scan_id = 0;
          if (pt.radius > 0 && InnoDataPacketUtils::is_robinw_inside_fov_point(full_angles.angles[channel])) {
            DEFINE_INNO_ITEM_TYPE_SPECIFIC_DATA(pkt.type);
            int index = block->header.scan_id * kMaxReceiverInSet + channel;
            scan_id = channel_mapping[index] + block->header.facet * tdc_channel_number;
            InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, scan_id, &xyzr,
                                                static_cast<InnoItemType>(pkt.type), pt.firing);
            if (force_vehicle_coordinate) {
              x = xyzr.z;
              y = -xyzr.y;
              z = xyzr.x;
            } else {
              x = xyzr.x;
              y = xyzr.y;
              z = xyzr.z;
            }
            h_angle = static_cast<double>(full_angles.angles[channel].h_angle) / kInnoAngleUnitPerDegree;
            v_angle = static_cast<double>(full_angles.angles[channel].v_angle) / kInnoAngleUnitPerDegree;

            if (transform_cs_) {
              sensor_to_world_cs(&x, &y, &z, xyzr.x, xyzr.y, xyzr.z, roll_, pitch_, yaw_);
            }

            if (recorder_point_type == POINT_TYPE_EN_PCD) {
              recorder->add_en_points(pkt.idx, x, y, z, h_angle, v_angle, pt.refl, channel, block->header.in_roi,
                                      block->header.facet, m, pkt.confidence_level, pt.firing << 2, 0,
                                      frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id,
                                      block->header.scan_idx, pt.is_2nd_return, scan_id);
            } else if (recorder_point_type == POINT_TYPE_LITE_PCD) {
              recorder->add_lite_points(x, y, z, pt.refl, pkt.confidence_level,
                                        frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id);
            } else {
              recorder->add_points(pkt.idx, x, y, z, pt.refl, channel, block->header.in_roi, block->header.facet, m,
                                   pkt.confidence_level, pt.firing << 2, 0,
                                   frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, scan_id,
                                   block->header.scan_idx, pt.is_2nd_return, scan_id);
            }
          }
        }
      }
    }
  }
  // this function use to convert falcon point to xyz point and recorder
  void block_to_xyz_point(const InnoDataPacket &pkt, FileRecorder *recorder, int unit_size, int return_number) {
    const InnoBlock *blocks = reinterpret_cast<const InnoBlock *>(pkt.payload);
    double frame_timestamp_sec = pkt.common.ts_start_us / kUsInSecond;
    const InnoBlock *block = reinterpret_cast<const InnoBlock *>(&blocks[0]);

    for (uint32_t i = 0; i < pkt.item_number;
         i++, block = reinterpret_cast<const InnoBlock *>(reinterpret_cast<const char *>(block) + unit_size)) {
      // calculate (x,y,z) cartesian coordinate from spherical coordinate
      // for each point in the block
      // 1. use get_full_angles() to restore angle for each channel
      // 2. use get_xyzr_meter() to calculate (x,y,z)
      InnoBlockFullAngles full_angles;
      InnoDataPacketUtils::get_block_full_angles(&full_angles, block->header, static_cast<InnoItemType>(pkt.type));
      double x, y, z, h_angle, v_angle;

      for (uint32_t channel = 0; channel < kInnoChannelNumber; channel++) {
        for (uint32_t m = 0; m < return_number; m++) {
          const InnoChannelPoint &pt = block->points[innoblock_get_idx(channel, m)];
          InnoXyzrD xyzr;

          if (pt.radius > 0) {
            InnoDataPacketUtils::get_xyzr_meter(full_angles.angles[channel], pt.radius, channel, &xyzr,
                                                static_cast<InnoItemType>(pkt.type), 1, pkt.long_distance_mode);
            if (force_vehicle_coordinate) {
              x = xyzr.z;
              y = -xyzr.y;
              z = xyzr.x;
            } else {
              x = xyzr.x;
              y = xyzr.y;
              z = xyzr.z;
            }
            h_angle = static_cast<double>(full_angles.angles[channel].h_angle) / kInnoAngleUnitPerDegree;
            v_angle = static_cast<double>(full_angles.angles[channel].v_angle) / kInnoAngleUnitPerDegree;

            if (transform_cs_) {
              sensor_to_world_cs(&x, &y, &z, xyzr.x, xyzr.y, xyzr.z, roll_, pitch_, yaw_);
            }

            int ring_id = inno_lidar_get_ring_id(lidar_handle_, static_cast<InnoLidarMode>(pkt.common.lidar_mode),
                                                 pkt.scanner_direction, block->header.scan_id, channel);
            if (recorder_point_type == POINT_TYPE_EN_PCD) {
              recorder->add_en_points(
                  pkt.idx, x, y, z, h_angle, v_angle, pt.refl, channel, block->header.in_roi, block->header.facet,
                  m, pkt.confidence_level, pt.type, pt.elongation,
                  frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, block->header.scan_id,
                  block->header.scan_idx, pt.is_2nd_return, ring_id);
            } else if (recorder_point_type == POINT_TYPE_LITE_PCD) {
              recorder->add_lite_points(x, y, z, pt.refl, pkt.confidence_level,
                                        frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, ring_id);
            } else {
              recorder->add_points(pkt.idx, x, y, z, pt.refl, channel, block->header.in_roi, block->header.facet, m,
                                 pkt.confidence_level, pt.type, pt.elongation,
                                 frame_timestamp_sec + block->header.ts_10us / k10UsInSecond, block->header.scan_id,
                                 block->header.scan_idx, pt.is_2nd_return, ring_id);
            }
          }
        }
      }
    }
  }
  /*
    this function shows how to enumerate blocks and points in the packet
    directly, and use InnoDataPacketUtils::get_xyzr_meter to get each
    point's x,y,z coodindate, then add it to the file recorder
  */
  void process_data_cpoint_(const InnoDataPacket &pkt) {
    uint32_t return_number;
    uint32_t unit_size;
    FileRecorder *recorder = get_recorder_(pkt.idx, pkt.use_reflectance);

    if (!recorder) {
      return;
    }

    if (recorder->can_record_bag()) {
      recorder->add_data_packet_to_bag(pkt);
      return;
    } else if (recorder->can_record_png()) {
      recorder->add_data_packet_to_png(pkt);
      return;
    } else if (recorder->can_record_packet()) {
      recorder->add_data_packet(pkt);
      return;
    }

    InnoDataPacketUtils::get_block_size_and_number_return(pkt, &unit_size, &return_number);

    if (pkt.type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      block_to_xyz_point(pkt, recorder, unit_size, return_number);
    } else if (pkt.type == INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD ||
               pkt.type == INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD) {
      co_block_to_en_xyz_point(pkt, recorder, unit_size, return_number);
    } else {
      en_block_to_en_xyz_point(pkt, recorder, unit_size, return_number);
    }

    return;
  }

  void start_check_datacallback_thread() {
    std::thread([&]() {
      uint8_t checkout_count = 10;
      while (!done_) {
        for (auto i = 0; i < checkout_count; i++) {
          if (receive_datacallback_.load()) {
            receive_datacallback_ = false;
            break;
          } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (i == checkout_count - 1) {
              done_ = true;
              inno_log_error("datacallback function did not receive data for more than 10s, app exiting");
              return;
            }
          }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }).detach();
  }

  ExampleProcessor() = delete;
  ExampleProcessor(const ExampleProcessor &) = delete;
  ExampleProcessor operator=(const ExampleProcessor &) = delete;

 private:
  std::string filename_;
  std::string filename_base_;
  int64_t frame_start_;
  int64_t frame_number_;
  int64_t file_number_;
  int use_xyz_point_;
  std::string latency_file_;
  std::fstream *latency_recorder_{NULL};
  int64_t current_frame_;
  int64_t frame_so_far_;
  int file_so_far_;
  FileRecorder *file_recorder_;
  std::string msg_filename_;
  std::string status_filename_;
  std::string galvo_check_filename_;
  std::string max_distance_filename_;
  int status_fd_;
  int msg_fd_;
  int galvo_check_fd_;
  int max_distance_fd_;
  char msg_buffer_[kMaxMsgBuf];
  char status_buffer_[kMaxMsgBuf];
  char galvo_check_buffer_[kMaxMsgBuf];
  char max_distance_buffer_[kMaxMsgBuf];
  enum FileRecorder::FileType file_type_;
  std::string file_extension_;
  bool done_;
  bool transform_cs_;
  double yaw_;
  double pitch_;
  double roll_;
  uint64_t run_time_;
  InnoSummaryPackage summary_package_;
  int need_process_data_ = 1;
  std::atomic_bool receive_datacallback_{false};
  int lidar_handle_ = -1;
  char *anglehv_table_ = nullptr;
  RingIdConverterInterface *ring_id_converter_{nullptr};
};

/***********************
 * usage()
 ***********************/
void usage(const char *arg0) {
  inno_fprintf(2,
               "Usage:\n"
               "   %s\n"
               "\t{[--lidar-ip <LIDAR_IP>]\n"
               "\t\t[--lidar-port <LIDAR_PORT>]\n"
               "\t\t[--lidar-udp-port <LIDAR_UDP_PORT>]\n"
               "\t\t[--use-tcp]\n"
               "\t\t[--lidar-mode <LIDAR_MODE 3=normal/5=calibration>]\n"
               "\t\t[--reflectance <REFLECTANCE_MODE "
               "1=intensity/2=reflectance>]\n"
               "\t\t[--multireturn <MULTI_RETURN_MODE "
               "1=single/2=two-strongest/3=strongest+furthest>]\n"
               "\t\t[--falcon-eye <HORIZONTAL_DEGREE -60 to 60>,"
               "<VERTICAL_DEGREE -13 to 13>] |\n"
               "\t  [--inno-pc-filename <INPUT_INNO_PC_FILENAME>]}\n"
               "\t[--use-xyz-point <1,2>]\n"
               "\t[--roll <ROLL_IN_DEGREES>]\n"
               "\t[--pitch <PITCH_IN_DEGREES>]\n"
               "\t[--yaw <YAW_IN_DEGREES>]\n"
               "\t[--file-number <NUMBER_OF_FILE>]\n"
               "\t[--frame-start <Nth_FRAME_TO_RECORD>]\n"
               "\t[--frame-number <NUMBER_OF_FRAME_TO_RECORD>]\n"
               "\t[--output-filename <OUTPUT_FILENAME."
               "pcd|csv|inno_pc|inno_pc_xyz|inno_cframe|bag|png>]\n"
               "\t[--output-filename-with-timestamp <OUTPUT_FILENAME."
               "pcd|csv|inno_pc|inno_pc_xyz|inno_cframe|bag|png>]\n"
               "\t[--ascii-pcd]\n"
               "\t[--extract-message]\n"
               "\t[--no-process-data]\n"
               "\t[--latency-file] <Save one latency per frame>\n"
               "\t[--table <input anglehvtable file>]\n"
               "\t[--run-time] <get_pcd run time,unit S>\n"
               "\t[--point-type <0,1,2> level of the pcd include element information\n"
               "\t\t0 = Lite PCD (XYZICRT:Contains X,Y,Z,Intensity,Confidence,Ring ID,and Timestamp)\n"
               "\t\t1 = Normal PCD (Includes additional information such as scan_id and Scan_idx)\n"
               "\t\t2 = Enhanced PCD (Includes all Normal PCD features plus Horizontal and Vertical Angle)]\n"
               "\t[--version]\n",
               arg0);
  inno_fprintf(2,
               "\n"
               "Examples:\n"
               " --record 2 frames from live LIDAR to test.pcd (LIDAR "
               "is configured to multicast or broadcast).\n"
               "   %s --frame-number 2 --output-filename test.pcd\n\n"
               " --record 100 frames from live LIDAR via UDP "
               "to test.inno_pc_xyz (LIDAR "
               "is NOT configured to multicast or broadcast).\n"
               "   %s --lidar-udp-port 8010 "
               "--frame-number 100 --output-filename test.inno_pc_xyz\n\n"
               " --record 100 frames from live LIDAR via TCP "
               "to test.inno_pc (LIDAR "
               "is NOT configured to multicast or broadcast).\n"
               "   %s --lidar-udp-port 0 "
               "--frame-number 100 --output-filename test.inno_pc\n\n"
               " --record 100 frames from pcap file to test.pcd\n"
               "   %s --pcap-filename input.pcap --frame-number 10"
               " --lidar-udp-port 8010"
               " --output-filename-with-timestamp test.pcd\n\n"
               " --record 1 frames per pcd, start from frame #10, "
               "for 20 frames "
               "from input.inno_pc file to test-xx.pcd\n"
               "   %s --inno-pc-filename input.inno_pc "
               "--frame-start 10 --frame-number 1 "
               "--file-number 20 "
               "--output-filename test.pcd\n\n"
               "Please see more usage examples in test_get_pcd.bash\n",
               arg0, arg0, arg0, arg0, arg0);
  return;
}

void print_api() {
  std::string api_version = std::string(INNO_SDK_VERSION_IN_HEADER) + "0";
  inno_log_info("version %s \n", api_version.c_str());
}

/***********************
 * main()
 ***********************/
int main(int argc, char **argv) {
  if (argc == 1) {
    usage(argv[0]);
    exit(0);
  }

  /***********************
   * parse command line
   ***********************/
  std::string inno_pc_filename;
  std::string pcap_filename;
  std::string lidar_ip = "172.168.1.10";
  uint16_t lidar_port = 8010;
  uint16_t lidar_udp_port = 0;
  std::string filename;
  int64_t frame_start = -1;
  int64_t frame_number = 1;
  int64_t file_number = 1;
  int use_xyz_point = 1;
  std::string latency_file = "";
  int extract_message = 0;
  int use_tcp = 0;
  int ascii_pcd = 0;
  int vehicle_speed = 0;
  int max_distance = 0;
  enum InnoLidarMode lidar_mode = INNO_LIDAR_MODE_NONE;
  enum InnoReflectanceMode reflectance = INNO_REFLECTANCE_MODE_NONE;
  enum InnoMultipleReturnMode multireturn = INNO_MULTIPLE_RETURN_MODE_NONE;
  double roi_center_h = 0;
  double roi_center_v = 0;
  bool roi_set = false;
  int simulation_galvo_check = 0;
  double roll = 0;
  double pitch = 0;
  double yaw = 0;
  bool transform_cs = false;
  bool use_ring_id_convert = false;
  // If just statistics, it doesn't need to process data
  int need_process_data = 1;
  uint64_t run_time = 0;
  std::string anghv_table_file;
  /* getopt_long stores the option index here. */
  int c;
  struct option long_options[] = {/* These options set a flag. */
                                  {"lidar-ip", required_argument, 0, 'n'},
                                  {"lidar-port", required_argument, 0, 'p'},
                                  {"lidar-udp-port", required_argument, 0, 'O'},
                                  {"frame-start", required_argument, 0, 's'},
                                  {"frame-number", required_argument, 0, 'c'},
                                  {"file-number", required_argument, 0, 'N'},
                                  {"output-filename", required_argument, 0, 'P'},
                                  {"output-filename-with-timestamp", required_argument, 0, 'W'},
                                  {"inno-pc-filename", required_argument, 0, 'f'},
                                  {"pcap-filename", required_argument, 0, 'C'},
                                  {"falcon-eye", required_argument, 0, 'e'},
                                  {"latency-file", required_argument, 0, 'l'},
                                  {"lidar-mode", required_argument, 0, 'm'},
                                  {"reflectance", required_argument, 0, 'F'},
                                  {"en-pcd", no_argument, 0, 'E'},
                                  {"multireturn", required_argument, 0, 'M'},
                                  {"use-xyz-point", required_argument, 0, 'x'},
                                  {"galvo-check", required_argument, 0, 'g'},
                                  {"vehicle-speed", required_argument, 0, 'v'},
                                  {"roll", required_argument, 0, 'r'},
                                  {"pitch", required_argument, 0, 't'},
                                  {"yaw", required_argument, 0, 'y'},
                                  {"extract-message", no_argument, &extract_message, 1},
                                  {"use-tcp", no_argument, &use_tcp, 1},
                                  {"ascii-pcd", no_argument, &ascii_pcd, 1},
                                  {"max-distance", no_argument, &max_distance, 1},
                                  {"use-ring-id-convert", no_argument, NULL, 'R'},
                                  {"run-time", required_argument, 0, 'T'},
                                  {"no-process-data", no_argument, &need_process_data, 0},
                                  {"table", required_argument, 0, 'A'},
                                  {"help", no_argument, NULL, 'h'},
                                  {"version", no_argument, NULL, 'V'},
                                  {"force-vehicle-coordinate", no_argument, &force_vehicle_coordinate, 1},
                                  {"point-type", required_argument, 0, 'S'},
                                  {0, 0, 0, 0}};
  const char *optstring = "ac:e:f:g:hl:m:n:p:r:s:t:x:v:y:F:M:N:O:P:C:V:W:S";
  while (1) {
    int option_index = 0;
    c = getopt_long(argc, argv, optstring, long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0) {
          break;
        }
        inno_log_verify(optarg == NULL, "option %s with arg %s", long_options[option_index].name, optarg);
        break;

      case 'a':
        ascii_pcd = 1;
        break;
      case 'V':
        print_api();
        exit(0);
        break;

      case 'h':
        usage(argv[0]);
        exit(0);
        break;

      case 'P':
        filename = optarg;
        break;

      case 'W':
        filename = optarg;
        filename_with_timestamp = true;
        break;

      case 'f':
        inno_pc_filename = optarg;
        break;

       case 'C':
        pcap_filename = optarg;
        break;

      case 'n':
        lidar_ip = optarg;
        if (!innovusion::InnoUtils::check_ip_valid(lidar_ip.c_str())) {
          inno_log_error("invalid --lidar_ip %s", optarg);
          exit(1);
        }
        break;

      case 's':
        frame_start = atoll(optarg);
        break;

      case 'c':
        frame_number = atoll(optarg);
        if (frame_number <= 0) {
          frame_number = std::numeric_limits<int>::max();
        }
        break;

      case 'F':
        reflectance = (enum InnoReflectanceMode)atoi(optarg);
        if (reflectance != INNO_REFLECTANCE_MODE_INTENSITY && reflectance != INNO_REFLECTANCE_MODE_REFLECTIVITY) {
          usage(argv[0]);
          exit(1);
        }
        break;

      case 'M':
        multireturn = (enum InnoMultipleReturnMode)atoi(optarg);
        if (multireturn != INNO_MULTIPLE_RETURN_MODE_SINGLE && multireturn != INNO_MULTIPLE_RETURN_MODE_2_STRONGEST &&
            multireturn != INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
          usage(argv[0]);
          exit(1);
        }
        break;

      case 'm':
        lidar_mode = (enum InnoLidarMode)atoi(optarg);
        if (lidar_mode != INNO_LIDAR_MODE_WORK_NORMAL && lidar_mode != INNO_LIDAR_MODE_WORK_CALIBRATION &&
            lidar_mode != INNO_LIDAR_MODE_WORK_EXHIBITION) {
          usage(argv[0]);
          exit(1);
        }
        break;

      case 'N':
        file_number = atoll(optarg);
        if (file_number <= 0) {
          file_number = std::numeric_limits<int>::max();
        }
        break;

      case 'e':
        if (sscanf(optarg, "%lf,%lf", &roi_center_h, &roi_center_v) == 2) {
          roi_set = true;
        } else {
          inno_log_error("invalid --falcon-eye option %s", optarg);
          exit(1);
        }
        break;

      case 'p':
        lidar_port = atoi(optarg);
        break;

      case 'O':
        lidar_udp_port = strtoul(optarg, NULL, 0);
        break;

      case 'x':
        use_xyz_point = atoi(optarg);
        break;

      case 'l':
        latency_file = optarg;
        break;

      case 'g':
        simulation_galvo_check = atoi(optarg);
        break;

      case 'v':
        vehicle_speed = atoi(optarg);
        break;

      case 'r':
        if (sscanf(optarg, "%lf", &roll) == 1) {
          if (roll != 0) {
            transform_cs = true;
          }
        } else {
          inno_log_error("invalid --roll option %s", optarg);
          exit(1);
        }
        break;

      case 't':
        if (sscanf(optarg, "%lf", &pitch) == 1) {
          if (pitch != 0) {
            transform_cs = true;
          }
        } else {
          inno_log_error("invalid --pitch option %s", optarg);
          exit(1);
        }
        break;

      case 'y':
        if (sscanf(optarg, "%lf", &yaw) == 1) {
          if (yaw != 0) {
            transform_cs = true;
          }
        } else {
          inno_log_error("invalid --yaw option %s", optarg);
          exit(1);
        }
        break;

      case 'R':
        use_ring_id_convert = true;
        break;

      case '?':
        abort();

      case 'T':
        run_time = strtoul(optarg, NULL, 0);
        break;
      case 'E':
        recorder_point_type = POINT_TYPE_EN_PCD;
         break;
      case 'S':
        recorder_point_type = atoi(optarg);
        break;
      case 'A':
        anghv_table_file = optarg;
        break;
      default:
        inno_log_error("unknown options %c\n", c);
        usage(argv[0]);
        exit(1);
    }
  }

  int ret;
#if !defined(__MINGW64__) && defined(_WIN32)
  ret = innovusion::InnoUtils::print_app_ver();
  if (ret != 0) {
    exit(ret);
  }
#else
  print_api();
#endif

  /***********************
   * set debug level
   ***********************/
  enum InnoLogLevel debug_level = INNO_LOG_LEVEL_INFO;
  inno_lidar_set_log_level(debug_level);

  if (frame_start == -1) {
    if (inno_pc_filename.size() > 0 || pcap_filename.size() > 0) {
      frame_start = 0;
    } else {
      frame_start = 1;
    }
  }


  /***********************
   * open lidar handle
   ***********************/
  int handle;
  if (inno_pc_filename.size() > 0) {
    handle = inno_lidar_open_file("file",                          /* name of lidar instance */
                                  inno_pc_filename.c_str(), false, /* inno_pc format                        */
                                  0,                               /* as fast as possible without drop data */
                                  0,                               /* play once, no rewind                  */
                                  0);                              /* no skip                               */
  } else if (pcap_filename.size() > 0) {
    InputParam param;
    memset(&param, 0, sizeof(param));
    param.pcap_param.source_type = SOURCE_PCAP;
    inno_log_verify(pcap_filename.length() < sizeof(param.pcap_param.filename),
    "%s length > %" PRI_SIZET, pcap_filename.c_str(), sizeof(param.pcap_param.filename));
    strncpy(param.pcap_param.filename, pcap_filename.c_str(), pcap_filename.length() + 1);
    param.pcap_param.rewind = 0;
    param.pcap_param.data_port = lidar_udp_port;
    param.pcap_param.message_port = lidar_udp_port;
    param.pcap_param.status_port = lidar_udp_port;
    handle = inno_lidar_open_ctx("pcap", &param);
  } else {
    handle = inno_lidar_open_live("live", /* name of lidar instance */
                                  lidar_ip.c_str(), lidar_port,
                                  use_tcp ? INNO_LIDAR_PROTOCOL_PCS_TCP : INNO_LIDAR_PROTOCOL_PCS_UDP, lidar_udp_port);
  }
  inno_log_verify(handle > 0, "cannot open lidar");
  g_lidar_handle = handle;
  /***********************
   * create ExampleProcessor object
   ***********************/
  ExampleProcessor processor(filename, frame_start, frame_number, file_number, use_xyz_point, latency_file, ascii_pcd,
                             extract_message, run_time, handle, anghv_table_file);
  processor.set_need_process_data(need_process_data);

  if (transform_cs) {
    processor.set_orientation(yaw, pitch, roll);
  }

  // the code shows 2 ways to process the data callback
 if (use_xyz_point == 1) {
    // callback with INNO_ITEM_TYPE_SPHERE_POINTCLOUD and
    // covert to INNO_ITEM_TYPE_XYZ_POINTCLOUD
    // then enumerate each xyz points
  } else if (use_xyz_point == 2) {
    // set SDK to callback with INNO_ITEM_TYPE_XYZ_POINTCLOUD and
    // then enumerate each xyz points
    inno_lidar_set_attribute_string(handle, "force_xyz_pointcloud", "1");
    // force to record normal pcd
    recorder_point_type = POINT_TYPE_PCD;
  } else {
    inno_log_verify(false,
                    "invalid use_xyz_point number, "
                    "must in [0, 1, 2]");
  }
  if(force_vehicle_coordinate) {
    inno_lidar_set_attribute_string(handle, "force_vehicle_coordinate", "1");
  }
  /***********************
   * set lidar reflectance/multi-return/roi/work mode
   ***********************/
  ret = inno_lidar_set_config_name_value(handle, "LidarClient_StageClientRead/misorder_correct_enable", "1");
  inno_log_verify(ret == 0,
                  "inno_lidar_set_config_name_value"
                  "'misorder_correct_enable 1' failed %d",
                  ret);

  if (inno_pc_filename.size() == 0 && pcap_filename.size() == 0) {
    ret = inno_lidar_set_config_name_value(handle, "LidarClient_Communication/get_conn_timeout_sec", "5.0");
    inno_log_verify(ret == 0,
                    "inno_lidar_set_config_name_value"
                    "'get_conn_timeout_sec 5.0' failed %d",
                    ret);
  }

  if (reflectance != INNO_REFLECTANCE_MODE_NONE) {
    ret = inno_lidar_set_reflectance_mode(handle, reflectance);
    inno_log_verify(ret == 0, "set_reflectance failed %d", ret);
  }

  if (multireturn != INNO_MULTIPLE_RETURN_MODE_NONE) {
    ret = inno_lidar_set_return_mode(handle, multireturn);
    inno_log_verify(ret == 0, "set_return_mode failed %d", ret);
  }

  if (roi_set) {
    ret = inno_lidar_set_roi(handle, roi_center_h, roi_center_v);
    inno_log_verify(ret == 0, "set_roi failed %d", ret);
  }

  inno_log_info("simulation_galvo_check=%d, max_distance=%d, vehicle_speed=%d", simulation_galvo_check, max_distance,
                vehicle_speed);
  if (simulation_galvo_check) {
    ret = inno_lidar_set_attribute_string(handle, "vehicle_speed", "85.0");
    inno_log_verify(ret == 0, "set vehicle_speed failed %d", ret);
    ret = inno_lidar_set_attribute_string(handle, "galvo_ext_ref", "-2.241,0.0094,-2.201,0,0,0");
    inno_log_verify(ret == 0, "set galvo_ext_ref failed %d", ret);
    processor.create_galvo_check_result_file();
  }

  if (max_distance) {
    processor.create_max_distance_result_file();
  }

  if (lidar_mode != INNO_LIDAR_MODE_NONE) {
    enum InnoLidarMode pre_mode;
    enum InnoLidarStatus status;
    ret = inno_lidar_set_mode(handle, lidar_mode, &pre_mode, &status);
    inno_log_verify(ret == 0, "set_mode failed %d", ret);
  }

  /***********************
   * set lidar pointcloud processing callbacks
   ***********************/
  ret = inno_lidar_set_callbacks(handle, processor.message_callback_s, processor.data_callback_s,
                                 processor.status_callback_s,
                                 NULL,  // use default get_host_time()
                                 &processor);
  inno_log_verify(ret == 0, "set_callbacks failed %d", ret);

  /***********************
   * start lidar pointcloud processing threads
   ***********************/
  ret = inno_lidar_start(handle);
  inno_log_verify(ret == 0, "start failed %d", ret);

  if (use_ring_id_convert) {
    ret = inno_lidar_set_attribute_string(handle, "use_ring_id", "1");
    if (ret != 0) {
      inno_log_warning("get ring id converter failed. Please ignore this if you needn't ring id.");
    } else {
      void *converter = inno_lidar_get_ring_id_converter(handle);
      if (converter) {
        processor.set_ring_id_converter(reinterpret_cast<RingIdConverterInterface *>(converter));
      }
    }
  }
  /***********************
   * wait until processor is done
   ***********************/
  while (!processor.is_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (max_distance) {
      inno_lidar_set_attribute_string(handle, "vehicle_speed", std::to_string(vehicle_speed).c_str());
    }
  }

  /***********************
   * stop lidar pointcloud processing threads
   ***********************/
  ret = inno_lidar_stop(handle);
  inno_log_verify(ret == 0, "stop failed %d", ret);

  /***********************
   * close lidar handle
   ***********************/
  ret = inno_lidar_close(handle);
  inno_log_verify(ret == 0, "close failed %d", ret);

  return 0;
}
