/*
 *  Copyright (C) 2021 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_INNO_LIDAR_PACKET_V1_ADAPT_INTERNAL_H_
#define SDK_CLIENT_INNO_LIDAR_PACKET_V1_ADAPT_INTERNAL_H_

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sdk_common/inno_faults_common.h"
#include "sdk_common/inno_lidar_packet.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/inno_lidar_log.h"
/*
 * Main data packet definition
 *
 * 26 + 12 + 10 + 2 + 4 = 54 bytes, max overhead is 54/1472 = 3.9%,
 *
 * for single-return (33 bytes), 1472 udp payload = 42 blocks,
 * 1.4M points/seconds uses BW = 1.4M / (44 * 4) * 1500 = 11.93MB/s
 *
 * for double-return (49 bytes), 1472 udp payload = 28 blocks
 * 1.4M points/seconds uses BW = 1.4M / (30 * 4) * 1500 = 17.50MB/s
 */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoDataPacketV1) {
  InnoCommonHeader common;

  /* 12 bytes */
  uint64_t idx;     /* frame index, start from 0                     */
  uint16_t sub_idx; /* sub-frame index, start from 0 for every frame */
  uint16_t sub_seq; /* sequence of InnoDataPacket, start from 0     */

  /* 10 byte */
  /* type in enum InnoItemType, each type uses independent global idx */
  uint32_t type : 8;
  uint32_t item_number : 24; /* max 4 * 1024 * 1024               */
  uint16_t item_size;        /* max 65535, 0 means variable size  */
  uint32_t topic;            /* reserved                          */

  /* 2 bytes */
  uint16_t scanner_direction : 1;  /* 0: top->bottom, 1: bottom->top          */
  uint16_t use_reflectance : 1;    /* 0: intensity mode, 1: reflectance mode  */
  uint16_t multi_return_mode : 3;  /* ... */
  uint16_t confidence_level : 2;   /* 0: no confidence, 3: higest             */
  uint16_t is_last_sub_frame : 1;  /* 1: the last sub frame of a frame        */
  uint16_t is_last_sequence : 1;   /* 1: the last piece of a sub frame        */
  uint16_t has_tail : 1;           /* has additional tail struct after points */
  uint16_t frame_sync_locked : 1;  /* 1: frame sync has locked                */
  uint16_t is_first_sub_frame : 1; /* 1: the first sub frame of a frame      */
  uint16_t last_four_channel : 1;
  uint16_t long_distance_mode : 1; /* lidar work in long distance mode,only for falconk */
  uint16_t reserved_flag : 2;      /* all 0 */

  /* 4 bytes */
  int16_t roi_h_angle; /* configured ROI in InnoAngleUnit */
  int16_t roi_v_angle;
// MSVC compiler does not support multi-dimensional flexible arrays.
#if !defined(_MSC_VER)
  union {
    char payload[0];
    InnoBlock1 inno_block1s[0];
    InnoBlock2 inno_block2s[0];
    InnoMessage messages[0];
    InnoXyzPoint xyz_points[0];
  };
#else
  char payload[0];
#endif
}
InnoDataPacketV1;
DEFINE_INNO_COMPACT_STRUCT_END

/*
 * Status packet definition
 */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusPacketV1) {
  InnoCommonHeader common;

  uint64_t idx; /* global index of all InnoStatusPacket */

  uint8_t status_packet_interval_ms; /* status packet send interval in ms    */
  uint8_t pre_lidar_mode;            /* previous InnoLidarMode               */
  uint16_t in_transition_mode_ms;    /* time (ms), LiDAR in the transition mode */

  char sn[INNO_SN_SZIE]; /* lidar serial number */
  uint16_t fault_version;
  uint16_t ref_count_enough_ts_ms;
  uint16_t ref_intensity[INNO_CHANNEL_NUMBER];
  uint32_t reserved;

  InnoStatusInFaults in_faults;
  InnoStatusExFaults ex_faults;
  InnoStatusCounters counters;
  InnoStatusSensorReadings sensor_readings;
}
InnoStatusPacketV1;
DEFINE_INNO_COMPACT_STRUCT_END

class InnoPacketV1Adapt {
 public:
  static const uint32_t kMemorryFrontGap = sizeof(InnoDataPacket) - sizeof(InnoDataPacketV1);
  static const uint32_t kInnoProtocolMajorV1 = 1;
  static const uint32_t kInnoProtocolMajorV2 = 2;
  static const uint32_t kInnoProtocolMinorV2 = 1;

 public:
  static inline size_t get_data_packet_v1_size(InnoItemType type, uint32_t item_count, InnoMultipleReturnMode mode) {
    size_t unit_size;
    if (type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      if (mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        unit_size = sizeof(InnoBlock1);
      } else if (mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        unit_size = sizeof(InnoBlock2);
      } else {
        return 0;
      }
    } else if (type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
      unit_size = sizeof(InnoXyzPoint);
    } else {
      inno_log_verify(false, "bad type = %d", type);
    }
    return sizeof(InnoDataPacketV1) + item_count * unit_size;
  }

  /**
   * @brief Sanity check the integrity of a InnoDataPacketV1 and convert to InnoDataPacket.
   * @param buff in: InnoDataPacketV1  out:InnoDataPacket
   * @param size Size of pkt if it is received from network or
   *        read from file
   * @param gap buff front gap size
   * @return false if the pkt is invalid, true otherwise
   */
  static bool check_data_packet_v1_and_convert_packet(char **buff, size_t size, int &gap) {
    InnoDataPacketV1 *pkt = reinterpret_cast<InnoDataPacketV1 *>(*buff);
    if (pkt->common.version.magic_number != kInnoMagicNumberDataPacket) {
      inno_log_warning("bad magic %x", pkt->common.version.magic_number);
      return false;
    }
    if (size && pkt->common.size > size) {
      inno_log_warning("bad size %" PRI_SIZELU " %u", size, pkt->common.size);
      return false;
    }
    if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD) {
      if (pkt->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_SINGLE) {
        if (pkt->item_size != sizeof(InnoBlock1)) {
          inno_log_warning("bad block1 item size %u, sizeof(InnoBlock1): %" PRI_SIZELU, pkt->item_size,
                           sizeof(InnoBlock1));
          return false;
        }
      } else if (pkt->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST ||
                 pkt->multi_return_mode == INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST) {
        if (pkt->item_size != sizeof(InnoBlock2)) {
          inno_log_warning("bad block2 item size %u", pkt->item_size);
          return false;
        }
      } else {
        inno_log_warning("bad return_mode %u", pkt->multi_return_mode);
        return false;
      }
    } else if (pkt->type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
      if (pkt->item_size != sizeof(InnoXyzPoint)) {
        inno_log_warning("bad InnoXyzPoint item size %u", pkt->item_size);
        return false;
      }
    }

    if (pkt->type == INNO_ITEM_TYPE_SPHERE_POINTCLOUD || pkt->type == INNO_ITEM_TYPE_XYZ_POINTCLOUD) {
      size_t s = get_data_packet_v1_size(InnoItemType(pkt->type), pkt->item_number,
                                         InnoMultipleReturnMode(pkt->multi_return_mode));
      if (pkt->common.size != s) {
        inno_log_warning("bad size %" PRI_SIZELU " %u", s, pkt->common.size);
        return false;
      }
      if (!InnoPacketReader::verify_packet_crc32(&pkt->common)) {
        inno_log_warning("crc32 mismatch for data packet");
        return false;
      }
      if (gap == kMemorryFrontGap) {
        *buff = *buff - kMemorryFrontGap;
        memmove(*buff, pkt, sizeof(InnoDataPacketV1));
        InnoDataPacket *pkt1 = reinterpret_cast<InnoDataPacket *>(*buff);
        // inno_log_info("convert v1-> InnoDataPacket");
        pkt1->common.size += kMemorryFrontGap;
        memset(pkt1->extend_reserved, 0, sizeof(pkt1->extend_reserved));
        pkt1->common.version.major_version = kInnoProtocolMajorV2;
        pkt1->common.version.minor_version = kInnoProtocolMinorV2;
        InnoPacketReader::set_packet_crc32(&pkt1->common);
      } else if (gap == 0) {
        gap = kMemorryFrontGap;
        return false;
      }
      return true;
    } else if (pkt->type == INNO_ITEM_TYPE_MESSAGE || pkt->type == INNO_ITEM_TYPE_MESSAGE_LOG) {
      if (pkt->item_number != 1) {
        inno_log_warning("bad item_number %u", pkt->item_number);
        return false;
      }
      // char *p_addr = reinterpret_cast<char *>(const_cast<InnoDataPacketV1 *>(&pkt));
      InnoMessage *messages = reinterpret_cast<InnoMessage *>(pkt->payload);
      if (static_cast<uint32_t>(pkt->item_size) != messages[0].size || pkt->item_size <= sizeof(InnoMessage)) {
        inno_log_warning("bad message size %u %u", pkt->item_size, messages[0].size);
        return false;
      }
      if (pkt->common.size != pkt->item_size + sizeof(InnoDataPacketV1)) {
        inno_log_warning("bad message size %u, %" PRI_SIZELU, pkt->common.size,
                         pkt->item_size + sizeof(InnoDataPacketV1));
        return false;
      }
      if (!InnoPacketReader::verify_packet_crc32(&pkt->common)) {
        inno_log_warning("crc32 mismatch for message packet");
        return false;
      }
      if (gap == kMemorryFrontGap) {
        *buff = *buff - kMemorryFrontGap;
        memmove(*buff, pkt, sizeof(InnoDataPacketV1));
        InnoDataPacket *pkt1 = reinterpret_cast<InnoDataPacket *>(*buff);
        // inno_log_info("convert v1-> InnoDataPacket");
        pkt1->common.size += kMemorryFrontGap;
        memset(pkt1->extend_reserved, 0, sizeof(pkt1->extend_reserved));
        pkt1->common.version.major_version = kInnoProtocolMajorV2;
        pkt1->common.version.minor_version = kInnoProtocolMinorV2;
        InnoPacketReader::set_packet_crc32(&pkt1->common);
      } else if (gap == 0) {
        gap = kMemorryFrontGap;
        return false;
      }
      return true;
    } else if (pkt->type == INNO_FALCON_RING_ID_TABLE) {
      if (pkt->item_number != 1) {
        inno_log_warning("bad item_number %u", pkt->item_number);
        return false;
      }
      if (pkt->common.size != pkt->item_size + sizeof(InnoDataPacketV1)) {
        inno_log_warning("bad ring id table size %u, %" PRI_SIZELU, pkt->common.size,
                         pkt->item_size + sizeof(InnoDataPacketV1));
        return false;
      }
      if (!InnoPacketReader::verify_packet_crc32(&pkt->common)) {
        inno_log_warning("crc32 mismatch for message packet");
        return false;
      }
      if (gap == kMemorryFrontGap) {
        *buff = *buff - kMemorryFrontGap;
        memmove(*buff, pkt, sizeof(InnoDataPacketV1));
        InnoDataPacket *pkt1 = reinterpret_cast<InnoDataPacket *>(*buff);
        // inno_log_info("convert v1-> InnoDataPacket");
        pkt1->common.size += kMemorryFrontGap;
        memset(pkt1->extend_reserved, 0, sizeof(pkt1->extend_reserved));
        pkt1->common.version.major_version = kInnoProtocolMajorV2;
        pkt1->common.version.minor_version = kInnoProtocolMinorV2;
        InnoPacketReader::set_packet_crc32(&pkt1->common);
      } else if (gap == 0) {
        gap = kMemorryFrontGap;
        return false;
      }
      return true;
    } else {
      inno_log_warning("bad type %u", pkt->type);
      return false;
    }
  }

  /**
   * @brief Sanity check the InnoStatusPacketV1 and convert to InnoDataPacket.
   * @param pkt in: InnoDataPacketV1  out:InnoDataPacket
   * @param size Size of pkt if it is received from network or
   *        read from file
   * @param gap  buff front gap size
   * @return false if the pkt is invalid, true otherwise
   */
  static bool check_status_packet_v1_and_convert_packet(char **buff, size_t size, int &gap) {
    InnoStatusPacketV1 *pkt = reinterpret_cast<InnoStatusPacketV1 *>(*buff);
    if (pkt->common.version.magic_number != kInnoMagicNumberStatusPacket) {
      inno_log_warning("bad magic %x", pkt->common.version.magic_number);
      return false;
    }
    if (size && pkt->common.size > size) {
      inno_log_warning("bad size status packet %" PRI_SIZELU " %u", size, pkt->common.size);
      return false;
    }
    if (pkt->common.size != sizeof(InnoStatusPacketV1)) {
      inno_log_warning("bad size status packet %u %" PRI_SIZELU, pkt->common.size, sizeof(InnoStatusPacketV1));
      return false;
    }

    if (!InnoPacketReader::verify_packet_crc32(&pkt->common)) {
      inno_log_warning("crc32 mismatch for status packet");
      return false;
    }
    if (gap == kMemorryFrontGap) {
      *buff = *buff - kMemorryFrontGap;
      memmove(*buff, pkt, sizeof(InnoStatusPacketV1));
      InnoStatusPacket *pkt1 = reinterpret_cast<InnoStatusPacket *>(*buff);
      // inno_log_info("convert v1-> InnoStatusPacket");
      pkt1->common.size += sizeof(InnoStatusPacket) - sizeof(InnoStatusPacketV1);
      pkt1->common.version.major_version = kInnoProtocolMajorV2;
      pkt1->common.version.minor_version = kInnoProtocolMinorV2;
      memset(&pkt1->in_faults2, 0, sizeof(pkt1->in_faults2) + sizeof(pkt1->extend_reserved));
      InnoPacketReader::set_packet_crc32(&pkt1->common);
    } else if (gap == 0) {
      gap = kMemorryFrontGap;
      return false;
    }
    return true;
  }
};

#endif  // SDK_CLIENT_INNO_LIDAR_PACKET_V1_ADAPT_INTERNAL_H_
