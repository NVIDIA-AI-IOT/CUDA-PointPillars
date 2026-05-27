/*
 *  Copyright (C) 2023 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */


#ifndef SDK_COMMON_INNO_LIDAR_OTHER_PACKET_H_
#define SDK_COMMON_INNO_LIDAR_OTHER_PACKET_H_

#include <stdint.h>

/*
 * Raw4 packet definition
 */
struct InnoRaw4Packet {
  uint32_t idx;

  enum RawPacketType {
    TYPE_SN = 0,
    TYPE_CAUSE = 1,
    TYPE_RAWDATA = 2,
  } field_type;

  bool is_field_last;

  const char* buffer;
  int buffer_size;

  /**
   * @brief Check if the message has ended.
   * @return True if the message has ended.
   */
  bool is_message_last() const { return field_type == TYPE_RAWDATA; }
};


/*
 * Raw4 udp header
 */
struct Raw4UdpHeader {
  // net size
  static const int kHeaderSize = 10;

  // offset
  static const uint16_t kIdxOffset = 0x0;
  static const uint16_t kFieldTypeOffset = 0x4;
  static const uint16_t kFieldSeqIdOffset = 0x5;
  static const uint16_t kFlagOffset = 0x9;

  // flag
  static const uint16_t kFlagFieldEnd = 0x1;

  // member
  uint32_t idx;
  uint8_t field_type;
  uint32_t field_sequence_id;
  uint8_t flag;

  /**
   * @brief Check if the filed has ended.
   * @return True if the field has ended.
   */
  bool is_field_end() const { return flag == kFlagFieldEnd; }

  /**
   * @brief Set the field end flag.
   */
  void set_field_end() { flag = kFlagFieldEnd; }
};

#endif  // SDK_COMMON_INNO_LIDAR_OTHER_PACKET_H_
