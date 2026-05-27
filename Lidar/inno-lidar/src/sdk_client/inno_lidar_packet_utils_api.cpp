/*
 *  Copyright (C) 2021 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */
#include "sdk_common/inno_lidar_api.h"
#include "sdk_common/inno_lidar_packet_utils.h"
#include "utils/utils.h"
#include "sdk_common/inno_lidar_other_api.h"
size_t inno_lidar_get_data_packet_size(InnoItemType type, uint32_t item_count, InnoMultipleReturnMode mode) {
  return InnoDataPacketUtils::get_data_packet_size(type, item_count, mode);
}

uint32_t inno_lidar_get_max_points_count(const InnoDataPacket *pkt) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::get_max_points_count(*pkt);
}

bool inno_lidar_convert_to_xyz_pointcloud(const InnoDataPacket *src, InnoDataPacket *dest, size_t dest_size,
                                          bool crc_disable) {
  inno_log_verify(src, "src");
  return InnoDataPacketUtils::convert_to_xyz_pointcloud(*src, dest, dest_size, crc_disable);
}

bool inno_lidar_convert_to_xyz_pointcloud2(const InnoDataPacket *src, InnoDataPacket *dest, size_t dest_size,
                                           bool crc_disable, const InnoDataPacket *hvangle_table) {
  const char *table =
      reinterpret_cast<const char *>(reinterpret_cast<const InnoAngleHVTable *>(hvangle_table->payload)->table);
  bool ret = InnoDataPacketUtils::convert_to_xyz_pointcloud(*src, dest, dest_size, crc_disable, nullptr, table);
  return ret;
}

bool inno_lidar_check_data_packet(const InnoDataPacket *pkt, size_t size) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::check_data_packet(*pkt, size);
}

bool inno_lidar_check_status_packet(const InnoStatusPacket *pkt, size_t size) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::check_status_packet(*pkt, size);
}

int inno_lidar_printf_status_packet(const InnoStatusPacket *pkt, char *buffer, size_t buffer_size) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::printf_status_packet(*pkt, buffer, buffer_size);
}

bool inno_lidar_check_status_packet_fault(const InnoStatusPacket *pkt) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::check_status_packet_fault(*pkt);
}

bool inno_lidar_check_status_packet_fault_id(const InnoStatusPacket *pkt, int fid) {
  inno_log_verify(pkt, "pkt");
  return InnoDataPacketUtils::check_status_packet_fault_id(*pkt, fid);
}

bool inno_lidar_correct_imu_status(const InnoStatusPacket *pkt, InnoStatusPacket *out_pkt, bool is_wgs) {
  inno_log_verify(pkt, "pkt");
  inno_log_verify(out_pkt, "out_pkt");
  return InnoDataPacketUtils::fix_imu_status(*pkt, *out_pkt, is_wgs);
}
