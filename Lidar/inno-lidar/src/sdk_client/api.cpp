/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include <fcntl.h>
#include <stdio.h>

#include <limits>

#include "sdk_client/lidar_client.h"
#include "sdk_common/lidar_base.h"
#include "sdk_common/inno_lidar_api.h"
#include "sdk_common/inno_lidar_other_api.h"
#include "utils/log.h"
#include "utils/utils.h"

int inno_lidar_open_live(const char *name,
                         const char *lidar_ip,
                         uint16_t port,
                         enum InnoLidarProtocol protocol,
                         uint16_t udp_port) {
  innovusion::InnoLidarBase *l = NULL;
  switch (protocol) {
    case INNO_LIDAR_PROTOCOL_RAW_MEM:
      inno_log_NOT_IMPLEMENTED();
      break;
    case INNO_LIDAR_PROTOCOL_PCS_UDP:
      l = new innovusion::InnoLidarClient(name, lidar_ip, port,
                                          false, udp_port);
      break;
    case INNO_LIDAR_PROTOCOL_RAW_TCP:
      inno_log_info("raw_tcp not supported, "
                    "force to use pcs_tcp");
      // fall through
    case INNO_LIDAR_PROTOCOL_PCS_TCP:
      l = new innovusion::InnoLidarClient(name, lidar_ip, port,
                                          true, udp_port);
      break;
    default:
      inno_log_panic("invalid protocol %d", protocol);
  }

  int ret = -1;
  if (l) {
    ret = innovusion::InnoLidarBase::add_lidar(l);
  }
  return ret;
}

int inno_lidar_open_file(const char *name,
                         const char *filename,
                         bool raw_format,
                         int play_rate,
                         int rewind,
                         int64_t skip) {
  innovusion::InnoLidarBase *l = NULL;

  // check whether the file exists
  int fd = innovusion::InnoUtils::open_file(filename,
                                            O_RDONLY, 0);
  if (fd < 0) {
    return -1;
  } else {
    close(fd);
  }

  if (raw_format) {
    inno_log_warning("inno_raw file not supported, "
                     "force to use inno_pc file");
  }

  l = new innovusion::InnoLidarClient(name, filename,
                                      play_rate, rewind, skip);

  int ret = -1;
  if (l) {
    ret = innovusion::InnoLidarBase::add_lidar(l);
  }
  return ret;
}


int inno_lidar_open_ctx(const char *name, void* ctx) {
  innovusion::InnoLidarBase *l = NULL;
  l = new innovusion::InnoLidarClient(name, ctx);

  int ret = -1;
  if (l) {
    ret = innovusion::InnoLidarBase::add_lidar(l);
  }
  return ret;
}
