/*
 *  Copyright (C) 2021 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */

#ifndef CONVERTER_CFRAME_LEGACY_H_
#define CONVERTER_CFRAME_LEGACY_H_

#include <math.h>
#include <stdint.h>
#include <sys/types.h>

#ifndef _MSC_VER
#define DEFINE_COMPACT_STRUCT(x) struct __attribute__((packed)) x
#define DEFINE_COMPACT_STRUCT_END
#else
#define DEFINE_COMPACT_STRUCT(x) __pragma(pack(push, 1)) struct x
#define DEFINE_COMPACT_STRUCT_END __pragma(pack(pop))

#define ssize_t __int64
#define __attribute__(X)
#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif
#endif

#ifdef __APPLE__
#include <unistd.h>
#endif

/*****************
 * data structure
 *****************/
typedef double inno_timestamp_us_t;

enum inno_cframe_type {
  INNO_CFRAME_NONE = 0,
  INNO_CFRAME_POINT = 1,
  INNO_CFRAME_CPOINT = 2,
  INNO_CFRAME_MAX = 6,
};
enum inno_timestamp_sync {
  INNO_TIMESTAMP_SYNC_NONE
};

static const int cframe_version_c = 7;
static const uint32_t cpoint_distance_unit_per_meter_c = 100;
static const uint32_t cpoint_angle_unit_per_PI_c = 8192;
static const double cpoint_angle_unit_c = M_PI / cpoint_angle_unit_per_PI_c;

/* compact format, 10 bytes per point */
DEFINE_COMPACT_STRUCT(inno_cpoint) {
  /* distance in cm, range [0, 655.35m] */
  unsigned int radius: 16;
  /* horizontal angle, 0 is straight forward, right is positive,
     unit is cpoint_angle_unit_c rad, range (-PI/2 to -PI/2) */
  int h_angle: 13;
  /* vertical angle, 0 is the horizon, up is positive,
     unit is cpoint_angle_unit_c rad, range (-PI/4 to PI/4) */
  int v_angle: 12;
  unsigned int ts_100us: 14; /* relative timestamp (to ts_us_start) in 100us */
  unsigned int scan_id: 10;
  unsigned int flags: 4;     /* channel and roi flag */
  unsigned int scan_idx: 11;
  unsigned int ref: 16;      /* reflectance, 1-254, 255 means a reflector */
                             /* or intensity, also 1-254 & 255=reflector  */
};
DEFINE_COMPACT_STRUCT_END

/* 24 bytes per point */
DEFINE_COMPACT_STRUCT(inno_point) {
  float x;                 /* in meter, pointing up                     */
  float y;                 /* in meter, pointing right                  */
  float z;                 /* in meter, pointing forward                */
  float radius;            /* in meter                                  */
  uint16_t ts_100us;       /* relative timestamp (to ts_us_start) in 100us */
  uint16_t ref;            /* reflectance, 1-254, 255 means a reflector */
                           /* or intensity, also 1-254 & 255=reflector  */
  unsigned char flags;     /* channel and roi flag                      */
  unsigned int scan_id: 10;
  unsigned int scan_idx: 11;
  unsigned int reserved: 3;
};
DEFINE_COMPACT_STRUCT_END

DEFINE_COMPACT_STRUCT(inno_cframe_header) {
  ssize_t get_size() const {
    if (type == INNO_CFRAME_CPOINT) {
      return sizeof(inno_cpoint) * item_number;
    } else if (type == INNO_CFRAME_POINT) {
      return sizeof(inno_point) * item_number;
    } else {
      return 0;
    }
  }
  uint8_t version;
  // lowest bit 0: not the last sequence of a sub-frame
  // lowest bit 1: not the last sub-frame of a frame
  uint8_t flags;
  unsigned char reserved0[2];
  uint32_t checksum;

  /* in 1/10000000 degree or cframe_geo_co_unit_c rad */
  int32_t longtitude;
  /* in 1/10000000 degree or cframe_geo_co_unit_c rad */
  int32_t latitude;
  /* in cm, range (-327.68 m, 327.68 m] */
  int16_t elevation;

  /* yaw angle, 0 is to the north, PI/2 is east,
     unit is cpoint_angle_unit_c rad, range [-PI to PI] */
  int16_t pose_yaw_angle;
  /* pitch angle, 0 is the horizon, up is positive,
     unit is cpoint_angle_unit_c rad, range [-PI/2 to PI/2] */
  int16_t pose_pitch_angle;
  /* roll angle, 0 is level, positive is clock-wise,
     unit is cpoint_angle_unit_c rad, range [-PI to PI] */
  int16_t pose_roll_angle;

  uint64_t idx;            /* frame index, start from 0                     */
  uint16_t sub_idx;        /* sub-frame index, start from 0 for every frame */
  uint16_t sub_seq;        /* sequence within a sub-frame                   */

  inno_timestamp_us_t ts_us_start;   /* start of frame, in microsecond      */
  inno_timestamp_us_t ts_us_end;     /* end of frame, in microsecond        */
  enum inno_cframe_type type;
  /* one lidar instance can have multiple topics,
     each topic has a serial of frames,
     within one topic, when the new frame arrives,
     the viewer should STOP showing the
     previous frame of the same topic
   */
  unsigned int topic;
  unsigned int item_number;
  /* indicate confidence level for each frame,
     low = 0 indicates the frame may have issue,
     high = 255 indicates a good frame
   */
  unsigned char conf_level;
  unsigned int source_id;
  unsigned char lidar_type;  // InnoLidarType, 0:falcon, 1:rowbinW 2:robinE
  unsigned char reserved[2];
  enum inno_timestamp_sync timestamp_sync_type;
  union {
    char c[0];
    inno_cpoint cpoints[0];
    inno_point points[0];
  };
};
DEFINE_COMPACT_STRUCT_END

#endif  // CONVERTER_CFRAME_LEGACY_H_
