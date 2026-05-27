/*
 *  Copyright (C) 2021 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */

#ifndef SDK_COMMON_INNO_LIDAR_PACKET_H_
#define SDK_COMMON_INNO_LIDAR_PACKET_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "sdk_common/inno_faults_common.h"

/************
 Innovusion LiDAR sends out pointcloud data (INNO_ITEM_TYPE_SPHERE_POINTCLOUD)
 and message data (INNO_ITEM_TYPE_MESSAGE) in InnoDataPacket format in
 UDP packets.

 In the pointcloud data, points (InnoChannelPoint) of the same firing
 cycle are organized to 1 block (InnoBlock). In each block, the horizontal
 angle and vertical angle are specifed for the channel-0. For other channels'
 angles, we need to lookup the angle-adjust-table which can be fetch from
 the LiDAR.

 Innovusion LiDAR sends out faults and detail status data in InnoStatusPacket
 format in UDP packets every 50 ms. It contains InnoStatusInFaults,
 InnoStatusExFaults, InnoStatusCounters and InnoStatusSensorReadings.
*************/

/*****************
 * SDK VERSION
 *****************/
#define INNO_SDK_V_MAJOR "3"
#define INNO_SDK_V_MINOR "102"
#define INNO_SDK_V_DOT "6"
#define INNO_SDK_VERSION_IN_HEADER INNO_SDK_V_MAJOR "." INNO_SDK_V_MINOR "." INNO_SDK_V_DOT "."

/************
 Macros
*************/
#ifndef DEFINE_INNO_COMPACT_STRUCT
#if !(defined(_MSC_VER))
#define DEFINE_INNO_COMPACT_STRUCT(x) struct __attribute__((packed)) x
#define DEFINE_INNO_COMPACT_STRUCT_END
#else

#define DEFINE_INNO_COMPACT_STRUCT(x) __pragma(pack(push, 1)) struct x
#define DEFINE_INNO_COMPACT_STRUCT_END __pragma(pack(pop))
#endif
#endif

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif


/************
 Enums
*************/
enum InnoLidarMode {
  INNO_LIDAR_MODE_NONE = 0,
  INNO_LIDAR_MODE_SLEEP = 1,               // falcon & robin
  INNO_LIDAR_MODE_STANDBY = 2,             // falcon & robin
  INNO_LIDAR_MODE_WORK_NORMAL = 3,         // falcon & robin
  INNO_LIDAR_MODE_WORK_SHORT_RANGE = 4,    // falcon
  INNO_LIDAR_MODE_WORK_CALIBRATION = 5,    // falcon & robin
  INNO_LIDAR_MODE_PROTECTION = 6,          // falcon & robin
  INNO_LIDAR_MODE_WORK_QUIET = 7,          // falcon
  INNO_LIDAR_MODE_WORK_INTERNAL_1 = 8,     // falcon
  INNO_LIDAR_MODE_FOTA = 9,                // [UDS upgrade & ADC FOTA]
  INNO_LIDAR_MODE_WORK_EXHIBITION = 10,    // falcon
  INNO_LIDAR_MODE_WORK_MAX = 11,
};

enum InnoLidarStatus {
  INNO_LIDAR_STATUS_NONE = 0,
  INNO_LIDAR_STATUS_TRANSITION = 1,
  INNO_LIDAR_STATUS_NORMAL = 2,
  INNO_LIDAR_STATUS_FAILED = 3,
  INNO_LIDAR_STATUS_MAX = 4,
};

enum InnoTimeSyncConfig {
  INNO_TIME_SYNC_CONFIG_HOST = 0,
  INNO_TIME_SYNC_CONFIG_PTP = 1,
  INNO_TIME_SYNC_CONFIG_GPS = 2,
  INNO_TIME_SYNC_CONFIG_FILE = 3,
  INNO_TIME_SYNC_CONFIG_NTP = 4,
  INNO_TIME_SYNC_CONFIG_MAX = 5,
};

enum InnoTimeSyncType {
  INNO_TIME_SYNC_TYPE_NONE = 0,
  INNO_TIME_SYNC_TYPE_RECORDED = 1,
  INNO_TIME_SYNC_TYPE_HOST = 2,
  INNO_TIME_SYNC_TYPE_GPS_INIT = 3,
  INNO_TIME_SYNC_TYPE_GPS_LOCKED = 4,
  INNO_TIME_SYNC_TYPE_GPS_UNLOCKED = 5,
  INNO_TIME_SYNC_TYPE_PTP_INIT = 6,
  INNO_TIME_SYNC_TYPE_PTP_LOCKED = 7,
  INNO_TIME_SYNC_TYPE_PTP_UNLOCKED = 8,
  INNO_TIME_SYNC_TYPE_FILE_INIT = 9,
  INNO_TIME_SYNC_TYPE_NTP_INIT = 10,
  INNO_TIME_SYNC_TYPE_NTP_LOCKED = 11,
  INNO_TIME_SYNC_TYPE_NTP_UNLOCKED = 12,
  INNO_TIME_SYNC_TYPE_GPS_LOST = 13,
  INNO_TIME_SYNC_TYPE_PTP_LOST = 14,
  INNO_TIME_SYNC_TYPE_NTP_LOST = 15,
  INNO_TIME_SYNC_TYPE_MAX = 16,
};

enum InnoLidarExFault {
  INNO_LIDAR_EX_FAULT_OTHER = 0,
};

enum InnoMessageLevel {
  INNO_MESSAGE_LEVEL_FATAL    = 0,
  INNO_MESSAGE_LEVEL_CRITICAL = 1,
  INNO_MESSAGE_LEVEL_ERROR    = 2,
  INNO_MESSAGE_LEVEL_TEMP     = 3,
  INNO_MESSAGE_LEVEL_WARNING  = 4,
  INNO_MESSAGE_LEVEL_DEBUG    = 5,
  INNO_MESSAGE_LEVEL_INFO     = 6,
  INNO_MESSAGE_LEVEL_TRACE    = 7,
  INNO_MESSAGE_LEVEL_DETAIL   = 8,
  INNO_MESSAGE_LEVEL_MAX      = 9,
};

enum InnoMessageCode {
  INNO_MESSAGE_CODE_NONE = 0,
  INNO_MESSAGE_CODE_LIB_VERSION_MISMATCH,
  INNO_MESSAGE_CODE_READ_TIMEOUT,
  INNO_MESSAGE_CODE_CANNOT_READ,
  INNO_MESSAGE_CODE_BAD_CONFIG_YAML,
  INNO_MESSAGE_CODE_OVERHEAT_PROTECTION,
  INNO_MESSAGE_CODE_TO_NON_WORKING_MODE,
  INNO_MESSAGE_CODE_READ_FILE_END,
  INNO_MESSAGE_CODE_RAW_RECORDING_FINISHED,
  INNO_MESSAGE_CODE_NEW_START,
  INNO_MESSAGE_CODE_ROI_CHANGED,
  INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT = 10001,
  INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT = 10002,
};

enum InnoFrameDirection {
  INNO_FRAME_DIRECTION_DOWN = 0,  /* top->bottom   */
  INNO_FRAME_DIRECTION_UP = 1,    /* bottom -> top */
  INNO_FRAME_DIRECTION_MAX = 2,
};

enum InnoItemType {
  INNO_ITEM_TYPE_NONE = 0,
  // Falcon SPHERE POINTCLOUD, InnoBlock
  INNO_ITEM_TYPE_SPHERE_POINTCLOUD = 1,
  INNO_ITEM_TYPE_MESSAGE = 2,
  INNO_ITEM_TYPE_MESSAGE_LOG = 3,
  // Falcon SPHERE POINTCLOUD, InnoXyzPoint
  INNO_ITEM_TYPE_XYZ_POINTCLOUD = 4,

  // ROBIN_E SPHERE POINTCLOUD, InnoEnBlock
  INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD = 5,
  // ROBIN_E SPHERE POINTCLOUD, InnoEnXyzPoint
  INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD = 6,

  // ROBIN_W SPHERE POINTCLOUD, InnoEnBlock
  INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD = 7,
  // ROBIN_W SPHERE POINTCLOUD, InnoEnXyzPoint
  INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD = 8,

  // Falcon2.1 SPHERE POINTCLOUD, InnoEnBlock
  INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD = 9,
  // Falcon2.1 XYZ POINTCLOUD, InnoEnXyzPoint
  INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD = 10,

  // FalconIII
  INNO_FALCONIII_ITEM_TYPE_SPHERE_POINTCLOUD = 11,
  INNO_FALCONIII_ITEM_TYPE_XYZ_POINTCLOUD = 12,

  // ROBINW COMPACT POINTCLOUD, InnoCoPoint
  INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD = 13,

  // ROBIN_E lite SPHERE POINTCLOUD
  INNO_ROBINELITE_ITEM_TYPE_SPHERE_POINTCLOUD = 14,
  // ROBIN_E lite SPHERE POINTCLOUD, InnoEnXyzPoint
  INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD = 15,
  // ROBINELITE COMPACT POINTCLOUD, InnoCoPoint
  INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD = 16,
  // ROBINW AngleHV TABLE
  INNO_ROBINW_ITEM_TYPE_ANGLEHV_TABLE = 100,
  // ROBINELITE AngleHV TABLE
  INNO_ROBINE_LITE_TYPE_ANGLEHV_TABLE = 101,
  // RingId Table
  INNO_FALCON_RING_ID_TABLE = 102,

  INNO_ITEM_TYPE_MAX = 103,
};

enum InnoReflectanceMode {
  INNO_REFLECTANCE_MODE_NONE = 0,
  INNO_REFLECTANCE_MODE_INTENSITY = 1,
  INNO_REFLECTANCE_MODE_REFLECTIVITY = 2,
  INNO_REFLECTANCE_MODE_MAX = 3,
};

enum InnoMultipleReturnMode {
  /* xxx TODO */
  INNO_MULTIPLE_RETURN_MODE_NONE = 0,
  INNO_MULTIPLE_RETURN_MODE_SINGLE = 1,
  INNO_MULTIPLE_RETURN_MODE_2_STRONGEST = 2,
  // one strongest return and one furthest return
  INNO_MULTIPLE_RETURN_MODE_2_STRONGEST_FURTHEST = 3,
  INNO_MULTIPLE_RETURN_MODE_MAX
};

enum InnoConfidenceLevel {
  INNO_NO_CONFIDENCE = 0,
  INNO_RARE_CONFIDENCE = 1,
  INNO_WELL_CONFIDENCE = 2,
  INNO_FULL_CONFIDENCE = 3,
};

enum InnoGalvoMode {
  INNO_GALVO_MODE_NONE = 0,
  INNO_GALVO_MODE_NORMAL,
  INNO_GALVO_MODE_FLYBACK,
  INNO_GALVO_MODE_MAX
};

enum InnoLidarType {
  INNO_LIDAR_TYPE_FALCON = 0,  // FalconK1
  INNO_LIDAR_TYPE_ROBINW = 1,
  INNO_LIDAR_TYPE_ROBINE = 2,
  INNO_LIDAR_TYPE_FALCONII_DOT_1 = 3,  // Falcon2.1
  INNO_LIDAR_TYPE_FALCONK2 = 3,        // FalconK2
  INNO_LIDAR_TYPE_FALCONIII = 4,
  INNO_LIDAR_TYPE_ROBINELITE = 5,
};

enum InnoDistanceUnitPerMeter {
  kInnoDistanceUnitPerMeter200 = 200,  // FalconK & FalconK2
  kInnoDistanceUnitPerMeter400 = 400   // Robin
};

enum InnoVAngleDiffBase {
  kInnoFaconVAngleDiffBase = 196,   // FalconK & FalconK2
  kInnoRobinEVAngleDiffBase = 0,    // RobinE
  kInnoRobinWVAngleDiffBase = 240,  // RobinW
};

enum InnoSetNumber {
  kInnoRobinWMaxSetNumber = 6,
  kInnoRobinELiteMaxSetNumber = 12
};
/************
 Simple types
*************/
/* epoch time in micro-sec */
typedef double InnoTimestampUs;

/************
 Constants
*************/
#define INNO_CHANNEL_NUMBER_BIT 2
#define INNO_CHANNEL_NUMBER (1 << INNO_CHANNEL_NUMBER_BIT)
#define INNO_MAX_MULTI_RETURN 2
#define INNO_SN_SZIE 16
#define INNO_HW_NUMBER_SIZE 3
#define INNO_COMPACT_CHANNEL_NUMBER_BIT 3
#define INNO_COMPACT_CHANNEL_NUMBER (1 << INNO_COMPACT_CHANNEL_NUMBER_BIT)
static const uint16_t kInnoMagicNumberDataPacket = 0x176A;
static const uint8_t kInnoMajorVersionDataPacket = 4;  // upgrade Major version from 3->4 2024/8/16
static const uint8_t kInnoMinorVersionDataPacket = 0;
static const uint16_t kInnoMagicNumberStatusPacket = 0x186B;
static const uint8_t kInnoMajorVersionStatusPacket = 4;  // upgrade Major version from 3->4 2023/8/16
static const uint8_t kInnoMinorVersionStatusPacket = 0;

static const uint32_t kInnoDistanceUnitPerMeter = 400;
static const double kMeterPerInnoDistanceUnit = 1.0 / kInnoDistanceUnitPerMeter;
static const double kMeterPerInnoDistanceUnit200 = 1.0 / kInnoDistanceUnitPerMeter200;  // falconK & falconK2
static const double kMeterPerInnoDistanceUnit400 = 1.0 / kInnoDistanceUnitPerMeter400;  // robin & falcon2.1
static const uint32_t kInnoDegreePerPiRad = 180;
static const uint32_t kInnoAngleUnitPerPiRad = 32768;
static const double kRadPerInnoAngleUnit = M_PI / kInnoAngleUnitPerPiRad;
static const double kDegreePerInnoAngleUnit =
    180.0 / kInnoAngleUnitPerPiRad;
static const double kInnoAngleUnitPerDegree =
    kInnoAngleUnitPerPiRad / 180.0;
static const int32_t kInnoInvalidAngleInUnit = kInnoAngleUnitPerDegree * 90;
static const uint32_t kInnoChannelNumberBit = INNO_CHANNEL_NUMBER_BIT;
static const uint32_t kInnoChannelNumber = INNO_CHANNEL_NUMBER;
static const uint32_t kInnoMaxMultiReturn = INNO_MAX_MULTI_RETURN;
static const int16_t kInnoVAngleDiffBase = 196;
static const uint32_t kInnoCompactChannelNumberBit = INNO_COMPACT_CHANNEL_NUMBER_BIT;
static const uint32_t kInnoCompactChannelNumber = INNO_COMPACT_CHANNEL_NUMBER;

static const int kPolygonMaxFacets = 4;
static const int kPolygonMinAngle = - (45 * kInnoAngleUnitPerPiRad / kInnoDegreePerPiRad);
static const int kPolygonMaxAngle = 45 * kInnoAngleUnitPerPiRad / kInnoDegreePerPiRad;
static const int kEncoderTableShift = 8;  // table resolution
static const int kEncoderTableStep = 1 << kEncoderTableShift;
static const int kEncoderTableMask = kEncoderTableStep - 1;
static const int kPolygonTableSize = ((kPolygonMaxAngle - kPolygonMinAngle) >> kEncoderTableShift) + 1;
static const int kMaxReceiverInSet = kInnoCompactChannelNumber;

static const int kInnoBaseFaultEnd = 64;
static const double kInnoNopROI = 10000.0;

static const uint8_t kInnoAngleHVTableVersionMajor = 0;
static const uint8_t kInnoAngleHVTableVersionMinor = 1;
static const uint32_t kInnoAngleHVTableMaxSize = 120000;
enum InputSource {
  SOURCE_NO = 0,
  SOURCE_FILE,
  SOURCE_TCP,
  SOURCE_UDP,
  SOURCE_PCAP,
  SOURCE_MAX,
};

union InputParam {
  struct {
    enum InputSource source_type;
    char lidar_ip[16];
  } base_param;

  struct {
    enum InputSource source_type;
    char lidar_ip[16];
    char filename[256];
    int32_t play_rate;
    int32_t rewind;
    int64_t skip;
  } file_param;

  struct {
    enum InputSource source_type;
    char lidar_ip[16];
    double read_timeout_sec;
    int64_t skip;
  } tcp_param;

  struct {
    enum InputSource source_type;
    char lidar_ip[16];
    uint16_t udp_port;
    double read_timeout_sec;
    int64_t skip;
  } udp_param;

  struct {
    enum InputSource source_type;
    char lidar_ip[16];
    char filename[256];
    int32_t play_rate;
    int32_t rewind;
    int64_t skip;
    uint16_t data_port;
    uint16_t status_port;
    uint16_t message_port;
  } pcap_param;
};

enum InnoLidarProtocol {
  INNO_LIDAR_PROTOCOL_NONE = 0,
  INNO_LIDAR_PROTOCOL_RAW_TCP = 1,
  INNO_LIDAR_PROTOCOL_RAW_MEM = 2,
  INNO_LIDAR_PROTOCOL_PCS_TCP = 3,
  INNO_LIDAR_PROTOCOL_PCS_UDP = 4,
  INNO_LIDAR_PROTOCOL_RAW_FILE = 5,
  INNO_LIDAR_PROTOCOL_PCS_FILE = 6,
  INNO_LIDAR_PROTOCOL_MAX = 7,
};

enum InnoLidarState {
  INNO_LIDAR_STATE_ERROR = 0,
  INNO_LIDAR_STATE_STARTING = 1,
  INNO_LIDAR_STATE_READY = 2,
  INNO_LIDAR_STATE_STREAMING = 3,
  INNO_LIDAR_STATE_UNKNOWN = 4,
  INNO_LIDAR_STATE_INVALID = 5,
};

enum InnoRecorderCallbackType {
  INNO_RECORDER_CALLBACK_TYPE_NONE = 0,
  INNO_RECORDER_CALLBACK_TYPE_RAW = 1,
  INNO_RECORDER_CALLBACK_TYPE_RAW2 = 2,
  INNO_RECORDER_CALLBACK_TYPE_RAW3 = 3,
  INNO_RECORDER_CALLBACK_TYPE_RAW4 = 4,
  INNO_RECORDER_CALLBACK_TYPE_CALI = 5,
  INNO_RECORDER_CALLBACK_TYPE_SCATTER = 6,
  INNO_RECORDER_CALLBACK_TYPE_HILOUT = 7,
  INNO_RECORDER_CALLBACK_TYPE_INNO_PC = 8,
  INNO_RECORDER_CALLBACK_TYPE_MAX = 9,
};


#if defined(__MINGW64__) || !defined(_WIN32)
/* 17 bytes per block header */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoBlockHeader) {
  /* horizontal angle, 0 is straight forward, right is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t h_angle;
  /* vertical angle, 0 is the horizon, up is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t v_angle;
  /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
  uint16_t ts_10us;
  uint16_t scan_idx;     /* point idx within the scan line */
  uint16_t scan_id: 9;   /* id of the scan line */
  // real angle is h_angle + h_angle_diff_1
  int64_t h_angle_diff_1: 9;
  int64_t h_angle_diff_2: 10;
  int64_t h_angle_diff_3: 11;
  // real angle is v_angle + v_angle_diff_1 + kVAngleDiffBase * channel
  int64_t v_angle_diff_1: 8;  // 196 + [-128, 127]
  int64_t v_angle_diff_2: 9;  // 392 + [-256, 255]
  int64_t v_angle_diff_3: 9;  // 588 + [-256, 255]
  /*   0: in sparse region
    0x01: in vertical slow region
    0x11: in center ROI */
  uint64_t in_roi: 2;
  uint64_t facet: 3;
  uint64_t reserved_flags: 2; /* all 0 */
} InnoBlockHeader;
DEFINE_INNO_COMPACT_STRUCT_END
#else
/* 17 bytes per block header */
DEFINE_INNO_COMPACT_STRUCT(InnoBlockHeader) {
  /* horizontal angle, 0 is straight forward, right is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t h_angle;
  /* vertical angle, 0 is the horizon, up is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t v_angle;
  /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
  uint16_t ts_10us;
  uint16_t scan_idx;    /* point idx within the scan line */
  union {
    uint16_t scan_id : 9; /* id of the scan line */
    struct {
      uint8_t scan_id0;
      int64_t scan_id1 : 1;
      // real angle is h_angle + h_angle_diff_1
      int64_t h_angle_diff_1 : 9;
      int64_t h_angle_diff_2 : 10;
      int64_t h_angle_diff_3 : 11;
      // real angle is v_angle + v_angle_diff_1 + kVAngleDiffBase * channel
      int64_t v_angle_diff_1 : 8;  // 196 + [-128, 127]
      int64_t v_angle_diff_2 : 9;  // 392 + [-256, 255]
      int64_t v_angle_diff_3 : 9;  // 588 + [-256, 255]
      /*   0: in sparse region
        0x01: in vertical slow region
        0x11: in center ROI */
      uint64_t in_roi : 2;
      uint64_t facet : 3;
      uint64_t reserved_flags : 2; /* all 0 */
    };
  };
};
DEFINE_INNO_COMPACT_STRUCT_END
#endif

typedef DEFINE_INNO_COMPACT_STRUCT(InnoXyzrD) {
  double x;
  double y;
  double z;
  double radius;
} InnoXyzrD;
DEFINE_INNO_COMPACT_STRUCT_END

/* compact format, 16 + 8 + 2 = 26 bytes per point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoXyzPoint) {
  float x;
  float y;
  float z;
  float radius;
  uint16_t ts_10us;
  uint16_t scan_id: 9;   /* id of the scan line */
  uint16_t in_roi: 2;
  uint16_t facet: 3;
  uint16_t multi_return: 1;
  uint16_t reserved_flags: 1; /* all 0 */
  uint32_t is_2nd_return: 1;
  uint32_t scan_idx: 14;   /* point idx within the scan line */
  uint32_t refl: 9;        /* reflectance, 1-254, 255 means a reflector     */
                           /* or intensity, also 1-254 & 255=reflector      */
  uint32_t type: 2;        /* 0: normal, 1: ground, 2: fog                  */
  uint32_t elongation: 4;  /* elongation */
  uint32_t channel: 2;
  uint16_t ring_id;
} InnoXyzPoint;
DEFINE_INNO_COMPACT_STRUCT_END

/* compact format, 4 bytes per point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoChannelPoint) {
  /* distance in distance unit, distance unit:1/200m, long_distance unit 1/100m */
  uint32_t radius : 17;
  uint32_t refl: 8;        /* reflectance, 1-254, 255 means a reflector     */
                           /* or intensity, also 1-254 & 255=reflector      */
  uint32_t is_2nd_return: 1; /* 0: 1st return, 1: 2nd return                */
  uint32_t type: 2;        /* 0: normal, 1: ground, 2: fog                  */
  uint32_t elongation: 4;  /* elongation */
} InnoChannelPoint;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoBlock) {
  InnoBlockHeader header;
  InnoChannelPoint points[0];
} InnoBlock;
DEFINE_INNO_COMPACT_STRUCT_END

/* 17 + 4 * 4 = 33 bytes */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoBlock1) {
  InnoBlockHeader header;
  InnoChannelPoint points[INNO_CHANNEL_NUMBER];
} InnoBlock1;
DEFINE_INNO_COMPACT_STRUCT_END

/* 17 + 8 * 4 = 49 bytes */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoBlock2) {
  InnoBlockHeader header;
  InnoChannelPoint points[INNO_CHANNEL_NUMBER * INNO_MAX_MULTI_RETURN];
} InnoBlock2;
DEFINE_INNO_COMPACT_STRUCT_END

/* compact format, 8 bytes per point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnChannelPoint) {
  uint16_t reflectance;            /* reflectance, falcon 1-65535,robin 1-4095  */
  uint16_t intensity;           /* intensity, falcon 1-65535,robin 1-4095  */
  uint32_t elongation: 7;      /* elongation unit: 1ns */
  uint32_t is_2nd_return: 1;    /* 0: 1st return, 1: 2nd return                  */
  uint32_t radius : 19;         /* distance in distance unit, distance unit:1/400m, range [0, 655.35m] */
  uint32_t type : 2;            /* 0: normal, 1: ground, 2: fog                  */
  uint32_t firing: 1;           /* 0: weak, 1: strong */
  uint32_t reserved_flags : 2;  /* all 0 */
} InnoEnChannelPoint;
DEFINE_INNO_COMPACT_STRUCT_END

/* compact format, 40 bytes per point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnXyzPoint) {
  float x;
  float y;
  float z;
  float radius;
  uint16_t ts_10us;   /*relate time to InnoDataPacket ts_start_us, deprecated */
  uint16_t scan_id;   /* id of the scan line */
  uint16_t scan_idx;  /* point idx within the scan line */
  double timestamp_s; /* epoch time of the point, in second */
  uint16_t reflectance;
  uint16_t intensity;
  uint8_t facet;  /* 0-4 */
  uint8_t channel;
  uint8_t firing;
  uint8_t in_roi;
  uint8_t is_2nd_return;
  uint8_t multi_return; /* multi return mode,true mean the 2nd point*/
} InnoEnXyzPoint;
DEFINE_INNO_COMPACT_STRUCT_END

/* 18 bytes per EnBlock header */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnBlockHeader) {
  /* horizontal angle, 0 is straight forward, right is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t h_angle;
  /* vertical angle, 0 is the horizon, up is positive,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t v_angle;
  /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
  uint16_t ts_10us;
  // real angle is h_angle + h_angle_diff_1
  int64_t h_angle_diff_1 : 11;
  int64_t h_angle_diff_2 : 11;
  int64_t h_angle_diff_3 : 12;
  // real angle is v_angle + v_angle_diff_1 + kVAngleDiffBase * channel
  int64_t v_angle_diff_1 : 10;
  int64_t v_angle_diff_2 : 10;
  int64_t v_angle_diff_3 : 10;
  uint16_t scan_idx;            /* point idx within the scan line */
  uint16_t scan_id : 9;         /* id of the scan line */
  /*   0: in sparse region
  0x01: in vertical slow region
  0x11: in center ROI
  only for falcon  */
  uint16_t in_roi : 2;
  uint16_t facet : 3;
  uint16_t reserved_flags : 2; /* all 0 */
} InnoEnBlockHeader;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnBlock) {
  InnoEnBlockHeader header;
  InnoEnChannelPoint points[0];
} InnoEnBlock;
DEFINE_INNO_COMPACT_STRUCT_END

/* 18 + 4 * 8 = 50 bytes */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnBlock1) {
  InnoEnBlockHeader header;
  InnoEnChannelPoint points[INNO_CHANNEL_NUMBER];
} InnoEnBlock1;
DEFINE_INNO_COMPACT_STRUCT_END

/* 18 + 8 * 8 = 82 bytes */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnBlock2) {
  InnoEnBlockHeader header;
  InnoEnChannelPoint points[INNO_CHANNEL_NUMBER * INNO_MAX_MULTI_RETURN];
} InnoEnBlock2;
DEFINE_INNO_COMPACT_STRUCT_END

/* compact format, 4 bytes per point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoChannelPoint) {
  uint32_t refl : 12;    /* reflectance or intensity robin 1-4095  */
  uint32_t radius : 18;         /* distance in distance unit, distance unit:1/400m, range [0, 655.35m] */
  uint32_t is_2nd_return: 1;    /* 0: 1st return, 1: 2nd return                  */
  uint32_t firing: 1;           /* 0: weak, 1: strong */
} InnoCoChannelPoint;

DEFINE_INNO_COMPACT_STRUCT_END
/* 10 bytes per CoBlock header */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlockHeader) {
  /* polygon angle, 0 is straight forward,
     unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
  int16_t p_angle;
  int16_t g_angle;
  /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
  uint16_t ts_10us;
  uint16_t scan_idx;            /* point idx within the scan line */
  uint16_t scan_id : 9;         /* id of the scan line */
  /*   0: in sparse region
  0x01: in vertical slow region
  0x11: in center ROI
  only for falcon  */
  uint16_t in_roi : 2;
  uint16_t facet : 3;
  uint16_t reserved_flags : 2; /* all 0 */
} InnoCoBlockHeader;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlock) {
  InnoCoBlockHeader header;
  InnoCoChannelPoint points[0];
} InnoCoBlock;
DEFINE_INNO_COMPACT_STRUCT_END

/* 10 + 4 * 8 = 42 bytes, 5.25 bytes/point */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlock1) {
  InnoCoBlockHeader header;
  InnoCoChannelPoint points[INNO_COMPACT_CHANNEL_NUMBER];
} InnoCoBlock1;
DEFINE_INNO_COMPACT_STRUCT_END

/* 10 + 8 * 8 = 74 bytes */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlock2) {
  InnoCoBlockHeader header;
  InnoCoChannelPoint points[INNO_COMPACT_CHANNEL_NUMBER * INNO_MAX_MULTI_RETURN];
} InnoCoBlock2;
DEFINE_INNO_COMPACT_STRUCT_END

static inline size_t innoblock_get_idx(size_t channel, size_t r) {
  /* r0_ch0 r0_ch1 r0_ch2 r0_ch3 r1_ch0 r1_ch1 r1_ch2 r1_ch3 */
  return channel + (r << kInnoChannelNumberBit);
}

static inline size_t innocoblock_get_idx(size_t channel, size_t r) {
  /* r0_ch0 r0_ch1 r0_ch2 r0_ch3 r0_ch4 r0_ch5 r0_ch6 r0_ch7
     r1_ch0 r1_ch1 r1_ch2 r1_ch3 r1_ch4 r1_ch5 r1_ch6 r1_ch7 */
  return channel + (r << kInnoCompactChannelNumberBit);
}

typedef DEFINE_INNO_COMPACT_STRUCT(InnoMessage) {
  uint32_t size;  // size of the whole InnoMessage,
                  //   i.e. size of content + sizeof(InnoMessage)
  uint32_t src;
  uint64_t id;
  uint32_t level;      /* enum InnoMessageLevel */
  uint32_t code;       /* message code          */
  int32_t reserved[4]; /* all 0                 */
  char content[0];     /* 0 end string          */
} InnoMessage;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoRingIdTable) {
  uint8_t table[256];  // uint8_t 0-255
} InnoRingIdTable;
DEFINE_INNO_COMPACT_STRUCT_END

/*
  Fixed header structure to indicate the firmware/software version.
  This structure won't change during firmware update in the future.
*/
typedef DEFINE_INNO_COMPACT_STRUCT(InnoCommonVersion) {
  /* 2 byte */
  uint16_t magic_number;

  /* 2 byte */
  uint8_t major_version;
  uint8_t minor_version;

  /* 2 byte */
  uint16_t fw_sequence;
} InnoCommonVersion;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoCommonHeader) {
  /* 6 bytes */
  InnoCommonVersion version;

  /* 4 bytes, cover every thing except checksum */
  uint32_t checksum;

  /* 4 bytes */
  uint32_t size;

  /* 2 bytes */
  uint8_t source_id : 4;           /* up to 16 different LiDAR source */
  uint8_t timestamp_sync_type : 4; /* enum InnoTimestampSyncType      */
  uint8_t lidar_type;          /* enum InnoLidarType */
  /* 8 bytes */
  InnoTimestampUs ts_start_us; /* epoch time of start of frame, in micro-sec */

  /* 2 bytes */
  uint8_t lidar_mode;        /* enum InnoLidarMode    */
  uint8_t lidar_status;      /* enum InnoLidarStatus  */
} InnoCommonHeader;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(AngleHV) {
  int16_t v;
  int16_t h;
} AngleHV;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoAngleHVTableVersion) {
  /* 2 byte */
  uint8_t major_version;
  uint8_t minor_version;
} InnoAngleHVTableVersion;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoAngleHVTable) {
  InnoAngleHVTableVersion version_number;
  uint64_t id;
  AngleHV table[kPolygonMaxFacets][kPolygonTableSize][kInnoRobinWMaxSetNumber][kMaxReceiverInSet];
  uint8_t reserved[512];
} InnoAngleHVTable;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoRobinELiteAngleHVTable) {
  InnoAngleHVTableVersion version_number;
  uint64_t id;
  AngleHV table[kPolygonMaxFacets][kPolygonTableSize][kInnoRobinELiteMaxSetNumber][kMaxReceiverInSet];
  uint8_t reserved[512];
} InnoRobinELiteAngleHVTable;
DEFINE_INNO_COMPACT_STRUCT_END

/*
 * Main data packet definition
 * 26 + 12 + 10 + 2 + 4 + 16 = 70 bytes, max overhead is 70/1472 = 4.7%,
 */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoDataPacket) {
  InnoCommonHeader common;

  /* 12 bytes */
  uint64_t idx;         /* frame index, start from 0                     */
  uint16_t sub_idx;     /* sub-frame index, start from 0 for every frame */
  uint16_t sub_seq;     /* sequence of InnoDataPacket, start from 0     */

  /* 10 byte */
  /* type in enum InnoItemType, each type uses independent global idx */
  uint32_t type :8;
  uint32_t item_number :24;        /* max 4 * 1024 * 1024               */
  uint16_t item_size;              /* max 65535, 0 means variable size  */
  uint32_t topic;                  /* reserved                          */

  /* 2 bytes */
  uint16_t scanner_direction :1; /* 0: top->bottom, 1: bottom->top          */
  uint16_t use_reflectance   :1; /* 0: intensity mode, 1: reflectance mode  */
  uint16_t multi_return_mode :3; /* ... */
  uint16_t confidence_level  :2; /* 0: no confidence, 3: higest             */
  uint16_t is_last_sub_frame :1; /* 1: the last sub frame of a frame        */
  uint16_t is_last_sequence  :1; /* 1: the last piece of a sub frame        */
  uint16_t has_tail :1;          /* has additional tail struct after points */
  uint16_t frame_sync_locked :1; /* 1: frame sync has locked                */
  uint16_t is_first_sub_frame :1; /* 1: the first sub frame of a frame      */
  uint16_t last_four_channel :1;
  uint16_t long_distance_mode : 1; /* lidar work in long distance mode,only for falconk */
  uint16_t reserved_flag : 2; /* all 0 */

  /* 4 bytes */
  int16_t roi_h_angle;           /* configured ROI in InnoAngleUnit */
  int16_t roi_v_angle;
  /* The output from the FalconK LiDAR does not include the extend_reserved field, which is
                                  inserted by the client SDK */
  uint32_t extend_reserved[4];
// MSVC compiler does not support multi-dimensional flexible arrays.
# if !defined(_MSC_VER)
  union {
    char payload[0];
    InnoBlock1 inno_block1s[0];
    InnoBlock2 inno_block2s[0];
    InnoMessage messages[0];
    InnoXyzPoint xyz_points[0];
        // Robin & Falcon2.1
    InnoEnBlock1 inno_en_block1s[0];
    InnoEnBlock2 inno_en_block2s[0];
    InnoCoBlock1 inno_co_block1s[0];
    InnoCoBlock2 inno_co_block2s[0];
    InnoEnXyzPoint en_xyz_points[0];
    InnoAngleHVTable inno_anglehv_table[0];
    InnoRobinELiteAngleHVTable inno_robinelite_anglehv_table[0];
    InnoRingIdTable ring_id_table[0];
  };
#else
  char payload[0];
#endif
} InnoDataPacket;
DEFINE_INNO_COMPACT_STRUCT_END

typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusInFaults) {
  /*
   *  Each bit represent one fault defined in enum InnoLidarInFault.
   *  0 means no fault.
   */
  uint64_t faults;
  uint32_t extended_faults;
} InnoStatusInFaults;
DEFINE_INNO_COMPACT_STRUCT_END


typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusExFaults) {
  /*
   *  Each bit represent one fault defined in enum InnoLidarExFault.
   *  0 means no fault.
   */
  uint32_t faults;
} InnoStatusExFaults;
DEFINE_INNO_COMPACT_STRUCT_END


/*
 * 320 bytes InnoStatusCounters
 */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusCounters) {
  uint64_t point_data_packet_sent;
  uint64_t point_sent;
  uint64_t message_packet_sent;
  uint64_t raw_data_read;
  uint64_t total_frame;
  uint64_t total_polygon_rotation;
  uint64_t total_polygon_facet;
  uint32_t power_up_time_in_second;
  uint32_t process_up_time_in_second;
  uint32_t lose_ptp_sync;
  uint32_t bad_data[4];
  uint32_t data_drop[8];
  uint32_t in_signals[8];
  uint16_t latency_10us_average[6];
  uint16_t latency_10us_variation[6];
  uint16_t latency_10us_max[6];
  uint32_t big_latency_frame;
  uint32_t bad_frame;
  uint32_t big_gap_frame;
  uint32_t small_gap_frame;
  uint16_t cpu_percentage;
  uint16_t mem_percentage;
  uint16_t motor[5];  /* std,min,max1,max2 */
  uint16_t galvo[5];  /* std,min,max1,max2 */
  uint16_t netstat_rx_speed_kBps;
  uint16_t netstat_tx_speed_kBps;
  uint16_t netstat_rx_drop;
  uint16_t netstat_tx_drop;
  uint16_t netstat_rx_err;
  uint16_t netstat_tx_err;
  uint16_t sys_cpu_percentage[4];
  uint32_t lifelong_uptime;
  uint32_t reserved[18];
} InnoStatusCounters;
DEFINE_INNO_COMPACT_STRUCT_END

/* 192 bytes InnoStatusSensorReadings */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusSensorReadings) {
  int16_t temperature_fpga_10th_c;
  int16_t temperature_laser_10th_c;
  int16_t temperature_adc_10th_c;
  int16_t temperature_board_10th_c;
  int16_t temperature_det_10th_c[4];
  int16_t temperature_other_10th_c[3];
  uint16_t heater_current_ma;

  uint32_t motor_rpm_1000th;          /* polygon rpm * 1000   */
  uint32_t galvo_fpm_1000th;          /* frame per min * 1000 */
  uint64_t motor_rotation_total;
  uint64_t galvo_round_total;
  uint16_t moisture_index[2];         /* moisture index        */
  uint16_t window_blockage_index[2];  /* window blockage index */
  uint16_t motor[6];  /* ma */
  uint16_t galvo[6];  /* ma */
  uint16_t laser[6];  /* ma */
  uint16_t galvo_status_client;        /* set in client sdk */
  uint16_t galvo_offset_angle_client;  /* set in client sdk */

  /* motor voltage, unit is 1mV(millivolt) */
  uint32_t motor_dc_bus_voltage;
  uint16_t motor_speed_control_err;
  uint16_t galvo_position_control_err;
  /* lidar unit current, unit is 1mA(milliampere) */
  uint16_t unit_current;
  /* channel apd, unit is 10mV(millivolt) */
  uint16_t apd_bias_feedback[4];
  uint16_t accel_x;
  uint16_t accel_y;
  uint16_t accel_z;
  uint16_t gyro_x;
  uint16_t gyro_y;
  uint16_t gyro_z;
  int32_t accel_unit_x;
  int32_t accel_unit_y;
  int32_t accel_unit_z;
  int32_t gyro_unit_x;
  int32_t gyro_unit_y;
  int32_t gyro_unit_z;
  uint16_t gyro_temp;
  uint16_t reserved[20];
} InnoStatusSensorReadings;
DEFINE_INNO_COMPACT_STRUCT_END

/*
 * Status packet definition
 */
typedef DEFINE_INNO_COMPACT_STRUCT(InnoStatusPacket) {
  InnoCommonHeader common;

  uint64_t idx;  /* global index of all InnoStatusPacket */

  uint8_t status_packet_interval_ms;  /* status packet send interval in ms    */
  uint8_t pre_lidar_mode;             /* previous InnoLidarMode               */
  uint16_t in_transition_mode_ms;  /* time (ms), LiDAR in the transition mode */

  char sn[INNO_SN_SZIE];                /* lidar serial number */
  uint16_t fault_version;
  uint16_t ref_count_enough_ts_ms;  // The time required for the number of reference to reach a certain proportion (50%)
                                    // of the trigger count
  uint16_t ref_intensity[INNO_CHANNEL_NUMBER]; /* average reference intensity */
  uint8_t hw_num[INNO_HW_NUMBER_SIZE];
  uint8_t reserved;

  InnoStatusInFaults in_faults;           /* fault id 0~95 */
  InnoStatusExFaults ex_faults;
  InnoStatusCounters counters;
  InnoStatusSensorReadings sensor_readings;
  // The output from the FalconK LiDAR does not include the in_faults2 and extend_reserved field, which is
  // inserted by the client SDK
  InnoStatusInFaults in_faults2;        /* fault id 96~191 */
  uint64_t extend_reserved[8];          /* 64 byte */
} InnoStatusPacket;
DEFINE_INNO_COMPACT_STRUCT_END

/**
 * @brief InnoMessageCallback
 * @param lidar_handle The handle of the lidar that triggered the callback.
 * @param context Callback context passed in inno_lidar_set_callbacks().
 * @param from_remote > 0 means the message is from remote source
 * @param level Severity of message.
 * @param code Error code.
 * @param error_message Error message.
 * @return Void.
 */
typedef void (*InnoMessageCallback)(int lidar_handle,
                                    void *context,
                                    uint32_t from_remote,
                                    enum InnoMessageLevel level,
                                    enum InnoMessageCode code,
                                    const char *error_message);


/**
 * @brief InnoDataPacketCallback
 * @param lidar_handle The handle of the lidar that triggered the callback.
 * @param context Callback context passed in inno_lidar_set_callbacks().
 * @param data Pointer to InnoDataPacket
 * @return 0
 */
typedef int (*InnoDataPacketCallback)(int lidar_handle,
                                      void *context,
                                      const InnoDataPacket *data);

/**
 * @brief InnoStatusPacketCallback
 * @param lidar_handle The handle of the lidar that triggered the callback.
 * @param context Callback context passed in inno_lidar_set_callbacks().
 * @param status Pointer to InnoDataPacket
 * @return 0
 */
typedef int (*InnoStatusPacketCallback)(int lidar_handle,
                                        void *context,
                                        const InnoStatusPacket *status);

/**
 * @brief InnoHosttimeCallback
 * @param context Callback context passed in inno_lidar_set_callbacks().
 * @return Return the unix time in second defined in
 *         https://en.wikipedia.org/wiki/Unix_time
 *         e.g. ros::Time::now().toSec();
 */
typedef double (*InnoHosttimeCallback)(void *context);

/**
 * @brief InnoStatusPacketCallback
 * @param lidar_handle The handle of the lidar that triggered the callback.
 * @param context Callback context passed in inno_lidar_set_recorder_callback().
 * @param buffer Pointer to the buffer
 * @param len Length of the buffer
 * @return 0: Normally callback this function in the next time, none 0 : stop recorder
 */
typedef int (*InnoRecorderCallback)(int lidar_handle, void *context, enum InnoRecorderCallbackType type,
                                    const char *buffer, int len);

#endif  // SDK_COMMON_INNO_LIDAR_PACKET_H_
