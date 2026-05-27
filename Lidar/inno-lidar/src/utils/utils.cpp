/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/utils.h"

#include <fcntl.h>
#if !(defined(__MINGW64__) || defined(_WIN32))
#include <netinet/in.h>
#include <dirent.h>
#else
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#include <winsock2.h>
#endif
#endif
#include <sys/types.h>
#include <sys/stat.h>

#if defined(__i386__) || defined(__x86_64__)
#if !(defined(__MINGW64__) || defined(_WIN32))
#include <x86intrin.h>
#else
#include <intrin.h>
#endif
#endif

#ifdef _QNX_
#define SCHED_IDLE 5
#endif

#if defined(_WIN32) && defined(HAVE_SSE42)
#include <nmmintrin.h>
#endif
#include <vector>

#include "utils/log.h"
#include "utils/net_manager.h"
#include "sdk_common/inno_lidar_packet.h"
namespace innovusion {

const char *InnoUtils::white_space_ = "\t\n\v\f\r ";
const char* InnoUtils::kInnoSeparator = "\a\a";

void InnoUtils::set_self_thread_priority(int priority) {
#if !defined(__MINGW64__) && defined(_WIN32)
  int currentPriority = GetThreadPriority(GetCurrentThread());
  int newPriority = THREAD_PRIORITY_TIME_CRITICAL;
  if (!SetThreadPriority(GetCurrentThread(), newPriority)) {
    inno_log_warning_errno("SetThreadPriority(%d)", newPriority);
    return;
  }
  currentPriority = GetThreadPriority(GetCurrentThread());
  if (currentPriority != newPriority) {
    inno_log_error("current priority=%d, target is %d",
                     currentPriority, newPriority);
  } else {
    inno_log_info("set thread priority to %d", currentPriority);
  }
#else
#if !(defined(__MINGW64__) || defined(__APPLE__))
  struct sched_param params;
  struct sched_param current_params;
  int policy;
  int current_policy;
  pthread_t this_thread = pthread_self();

  int ret = pthread_getschedparam(this_thread, &current_policy,
                                  &current_params);
  if (ret) {
    inno_log_error_errno("getschedparam %d", ret);
    return;
  } else {
    inno_log_trace("thread current priority is %d (%d), target is %d",
                   current_params.sched_priority, current_policy,
                   priority);
  }
  if (priority == 0) {
    return;
  } else if (priority > 0) {
    policy = SCHED_FIFO;
    params.sched_priority = current_params.sched_priority + priority;
  } else {
    policy = SCHED_IDLE;
    params.sched_priority = 0;
  }
  if (params.sched_priority > 99) {
    params.sched_priority = 99;
  }
  if (params.sched_priority < 0) {
    params.sched_priority = 0;
  }
  ret = pthread_setschedparam(this_thread, policy, &params);
  if (ret != 0) {
    inno_log_warning_errno("setschedparam(%d)", params.sched_priority);
    return;
  }
  ret = pthread_getschedparam(this_thread, &current_policy,
                              &current_params);
  if (ret) {
    inno_log_error_errno("getschedparam 2 %d", ret);
  } else {
    if (current_params.sched_priority != params.sched_priority) {
      inno_log_error("current priority=%d (%d), target is %d",
                     current_params.sched_priority, current_policy,
                     params.sched_priority);
    } else {
      inno_log_info("set thread priority to %d (%d)",
                    current_params.sched_priority, current_policy);
    }
  }
#endif
#endif
  return;
}

#ifdef __MINGW64__
__inline static uint32_t __attribute__((__gnu_inline__,
                                        __always_inline__,
                                        __artificial__))
_mm_crc32_u8(uint32_t __C, uint8_t __V) {
  return __builtin_ia32_crc32qi(__C, __V);
}

__inline static uint32_t __attribute__((__gnu_inline__,
                                        __always_inline__,
                                        __artificial__))
_mm_crc32_u16(uint32_t __C, uint16_t __V) {
  return __builtin_ia32_crc32hi(__C, __V);
}

__inline static uint32_t __attribute__((__gnu_inline__,
                                        __always_inline__,
                                        __artificial__))
_mm_crc32_u32(uint32_t __C, uint32_t __V) {
  return __builtin_ia32_crc32si(__C, __V);
}
#endif

#if defined (_QNX_) || defined (_WIN32) || defined(__amd64__)

#ifdef HAVE_SSE42
uint32_t InnoUtils::crc32_do(uint32_t crc, const void *const buf,
                             const size_t buf_len) {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(buf);
  size_t bytes = buf_len;
  // Calculate the crc for the data
  for (size_t i = 0; i < (buf_len / sizeof(uint32_t)); ++i) {
    crc = _mm_crc32_u32(crc, *(const uint32_t *)data);
    data += sizeof(uint32_t);
  }
  // Handle remaining bytes: check if there are remaining 16-bit data blocks
  if (buf_len & sizeof(uint16_t)) {
    crc = _mm_crc32_u16(crc, *(const uint16_t *)data);
    data += sizeof(uint16_t);
  }
  // Handle remaining bytes: check if there are remaining 8-bit data blocks
  if (buf_len & sizeof(uint8_t)) {
    crc = _mm_crc32_u8(crc, *(const uint8_t *)data);
  }
  // crc ^= 0xFFFFFFFF;
  return crc;
}

#else
/* from:
https://github.com/andry81/
tacklelib/blob/34ae9f73f5fb209ad8a5d7e9817e90979680efc6/
include/tacklelib/utility/crc_tables.hpp
Table of g_crc32_1EDC6F41 */
static uint64_t crctable[256] = {
  0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
  0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
  0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
  0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
  0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
  0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
  0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
  0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
  0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
  0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
  0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
  0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
  0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
  0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
  0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
  0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
  0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
  0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
  0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
  0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
  0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
  0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
  0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
  0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
  0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
  0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
  0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
  0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
  0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d, 0xef087a76, 0x1d63f975,
  0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
  0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
  0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
  0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
  0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
  0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
  0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
  0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
  0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
  0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
  0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
  0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
  0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
  0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351};


uint32_t InnoUtils::crc32_do(uint32_t crc, const void *const in_buf, const uint64_t in_buf_len) {
  const uint8_t *data = (const uint8_t *)(in_buf);
  uint64_t bytes = in_buf_len;
  // Calculate the crc for the data
  while (bytes >= 4) {
    // Calculate the crc for 4 bytes
    crc ^= (uint32_t)(*data++) & 0xFFU;
    crc ^= ((uint32_t)(*data++) & 0xFFU) << 8;
    crc ^= ((uint32_t)(*data++) & 0xFFU) << 16;
    crc ^= ((uint32_t)(*data++) & 0xFFU) << 24;

    // Update the crc
    crc = crctable[crc & 0xFFU] ^ (crc >> 8);
    crc = crctable[crc & 0xFFU] ^ (crc >> 8);
    crc = crctable[crc & 0xFFU] ^ (crc >> 8);
    crc = crctable[crc & 0xFFU] ^ (crc >> 8);

    bytes -= 4;
  }

  // Handle any remaining bytes
  while (bytes != 0) {
    bytes--;
    crc = crctable[(crc ^ (uint32_t)(*data++)) & 0xFFU] ^ (crc >> 8);
  }
  return crc;
}
#endif
#else

uint32_t InnoUtils::crc32_do(uint32_t crc, const void *const buf,
                             const size_t buf_len) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
#if !(defined(__APPLE__))
  register uint32_t l = crc;
  #else
  uint32_t l = crc;
  #endif

#if (defined(__i386__) || defined(__x86_64__)) && (!defined(__MINGW64__dd)) \
  && (!defined(__APPLE__))
  for (size_t i = 0; i < (buf_len / sizeof(uint32_t)); ++i) {
    l = _mm_crc32_u32(l, *(const uint32_t *)p);
    p += sizeof(uint32_t);
  }
  if (buf_len & sizeof(uint16_t)) {
    l = _mm_crc32_u16(l, *(const uint16_t *)p);
    p += sizeof(uint16_t);
  }
  if (buf_len & sizeof(uint8_t)) {
    l = _mm_crc32_u8(l, *(const uint8_t *)p);
  }
#else

#if defined(__aarch64__)
  // from https://www.programmersought.com/article/13506713080/
#define CRC32X(crc, value) __asm__("crc32x %w[c], %w[c], %x[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32W(crc, value) __asm__("crc32w %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32H(crc, value) __asm__("crc32h %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32B(crc, value) __asm__("crc32b %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32CX(crc, value) __asm__("crc32cx %w[c], %w[c], %x[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32CW(crc, value) __asm__("crc32cw %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32CH(crc, value) __asm__("crc32ch %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
#define CRC32CB(crc, value) __asm__("crc32cb %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))  // NOLINT
  // Use local variables, use register optimization
#if !(defined(__APPLE__))
  register size_t len = buf_len;
#else
  size_t len = buf_len;
#endif
#define STEP1 do {                                              \
    CRC32CB(l, *p++);                                           \
    len--;                                                      \
} while (0)

#define STEP2 do {                                              \
    CRC32CH(l, *(uint16_t *)p);                                 \
    p += 2;                                                     \
    len -= 2;                                                   \
} while (0)

#define STEP4 do {                                              \
    CRC32CW(l, *(uint32_t *)p);                                 \
    p += 4;                                                     \
    len -= 4;                                                   \
} while (0)

#define STEP8 do {                                              \
    CRC32CX(l, *(uint64_t *)p);                                 \
    p += 8;                                                     \
    len -= 8;                                                   \
} while (0)

  // 512 way loop inline expansion
  while (len >= 512) {
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
  }
  // Use if to judge directly, the effect will be higher
  if (len >= 256) {
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
  }

  if (len >= 128) {
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
  }

  if (len >= 64) {
    STEP8; STEP8; STEP8; STEP8;
    STEP8; STEP8; STEP8; STEP8;
  }

  if (len >= 32) {
    STEP8; STEP8; STEP8; STEP8;
  }

  if (len >= 16) {
    STEP8; STEP8;
  }

  if (len >= 8) {
    STEP8;
  }

  if (len >= 4) {
    STEP4;
  }

  if (len >= 2) {
    STEP2;
  }

  if (len >= 1) {
    STEP1;
  }
#undef STEP8
#undef STEP4
#undef STEP2
#undef STEP1

#else
//  for (size_t i = 0; i < buf_len; ++i) {
//    l = (l << 5) + l + p[i];
//  }
#error "unsupported ARCH"

#endif  // ARM
#endif
  return l;
}
#endif

uint32_t InnoUtils::calculate_http_crc32(const char* buffer,
                                         uint32_t length,
                                         bool append) {
  uint32_t crc = InnoUtils::crc32_start();
  crc = InnoUtils::crc32_do(crc, buffer, length);
  if (append) {
    // add calc BEL 0x07 for separator
    crc = crc32_do(crc, kInnoSeparator, 2);
  }
  return crc32_end(crc);
}

int InnoUtils::verify_http_crc32(const char* buffer,
                                 const char* url) {
  std::string recv_buf(buffer);
  std::string recv_crc32;
  std::string context;
  uint32_t calc_crc32 = crc32_start();
  char ch_crc32[20] = {0};

  uint32_t buf_len = strlen(buffer);
  size_t pos = recv_buf.find("X-INNO-CRC32");
  if (pos != recv_buf.npos) {
    recv_crc32 = recv_buf.substr(pos + kInnoStrCRC32len,
                                 kInnoCRC32ValueLen);
    pos = recv_buf.find("\r\n\r\n");
    if (pos != recv_buf.npos) {
      context = recv_buf.substr(pos + 4, buf_len - pos - 4);
      if (url != NULL) {
        calc_crc32 = crc32_do(calc_crc32, url, strlen(url));
        // add calc BEL 0x07 for separator
        calc_crc32 = crc32_do(calc_crc32, kInnoSeparator, 2);
      }
      if (context.size() > 0) {
        calc_crc32 = crc32_do(calc_crc32, context.c_str(),
                              context.size());
      }
      calc_crc32 = crc32_end(calc_crc32);
      snprintf(ch_crc32, sizeof(ch_crc32), "%8x", calc_crc32);
      if (strcmp(ch_crc32, recv_crc32.c_str()) != 0) {
        inno_log_warning("CRC check failed. "
                         "calc_crc32 %s != recv_crc32 %s",
                         ch_crc32, recv_crc32.c_str());
        return -1;
      } else {
        return 0;
      }
    } else {
      inno_log_warning("Can't find \r\n\r\n");
      return -1;
    }
  } else {
    return 1;   // not check crc32 for Web
  }
}

int InnoUtils::verify_lidar_version(const char *buffer) {
  std::string recv_buf(buffer);
  uint32_t major_version = 0;
  uint32_t minor_version = 0;
  uint32_t buf_len = strlen(buffer);
  size_t pos = recv_buf.find("X-INNO-MAJOR-VERSION");

  if (pos != recv_buf.npos) {
    sscanf(recv_buf.c_str() + pos + sizeof("X-INNO-MAJOR-VERSION:"), "%u", &major_version);
    if (major_version > kInnoMajorVersionDataPacket) {
      inno_log_error("please upgrade client sdk, lidar protocol major version:%u sdk major version:%u", major_version,
                     kInnoMajorVersionDataPacket);
      return -1;
    } else if (major_version == kInnoMajorVersionDataPacket) {
      pos = recv_buf.find("X-INNO-MINOR-VERSION");
      sscanf(recv_buf.c_str() + pos + sizeof("X-INNO-MINOR-VERSION:"), "%u", &minor_version);
      if (minor_version > kInnoMinorVersionDataPacket) {
        inno_log_warning("please upgrade client sdk, lidar protocol minor version:%u sdk minor version:%u",
                         minor_version, kInnoMinorVersionDataPacket);
      }
    }
  }
  return 0;
}

/**
 * @Brief : check if the ip is valid
 * @param  ip
 * @return true  ip valid
 * @return false ip invalid
 */
bool InnoUtils::check_ip_valid(const char *ip) {
  std::string ip_address(ip);
  if (ip_address.empty() || ip_address == kInvalidIpAddress) {
    return false;
  }

  if (ip_address == kDefaultInterface)
    return true;

  struct in_addr new_ip;
  if (NetManager::inno_inet_pton(ip_address.c_str(), &new_ip) <= 0) {
    return false;
  }

  return true;
}

int InnoUtils::open_file(const char *filename, int flag_in, int mode) {
  int flag = flag_in;
#if defined(__MINGW64__) || defined(_WIN32)
  flag |= O_BINARY;
#endif
  int file_fd = open(filename, flag, mode);
  if (file_fd < 0) {
    inno_log_info("cannot open %s", filename);
    const char *ppwd = getenv("PWD");
    if (ppwd != NULL) {
      std::string pwd = ppwd;
      std::string file_fullpath = pwd + "/" + filename;
      file_fd = open(file_fullpath.c_str(), mode);
      if (file_fd < 0) {
        inno_log_error_errno("cannot open %s", file_fullpath.c_str());
        return -1;
      } else {
        // pwd+file opened
        inno_log_info("open %s", file_fullpath.c_str());
        return file_fd;
      }
    } else {
      inno_log_error("cannot get pwd, cannot open file");
      return -3;
    }
  } else {
    inno_log_info("open %s", filename);
  }
  return file_fd;
}

#if defined (__MINGW64__) || defined(_WIN32)
bool InnoUtils::is_socket_fd(int fd) {
  struct sockaddr_in addr;
  int ilen = sizeof(addr);
  int status = getsockname(fd, (struct sockaddr*)&addr, &ilen);
  if (status == 0) {
    return true;
  } else {
    return false;
  }
}
#endif

int InnoUtils::close_fd(sock_t fd) {
#if !(defined(__MINGW64__) || defined(_WIN32))
  return close(fd);
#else
  if (is_socket_fd(fd)) {
    return closesocket(fd);
  } else {
    inno_log_info("close file, fd = %d", fd);
#if !defined (__MINGW64__) && defined(_WIN32)
    return closesocket(fd);
#else
    return close(fd);
#endif
  }
#endif
}

int InnoUtils::list_file(const std::string &path,
                         const std::string &pattern,
                         std::vector<std::string> *ret) {
#ifndef _WIN32
  DIR *dp = ::opendir(path.c_str());
  if (dp == nullptr) {
    inno_log_error("open dir %s error", path.c_str());
    return -1;
  }
  struct dirent *entry;
  while ((entry = ::readdir(dp)) != nullptr) {
    if (strstr(entry->d_name, pattern.c_str())) {
      std::string entry_full_name = path +
        (InnoUtils::ends_with(path.c_str(), "/") ? "" : "/") + entry->d_name;
      ret->push_back(entry_full_name);
    }
  }
  ::closedir(dp);
#endif
  return 0;
}

std::string InnoUtils::get_current_time_str(const std::string& format) {
  time_t rawtime;
  struct tm *info;
  char temp[80];
  struct tm result_time;

  time(&rawtime);
#if !(defined(__MINGW64__) || defined(_WIN32))
  info = localtime_r(&rawtime, &result_time);
#else
  info = localtime_s(&result_time, &rawtime) == 0 ?
         &result_time : NULL;
#endif
  strftime(temp, sizeof(temp), format.c_str(), info);
  return std::string(temp);
}

/**
 * 1. Create a SOCK_DGRAM socket(AF_INET, SOCK_DGRAM, 0)
 * 2. set reuse addr
 * 3. bind to input port
 * 4. set input opts
 * @param port
 * @param opts
 * @return -1 if failed. socket_fd if success.
 */
int InnoUdpHelper::bind(uint16_t port,
                        const std::vector<InnoUdpOpt> &opts) {
  int socket_fd = -1;
  struct sockaddr_in udp_listener_addr;
  if ((socket_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    inno_log_error_errno("udp listener socket creation error %d", socket_fd);
    return -1;
  }

#if !(defined(__MINGW64__) || defined(_WIN32))
  int reuse = 1;
#else
  char reuse = 1;
#endif
  if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR,
                 &reuse, sizeof(reuse)) < 0) {
    inno_log_error("udp listener set reuse address error");
    InnoUtils::close_fd(socket_fd);
    return -1;
  }

  memset(&udp_listener_addr, 0, sizeof(udp_listener_addr));
  udp_listener_addr.sin_family = AF_INET;
  udp_listener_addr.sin_addr.s_addr = INADDR_ANY;
  udp_listener_addr.sin_port = htons(port);
  if (::bind(socket_fd, (const sockaddr*)&udp_listener_addr,
           sizeof(udp_listener_addr)) < 0) {
    inno_log_error_errno("failed to bind to port:%d", port);
    InnoUtils::close_fd(socket_fd);
    return -1;
  }

  for (auto & opt : opts) {
    int rs = setsockopt(socket_fd, opt.level, opt.optname,
                        reinterpret_cast<const char *>(opt.optval),
                        opt.optlen);
    if (rs < 0) {
      inno_log_info("opt: %d, %d, %p, %u, %s",
                    opt.level, opt.optname,
                    opt.optval, opt.optlen, opt.optname_str);
      inno_log_error_errno("setsockopt %s error %hu", opt.optname_str, port);
      InnoUtils::close_fd(socket_fd);
      return -1;
    }
  }

#if (defined(__MINGW64__) || defined(_WIN32))
  int timeout = 500;  // 0.5 second
  setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof(timeout));
#endif

  return socket_fd;
}
}  // namespace innovusion
