/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

#if defined(__MINGW64__) || defined(_WIN32)
#include <ws2tcpip.h>
#endif

#if defined(_QNX_) || defined(__APPLE__)
#include <sys/socket.h>
#endif

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <string>
#include <vector>
#include <utility>

#include "utils/log.h"

namespace innovusion {

#if !defined(__MINGW64__) && defined(_WIN32)
#undef max
#undef min
#include "Ws2tcpip.h"
#include "winsock2.h"  // almost everything is contained in one file.

#ifndef socklen_t
typedef int socklen_t;
#endif
typedef int pthread_t;
typedef SOCKET sock_t;  // Although sockets are int's on unix,
                        // windows uses it's own typedef of
                        // SOCKET to represent them. If you look
                        // in the Winsock2 source code, you'll see
                        // that it is just a typedef for int, but
                        // there is absolutely no garuntee that it
                        // won't change in a later version.
                        // therefore, in order to avoid confusion,
                        // this library will create it's own basic
                        // socket descriptor typedef

#else  // UNIX/Linux
typedef int sock_t;  // see the description above.

#endif

#if !defined(__MINGW64__) && defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <unistd.h>

#endif

#if defined(_QNX_) || defined(__MINGW64__)
#define CLOCK_MONOTONIC_RAW (CLOCK_REALTIME)
#endif

// A macro to disallow the copy constructor and operator= functions
// This should be used in the priavte:declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &);              \
  TypeName &operator=(const TypeName &)

#define SAFE_COPY_FIX_STRING_ARRAY(src, dest)                                         \
  do {                                                                                \
    inno_log_verify(src &&strlen(src) && strlen(src) < sizeof(dest), "invalid copy"); \
    strncpy(dest, src, sizeof(dest));                                                 \
    dest[sizeof(dest) - 1] = 0;                                                       \
  } while (0)

const char *const kInvalidIpAddress = "0.0.0.0";
const char *const kDefaultInterface = "eth0";

/**
 * @brief InnoUtils
 */
class InnoUtils {
 private:
  static const char *white_space_;

 public:
  static const uint32_t kInnoCRC32ValueLen = 8;
  static const uint32_t kInnoStrCRC32len = 14;
  static const char *kInnoSeparator;

 public:
  /**
   * @brief Split string with the specified spliter
   * @param tokens  Vector to store the splited string
   * @param str     String to be splited
   * @param spliter Delimiter to split string
   * @return Return the count of splited string
   */
  static int splitby(std::vector<std::string> &tokens, const std::string &str, const char *spliter) {
    std::string::size_type last = 0;
    std::string::size_type pos = 0;
    int count = 0;
    while (std::string::npos != (pos = str.find_first_of(spliter, last))) {
      auto partion = str.substr(last, pos - last);
      last = pos + strlen(spliter);
      if (!partion.empty()) {
        tokens.emplace_back(partion);
        count++;
      }
    }
    auto partion = str.substr(last);
    if (!partion.empty()) {
      tokens.emplace_back(partion);
      count++;
    }

    return count;
  }

  /**
   * @brief  Split string with the specified spliter
   * @tparam ...ARGS Spliter types
   * @param tokens  Vector to store the splited string
   * @param str     String to be splited
   * @param spliter Delimiter to split string
   * @param ...spliters Delimiter to split string
   * @return Return the count of splited string
   */
  template <typename... ARGS>
  static int splitby(std::vector<std::string> &tokens, const std::string &str, const char *spliter, ARGS... spliters) {
    int count = 0;
    std::string::size_type last = 0;
    std::string::size_type pos = 0;
    while (std::string::npos != (pos = str.find_first_of(spliter, last))) {
      auto partion = str.substr(last, pos - last);
      last = pos + strlen(spliter);
      if (!partion.empty()) {
        count += splitby(tokens, partion, spliters...);
      }
    }
    auto partion = str.substr(last);
    if (!partion.empty()) {
      count += splitby(tokens, partion, spliters...);
    }

    return count;
  }

  /**
   * @brief Convert string to array
   * @tparam TABLE Array type
   * @tparam CONVEROTR Convertor type
   * @param table Array to store the converted result
   * @param str   String to be converted
   * @param rows  Array rows
   * @param cols  Array cols
   * @param convertor Convertor to convert string to array element
   */
  template <typename TABLE, typename CONVEROTR>
  static void string_to_table(TABLE &table, const std::string &str, int rows, int cols, CONVEROTR &&convertor) {
    std::vector<std::string> tokens;
    splitby(tokens, str, "\n", " ", "\t", ";");

    if (size_t(cols * rows) != tokens.size()) {
      inno_log_error("invalid string_to_table, rows: %d, cols: %d, tokens: %lu", rows, cols, tokens.size());
      return;
    }

    auto token = tokens.begin();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        convertor(table, row, col, *token);
        token++;
      }
    }
  }

  /**
   * @brief Trim left of the string with the specified chars
   * @param str   String to be trimmed
   * @param chars Specific chars to be trimmed
   * @return Return trimmed string
   */
  static inline std::string *ltrim(std::string *str, const char *chars) {
    str->erase(0, str->find_first_not_of(chars));
    return str;
  }
  /**
   * @brief Trim right of the string with the specified chars
   * @param str   String to be trimmed
   * @param chars Specific chars to be trimmed
   * @return Return trimmed string
   */
  static inline std::string *rtrim(std::string *str, const char *chars) {
    str->erase(str->find_last_not_of(chars) + 1);
    return str;
  }

  /**
   * @brief Trim left and right of the string with the specified chars
   * @param str   String to be trimmed
   * @param chars Specific chars to be trimmed
   * @return Return trimmed string
   */
  static inline std::string *trim(std::string *str, const char *chars) {
    return ltrim(rtrim(str, chars), chars);
  }

  /**
   * @brief Trim left and right space of the string
   * @param str String to be trimmed
   * @return Return trimmed string
   */
  static inline std::string *trim_space(std::string *str) {
    return trim(str, white_space_);
  }

  /**
   * @brief Remove all space of the string
   * @param str String to be processed
   * @return Return processed string with no space
   */
  static std::string *remove_all_space(std::string *str) {
    return remove_all_char(str, ' ');
  }

  /**
   * @brief Remove specified char of the string
   * @param str String to be processed
   * @param c   Specific char to be removed
   * @return Return processed string with no specified char
   */
  static std::string *remove_all_char(std::string *str, const char c) {
    std::string::iterator end_pos = std::remove(str->begin(), str->end(), c);
    str->erase(end_pos, str->end());
    return str;
  }

  /**
   * @brief Remove specified chars of the string
   * @param str   String to be processed
   * @param chars Specific chars to be removed
   * @return Return processed string with no specified chars
   */
  static std::string *remove_all_chars(std::string *str, const char *chars) {
    int i = 0;
    while (chars[i] != '\0') {
      remove_all_char(str, chars[i]);
      i++;
    }
    return str;
  }

  /**
   * @brief Split string with the specified delimiter
   * @param raw_str   String to be split
   * @param delimiter Delimiter to split string
   * @return Return splited string in vector
   */
  static std::vector<std::string> split(std::string raw_str, const std::string &delimiter) {
    size_t pos = 0;
    std::string token;
    std::vector<std::string> splited_strings;
    while ((pos = raw_str.find(delimiter)) != std::string::npos) {
      token = raw_str.substr(0, pos);
      splited_strings.push_back(token);
      raw_str.erase(0, pos + delimiter.length());
    }
    splited_strings.push_back(raw_str);
    return splited_strings;
  }

  /**
   * @brief Check if the string is unsigned decimal integer
   * @param str String to be checked
   * @return Return true if the string is unsigned decimal integer, otherwise return false
   */
  static inline bool is_unsinged_decimal_integer(const std::string &str) {
    return !str.empty() && str.find_first_not_of("0123456789") == std::string::npos;
  }

  /**
   * @brief Set current thread priority
   * @param priority Thread priority to be set
   */
  static void set_self_thread_priority(int priority);

#if defined(__MINGW64__) || !defined(_WIN32)
  /**
   * @brief Get time in nanoseconds by given clock id
   * @param clk_id Clock id
   * @return Return time in nanoseconds
   */
  static inline uint64_t get_time_ns(clockid_t clk_id) {
    struct timespec spec;
    clock_gettime(clk_id, &spec);
    return spec.tv_sec * 1000000000L + spec.tv_nsec;
  }

  /**
   * @brief Get time in microseconds by given clock id
   * @param clk_id Clock id
   * @return Return time in microseconds
   */
  static inline uint64_t get_time_us(clockid_t clk_id) {
    return get_time_ns(clk_id) / 1000;
  }

  /**
   * @brief Get time in milliseconds by given clock id
   * @param clk_id Clock id
   * @return Return time in milliseconds
   */
  static inline uint64_t get_time_ms(clockid_t clk_id) {
    return get_time_ns(clk_id) / 1000000;
  }
#endif

  /**
   * @brief Get system time in nanoseconds
   * @return Return system time in nanoseconds
   */
  static inline uint64_t get_time_ns(void) {
#if defined(__MINGW64__) || !defined(_WIN32)
    return get_time_ns(CLOCK_MONOTONIC_RAW);
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
  }

  /**
   * @brief Get system time in microseconds
   * @return Return system time in microseconds
   */
  static inline uint64_t get_time_us(void) {
#if defined(__MINGW64__) || !defined(_WIN32)
    return get_time_us(CLOCK_MONOTONIC_RAW);
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count() * 1000;
#endif
  }

  /**
   * @brief Get system time in milliseconds
   * @return Return system time in milliseconds
   */
  static inline uint64_t get_time_ms(void) {
#if defined(__MINGW64__) || !defined(_WIN32)
    return get_time_ms(CLOCK_MONOTONIC_RAW);
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count() * 1000000;
#endif
  }

  /**
   * @brief Convert microseconds to timespec
   * @param us   Microseconds to be converted
   * @param spec Timespec to store converted result
   */
  static void us_to_timespec(uint64_t us, struct timespec *spec) {
    spec->tv_sec = us / 1000000;
    spec->tv_nsec = (us % 1000000) * 1000;
  }

  /**
   * @brief Check if the string ends with the specified suffix
   * @param str    String to be checked
   * @param suffix Specified suffix
   * @return Return true if the string ends with the specified suffix, otherwise return false
   */
  static bool ends_with(const char *str, const char *suffix) {
    if (!str || !suffix) {
      return false;
    }
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr) {
      return false;
    }
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
  }

  /**
   * @brief Check if the string starts with the specified prefix
   * @param str    String to be checked
   * @param prefix Specified prefix
   * @return Return true if the string starts with the specified prefix, otherwise return false
   */
  static bool start_with(const char *str, const char *prefix) {
    if (!str || !prefix) {
      return false;
    }
    size_t lenstr = strlen(str);
    size_t lenprefix = strlen(prefix);
    if (lenprefix > lenstr) {
      return false;
    }
    return strncmp(str, prefix, lenprefix) == 0;
  }

  /**
   * @brief Get initial crc32 value
   * @return Return the initial crc32 value
   */
  static inline uint32_t crc32_start() {
    return 0xffffffffu;
  }

  /**
   * @brief Get final crc32 value
   * @param crc CRC32 value to be finalized
   * @return Return the final crc32 value
   */
  static inline uint32_t crc32_end(uint32_t crc) {
    return crc ^ 0xffffffffu;
  }

  /**
   * @brief Calculate crc32 value with given buffer
   * @param crc A initialized crc32 value
   * @param buf Buffer to be calculated
   * @param len Buffer length
   * @return Return the calculated crc32 value
   */
  static uint32_t crc32_do(uint32_t crc, const void *buf, const size_t len);

  /**
   * @brief Get http buffer crc32
   * @param buffer    Buffer received from network
   * @param length    Buffer length
   * @param append    True to calculate crc32 with separator
   * @return Return crc32
   */
  static uint32_t calculate_http_crc32(const char *buffer, uint32_t length, bool append = false);

  /**
   * @brief Check if the http buffer crc32 is valid
   * @param buffer Http buffer to be checked
   * @param url    Url of the http buffer
   * @return Return 0 if the crc32 is valid, others for error
   */
  static int verify_http_crc32(const char *buffer, const char *url);

  /**
   * @brief Check lidar protocol version
   * @param buffer Http buffer to be checked
   * @return Return 0 if the version is match, others for error
   */
  static int verify_lidar_version(const char *buffer);
  /**
   * @brief Open file with given filename, flag and mode
   * @param filename Filename
   * @param flag_in  Flag
   * @param mode     Mode
   * @return Return positive fd if success, others for error
   */
  static int open_file(const char *filename, int flag_in, int mode);

  /**
   * @brief Check if valid ip
   * @param ip Ip to be checked
   * @return Return true if valid, otherwise return false
   */
  static bool check_ip_valid(const char *ip);

#if defined(__MINGW64__) || defined(_WIN32)
  /**
   * @brief Check if a valid socket fd
   * @param fd Fd to be checked
   * @return Return true if valid, otherwise return false
   */
  static bool is_socket_fd(int fd);
#endif
  /**
   * @brief Close the given fd
   * @param fd Fd to be closed
   * @return Return 0 for success, others for error
   */
  static int close_fd(sock_t fd);

  /**
   * @brief Get time in nanoseconds by given timespec
   * @param spec The given timespec
   * @return Return time in nanoseconds
   */
  static inline uint64_t get_timestamp_ns(timespec spec) {
    return spec.tv_sec * 1000000000 + spec.tv_nsec;
  }

  /**
   * @brief List all files' full path in the given path filtered by the given pattern
   * @param path    Path to be listed
   * @param pattern Pattern to filter files
   * @param ret     Vector to store the result
   * @return Return 0 for success, others for error
   */
  static int list_file(const std::string &path, const std::string &pattern, std::vector<std::string> *ret);

  /**
   * @brief Get current system time in formated string
   * @param format Format of the time string
   * @return Return the time string
   */
  static std::string get_current_time_str(const std::string &format = "%m_%d_%Y_%H_%M_%S");

  /**
   * @brief Run command and get the output
   * @param cmd      Command to be executed
   * @param buffer   Buffer to store the output
   * @param buf_size Buffer size
   */
  static void run_command(const char *cmd, char *buffer, const int buf_size) {
  #if !defined(__MINGW64__) && defined(_WIN32)
  #define popen _popen
  #define pclose _pclose
  #endif
    inno_log_verify(cmd && buffer && buf_size > 0, "Run command conditions error");
    buffer[0] = '\0';
    FILE *fp = popen(cmd, "r");
    if (fp == NULL) {
      inno_log_error("fail to exec popen(%s)", cmd);
      return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    size_t numread = fread(buffer, 1, buf_size, fp);
    pclose(fp);
    if (numread > 0) {
      buffer[buf_size - 1] = '\0';
    }
  }

  static int print_app_ver() {
#if !defined(__MINGW64__) && defined(_WIN32)
  #include <windows.h>
  #pragma comment(lib, "version.lib")
    TCHAR szFileName[MAX_PATH] = {};
    GetModuleFileName(NULL, szFileName, MAX_PATH);

    DWORD dwSize = GetFileVersionInfoSize(szFileName, NULL);

    if (dwSize == 0) {
      inno_log_error("Failed to get file version info size\n");
      return -1;
    }

    BYTE* pBuffer = new BYTE[dwSize];

    if (!GetFileVersionInfo(szFileName, 0, dwSize, pBuffer)) {
      inno_log_error("not found ver\n");
      // Failed to get file version information.
      delete[] pBuffer;
      return -2;
    }

    VS_FIXEDFILEINFO* pVersionInfo = NULL;
    UINT uLength = 0;

    if (!VerQueryValue(pBuffer, "\\", reinterpret_cast<LPVOID*>(&pVersionInfo), &uLength)) {
      // Failed to get version information of file.
      delete[] pBuffer;
      return -3;
    }

    WORD wFMajor = HIWORD(pVersionInfo->dwFileVersionMS);
    WORD wFMinor = LOWORD(pVersionInfo->dwFileVersionMS);
    WORD wFRevision = HIWORD(pVersionInfo->dwFileVersionLS);
    WORD wFBuild = LOWORD(pVersionInfo->dwFileVersionLS);

    delete[] pBuffer;

    // Output the file version information.
    inno_log_info("version %hu.%hu.%hu.%hu\n", wFMajor, wFMinor, wFRevision, wFBuild);
    return 0;
#else
    return 0;
#endif
  }
};

/**
 * @brief InnoMean
 */
class InnoMean {
 public:
  InnoMean() {
    reset();
  }

  ~InnoMean() {
  }

 public:
  /**
   * @brief Reset the object
   */
  void reset() {
    sum_ = 0;
    sum2_ = 0;
    count_ = 0;
    delta_max_ = 0;
    max_ = 0;
    min_ = 0;
    pre_ = 0;
  }

  /**
   * @brief Get mean value of the given values
   * @return Return mean
   */
  double mean() const {
    uint64_t c = count_;
    double s = sum_;
    if (c > 0) {
      return s / c;
    } else {
      return 0;
    }
  }

  /**
   * @brief Get the count of values added to the object
   * @return Return the count of values
   */
  uint64_t count() const {
    return count_;
  }

  /**
   * @brief Get the standard deviation of the given values
   * @return Return the standard deviation
   */
  double std_dev() const {
    uint64_t c = count_;
    double s = sum_;
    double s2 = sum2_;
    if (c > 0) {
      double a = s / c;
      double d = s2 / c - a * a;
      if (d > 0) {
        return sqrt(d);
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  /**
   * @brief Get the maximum delta value between each neighbor value
   * @return Return the maximum delta value
   */
  double max_delta() const {
    return delta_max_;
  }

  /**
   * @brief Get the maximum value of the given values
   * @return Return the maximum value
   */
  double max() const {
    return max_;
  }

  /**
   * @brief Get the minimum value of the given values
   * @return Return the minimum value
   */
  double min() const {
    return min_;
  }
  /**
   * @brief Get the last value of the given values
   * @return Return the last value
   */
  double pre() const {
    return pre_;
  }

  /**
   * @brief Add value to the object
   * @param value The value to be added
   */
  inline void add(double value) {
    if (count_ > 0) {
      double delta = fabs(value - pre_);
      if (delta_max_ < delta) {
        delta_max_ = delta;
      }
      if (max_ < value) {
        max_ = value;
      }
      if (min_ > value) {
        min_ = value;
      }
    } else {
      max_ = value;
      min_ = value;
    }
    count_++;
    sum_ += value;
    sum2_ += value * value;
    pre_ = value;
  }

 private:
  double delta_max_;
  double max_;
  double min_;
  double pre_;
  double sum_;
  double sum2_;
  uint64_t count_;
};

/**
 * @brief InnoMeanLite
 */
class InnoMeanLite {
 public:
  InnoMeanLite() {
    reset();
  }

  ~InnoMeanLite() {
  }

 public:
  /**
   * @brief Reset the object
   */
  void reset() {
    sum_ = 0;
    sum2_ = 0;
    count_ = 0;
  }

  /**
   * @brief Get the mean value of the given values
   * @return Return the mean value
   */
  double mean() const {
    int64_t c = count_;
    double s = sum_;
    if (c > 0) {
      return s / c;
    } else {
      return 0;
    }
  }

  /**
   * @brief Get the given values count
   * @return Return the count of values
   */
  int64_t count() const {
    return count_;
  }

  /**
   * @brief Get the standard deviation of the given values
   * @return Return the standard deviation
   */
  double std_dev() const {
    int64_t c = count_;
    double s = sum_;
    double s2 = sum2_;
    if (c > 0) {
      double a = s / c;
      double d = s2 / c - a * a;
      if (d > 0) {
        return sqrt(d);
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  /**
   * @brief Add value to the object
   * @param value The value to be added
   */
  inline void add(int64_t value) {
    count_++;
    sum_ += value;
    sum2_ += value * value;
  }

  /**
   * @brief Remove value from the object
   * @param value The value to be removed
   */
  inline void subtract(int64_t value) {
    count_--;
    sum_ -= value;
    sum2_ -= value * value;
  }

  /**
   * @brief Get the sum of the given values
   * @return Return the sum
   */
  inline int sum() {
    return sum_;
  }

 private:
  int64_t sum_;
  int64_t sum2_;
  int64_t count_;
};

/**
 * @brief InnoSlidingMean
 */
class InnoSlidingMean {
 public:
  explicit InnoSlidingMean(int max_q_len = 15) {
    max_q_len_ = max_q_len;
  }

  ~InnoSlidingMean() {
  }

  /**
   * @brief Append value to the object with a sliding window
   * @param value The value to be appended
   */
  inline void add(int64_t value) {
    mean_.add(value);
    q_.push(value);
    while (q_.size() > max_q_len_) {
      int64_t front = q_.front();
      q_.pop();
      mean_.subtract(front);
    }
  }

  /**
   * @brief Remove the oldest value from the object
   */
  inline void pop() {
    if (q_.empty()) {
      return;
    }
    int front = q_.front();
    q_.pop();
    mean_.subtract(front);
  }

  /**
   * @brief Get the mean value of the given values
   * @return Return the mean value
   */
  inline double mean() const {
    return mean_.mean();
  }

  /**
   * @brief Get the standard deviation of the given values
   * @return Return the standard deviation
   */
  inline double std_dev() {
    return mean_.std_dev();
  }

  /**
   * @brief Set the maximum length of the sliding window
   * @param len The maximum length of the sliding window
   */
  inline void set_max_q_len(int len) {
    max_q_len_ = len;
  }

  /**
   * @brief Get the sliding window length
   * @return Return the sliding window length
   */
  inline int size() {
    return q_.size();
  }

  /**
   * @brief Check if the sliding window is full
   * @return Return true if the sliding window is full, otherwise return false
   */
  inline bool is_full() {
    return q_.size() >= max_q_len_;
  }

  /**
   * @brief Check if the sliding window is empty
   * @return Return true if the sliding window is empty, otherwise return false
   */
  inline bool is_empty() {
    return q_.empty();
  }

  /**
   * @brief Remove all values from the object and reset
   */
  inline void clear() {
    while (!q_.empty()) {
      q_.pop();
    }
    mean_.reset();
  }
  /**
   * @brief Get the sum of the given values
   * @return Return the sum
   */
  inline int sum() {
    return mean_.sum();
  }

 private:
  std::queue<int> q_;
  size_t max_q_len_;
  InnoMeanLite mean_;
};

class InnoUdpOpt {
 public:
  InnoUdpOpt(int level, int optname, const void *optval, socklen_t optlen, const char *optname_str) {
    this->level = level;
    this->optname = optname;
    this->optval = optval;
    this->optlen = optlen;
    this->optname_str = optname_str;
  }

 public:
  int level;
  int optname;
  const void *optval;
  socklen_t optlen;
  const char *optname_str;
};

class InnoUdpHelper {
 public:
  static int bind(uint16_t port, const std::vector<InnoUdpOpt> &opts);
  static int bind(const char *ip, uint16_t port, const std::vector<InnoUdpOpt> &opts);
  static bool is_multicast_ip_addr(std::string ip) {
    std::vector<std::string> ip_segs = InnoUtils::split(ip, ".");
    if (ip_segs.size() != 4) {
      return false;
    }
    int i = atoi(InnoUtils::split(ip, ".")[0].c_str());
    return i >= 224 && i <= 239;
  }
};

// std::string_view is not supported until C++17
// this implementation is not compatible with std::string_view, it has been adapted to our needs
class StrView {
 public:
  StrView() = default;
  // data end with '\0'
  explicit StrView(const char *data) : data_(data), len_(0), length_valid_(false) {}
  // user promise that data and len are valid
  StrView(const char *data, uint64_t len) : data_(data), len_(len), length_valid_(true) {}

 public:
  operator bool() const {
    return data_ != nullptr && ((length_valid_ && len_ > 0) || (!length_valid_ && *data_ != '\0'));
  }
  operator std::string() const {
    return std::string(data_, len_);
  }

  char operator[](uint64_t pos) const {
    return data_[pos];
  }

 public:
  // ignore buffer length, check end by '\0'
  bool getline(StrView &str_view, char delim = '\n') {
    if (*data_ != '\0') {
      const char *p = data_;
      while (*p != '\0' && *p != delim) {
        p += 1;
      }
      str_view = StrView(data_, p - data_);
      if (*p != '\0') {
        p += 1;
      }
      data_ = p;
      return true;
    } else {
      return false;
    }
  }

  StrView &trim(const char *chars = " \t\n\r\f\v") {  // be careful with the space ' \t\n\r\f\v'
    while (len_ > 0 && strchr(chars, *data_)) {
      data_++;
      len_--;
    }
    while (len_ > 0 && strchr(chars, data_[len_ - 1])) {
      len_--;
    }
    return *this;
  }

  // user promise that pos and count are valid
  StrView substr(uint64_t pos, uint64_t count) const {
    return StrView(data_ + pos, count);
  }

  // user promise that pos is valid
  StrView substr(uint64_t pos) const {
    return StrView(data_ + pos, len_ - pos);
  }

  uint64_t find(char c, uint64_t pos = 0) const {
    for (uint64_t i = pos; i < len_; i++) {
      if (data_[i] == c) {
        return i;
      }
    }
    return std::string::npos;
  }

  const char* data() const {
    return data_;
  }

  const char* c_str() const {
    return data_;
  }

  uint64_t size() const {
    return len_;
  }

  bool empty() const {
    return !static_cast<bool>(*this);
  }

  std::string to_string() const {
    return std::string(data_, len_);
  }

 private:
  const char* data_{nullptr};
  uint64_t len_{0};
  bool length_valid_{true};
};

inline bool operator==(const StrView &lhs, const StrView &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return strncmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}
inline bool operator!=(const StrView &lhs, const StrView &rhs) {
  return !(lhs == rhs);
}
inline bool operator==(const StrView &lhs, const std::string &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return strncmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}
inline bool operator!=(const StrView &lhs, const std::string &rhs) {
  return !(lhs == rhs);
}
inline bool operator==(const std::string &lhs, const StrView &rhs) {
  return rhs == lhs;
}
inline bool operator!=(const std::string &lhs, const StrView &rhs) {
  return !(rhs == lhs);
}

#ifdef __linux__
// memory map file, no memory copy
class FileView {
 public:
  FileView() = default;
  explicit FileView(const char *filename) { open(filename); }
  ~FileView() {
    if (data_ != nullptr) {
      munmap(data_, size_);
      ::close(fd_);
    }
  }

  FileView(const FileView &) = delete;
  FileView &operator=(const FileView &) = delete;

  FileView(FileView &&other) : fd_(other.fd_), size_(other.size_), data_(other.data_) {
    other.fd_ = -1;
    other.size_ = 0;
    other.data_ = nullptr;
  }
  FileView &operator=(FileView &&other) {
    inno_log_verify(fd_ == -1, "FileView is already opened");
    fd_ = other.fd_;
    size_ = other.size_;
    data_ = other.data_;
    other.fd_ = -1;
    other.size_ = 0;
    other.data_ = nullptr;
    return *this;
  }

 public:
  const char *data() const { return data_; }

  int64_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  void open(const char *filename) {
    inno_log_verify(fd_ == -1, "FileView is already opened");
    fd_ = ::open(filename, O_RDONLY);
    inno_log_verify(fd_ != -1, "open file %s failed", filename);
    size_ = lseek(fd_, 0, SEEK_END);
    lseek(fd_, 0, SEEK_SET);

    data_ = static_cast<char *>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    inno_log_verify(data_ != MAP_FAILED, "mmap file %s failed", filename);
  }

  void close() {
    if (data_ != nullptr) {
      munmap(data_, size_);
      ::close(fd_);
      fd_ = -1;
      size_ = 0;
      data_ = nullptr;
    }
  }

 private:
  int fd_{-1};
  int64_t size_{0};
  char *data_{nullptr};
};
#else
// copy file content to memory
class FileView {
 public:
  FileView() = default;
  ~FileView() = default;
  explicit FileView(const char *filename) { open(filename); }

  // disable copy
  FileView(const FileView &) = delete;
  FileView &operator=(const FileView &) = delete;

  FileView(FileView &&other) : data_(std::move(other.data_)) {}
  FileView &operator=(FileView &&other) {
    data_ = std::move(other.data_);
    return *this;
  }

 public:
  const char *data() const { return data_.c_str(); }

  int64_t size() const { return data_.size(); }

  bool empty() const { return data_.empty(); }

  void open(const char *filename) {
    FILE *file = fopen(filename, "rb");
    inno_log_verify(file != nullptr, "open file %s failed", filename);
    fseek(file, 0, SEEK_END);
    int64_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    data_.resize(size);
    inno_log_verify(fread(&data_[0], 1, size, file) == static_cast<uint64_t>(size), "read file %s failed", filename);
    fclose(file);
  }

  void close() {
    // do nothing
  }

 private:
  std::string data_;
};
#endif

}  // namespace innovusion

#endif  // UTILS_UTILS_H_
