/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_NET_MANAGER_H_
#define UTILS_NET_MANAGER_H_

#include <errno.h>
#include <fcntl.h>
#ifndef _WIN32
#include <libgen.h>
#endif
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !(defined(__MINGW64__) || defined(_WIN32))
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#if defined(__APPLE__)
#include <sys/select.h>
#else
#include <sys/epoll.h>
#endif
#else
#include <ws2tcpip.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>

#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
  #include <sys/sendfile.h>
#endif

#if !defined (__MINGW64__) && defined(_WIN32)
#define ssize_t __int64
#define __attribute__(X)
#endif

namespace innovusion {
/**
 * @brief NetManager
 */
class NetManager {
 private:
  static const uint16_t kDefaultHttpPort = 80;
  static const uint16_t kDefaultLidarPort = 8002;
  static const double kDefaultReadTimeoutSec;
  static const double kMinReadTimeoutSec;
  static const size_t kDefaultRecvBufferSize = 512 * 1024;

 public:
  /**
   * @brief Connect to Lidar with ip and port,
   *        set the connection timeout and recv buffer size
   * @param ip               Lidar ip to connect
   * @param port             Lidar port to connect
   * @param read_timeout_sec Connection timeout
   * @param recv_buffer_size Recv buffer size
   * @return Return connection positive fd if success, others for error
   */
  static int get_connection(const char *ip,
                            unsigned short port,
                            double read_timeout_sec = kDefaultReadTimeoutSec,
                            int recv_buffer_size = kDefaultRecvBufferSize);
  /**
   * @brief Send command to connection fd, and recv reply
   * @param fd         Connection fd
   * @param reply      Buffer to store reply
   * @param reply_len  Buffer size
   * @param cmd_fmt    Command format
   * @param ...        Command arguments
   * @return Return positive fd if success, others for error
   */
  static int send_command_with_fd(int fd, char *reply, int *reply_len,
                                  const char *cmd_fmt, ...)
      __attribute__((format(printf, 4, 5)));
  /**
   * @brief Receive data from connection fd
   *        return until recv_len bytes received or some error occurs
   * @param fd       Connection fd
   * @param buffer   Buffer to store data
   * @param recv_len Buffer size
   * @param flag     Flag for recv
   * @return Return positive integer for received data length, others for error
   */
  static ssize_t recv_full_buffer(int fd, char *buffer,
                                  size_t recv_len, int flag);
  /**
   * @brief Write data to connection fd
   *        return until write count bytes or some error occurs
   * @param fd    Connection fd
   * @param buf   Buffer write to fd
   * @param count Buffer size
   * @return Return positive integer for written data length, others for error
   */
  static ssize_t write_full_buffer(int fd, const void *buf, size_t count);

 private:
  /**
   * @brief Send command to connection fd, and recv reply
   * @param fd        Connection fd
   * @param reply     Buffer to store reply
   * @param reply_len Buffer size
   * @param cmd_fmt   Command format
   * @param valist    Command arguments
   * @return Return positive fd if success, others for error
   */
  static int send_command_with_fd_v_(int fd, char *reply, int *reply_len,
                                     const char *cmd_fmt, va_list valist);

 public:
  /**
   * @brief NetManager constructor
   * @param ip           Lidar ip to connect
   * @param port         Lidar port to connect
   * @param timeout_sec  Connection timeout
   */
  NetManager(const char *ip,
             unsigned short port = kDefaultLidarPort,
             double timeout_sec = kDefaultReadTimeoutSec);
  ~NetManager();
  /**
   * @brief Set Lidar port
   * @param port Lidar port to connect
   */
  void set_base_port(uint16_t port) {
    port_ = port;
  }
  /**
   * @brief Get connetion fd to Lidar
   * @param timeout_sec      Connection timeout
   * @param recv_buffer_size Recv buffer size
   * @return Return connection positive fd if success, others for error
   */
  int get_connection(double timeout_sec,
                     int recv_buffer_size = kDefaultRecvBufferSize);
  /**
   * @brief Send get request by http to Lidar
   * @param url          Url to send
   * @param buffer       Buffer to store the whole response
   * @param buff_size    Buffer size
   * @param status_code  Buffer to store status code of the response
   * @param content      Buffer to store the content of the response
   * @param content_size Content length
   * @param timeout_sec  Connection timeout
   * @param port         Http connection port
   * @return Return 0 if success, others for error
   */
  int http_get(const char *url, char *buffer, size_t buff_size,
               int *status_code, char **content, int *content_size,
               double timeout_sec, unsigned short port = 0);
  /**
   * @brief Send command to Lidar, and recv reply from Lidar
   *        recieved data will be written to file
   * @param file_fd    File fd to write response
   * @param expect_md5 Set non-zero to check md5 of the response
   * @param cmd        Command format
   * @param ...        Command arguments
   * @return Return 0 if success, others for error
   */
  int recv_file(int file_fd, int expect_md5, const char *cmd, ...)
      __attribute__((format(printf, 4, 5)));
  /**
   * @brief Send command to Lidar, and recv reply from Lidar
   *        recieved data will be written to buffer
   * @param buff        Buffer to store response
   * @param buff_len    Buffer size
   * @param expect_md5  Set non-zero to check md5 of the response
   * @param cmd         Command format
   * @param ...         Command arguments
   * @return Return positive integer for data length written to buffer,
   *         others for error
   */
  int recv_length_buffer(char *buff, int buff_len, int expect_md5,
                         const char *cmd, ...)
      __attribute__((format(printf, 5, 6)));
  /**
   * @brief Send file to Lidar with specified command
   * @param filename File to be sent
   * @param cmd      Command format
   * @param ...      Command arguments
   * @return Return 0 if success, others for error
   */
  int send_file(const char *filename, const char *cmd, ...)
      __attribute__((format(printf, 3, 4)));
  /**
   * @brief Send command to Lidar, and recv reply from Lidar
   * @param reply       Buffer to store reply
   * @param reply_len   Buffer size
   * @param cmd_fmt     Command format
   * @param ...         Command arguments
   * @return Return positive fd if success, others for error
   */
  int send_command_return_fd(char *reply, int *reply_len,
                             const char *cmd_fmt, ...)
      __attribute__((format(printf, 4, 5)));
  /**
   * @brief Send command to Lidar, and recv reply from Lidar,
   *        this function will return a new allocated buffer which stores
   *        the reply
   * @param cmd        Command format
   * @param ...        Command arguments
   * @return Return a new allocated buffer if success, NULL for error
   */
  char *send_command_and_get_reply(const char *cmd, ...)
      __attribute__((format(printf, 2, 3)));
  /**
   * @brief Send command to Lidar, and recv nothings
   * @param cmd       Command format
   * @param ...       Command arguments
   * @return Return 0 if success, others for error
   */
  int send_command_and_free_reply(const char *cmd, ...)
      __attribute__((format(printf, 2, 3)));
  /**
   * @brief Send command to Lidar, and print reply to stdout
   * @param cmd      Command format
   * @param ...      Command arguments
   * @return Return 0 if success, others for error
   */
  int send_command_and_print_reply(const char *cmd, ...)
      __attribute__((format(printf, 2, 3)));
  /**
   * @brief Set default connection timeout
   * @param s Timeout in second
   */
  void set_default_timeout_sec(double s) {
    default_timeout_sec_ = s;
  }
  /**
   * @brief Convert ip string to in_addr
   * @param src Ip string
   * @param dst in_addr struct
   * @return Return 1 if success, others for error
   */
  static int inno_inet_pton(const char *src, struct in_addr *dst);

 private:
  /**
   * @brief Send command to Lidar, and recv reply from Lidar
   *        this function will return a new allocated buffer which stores
   * @param cmd     Command format
   * @param valist  Command arguments
   * @return Return a new allocated buffer if success, NULL for error
   */
  char *send_command_and_get_reply_v_(const char *cmd,
                                      va_list valist);
  /**
   * @brief Send command to Lidar, and print reply to stdout
   * @param cmd    Command format
   * @param valist Command arguments
   * @return Return 0 if success, others for error
   */
  int send_command_and_print_reply_v_(const char *cmd,
                                      va_list valist);

 protected:
  char ip_[64] = {0};

 private:
  uint16_t port_;
  double default_timeout_sec_;

 public:
  static class NetManagerInit {
   public:
    NetManagerInit() {
#if defined(__MINGW64__) || defined(_WIN32)
      WSAStartup(MAKEWORD(2, 2), &wsaData_);
#endif
    }

#if defined(__MINGW64__) || defined(_WIN32)

   private:
    WSADATA wsaData_;
#endif
// The API is initialized by default at the time of namespace,
// and is automatically destructed at the end.
  } manager_init;
};

}  // namespace innovusion
#endif  // UTILS_NET_MANAGER_H_
