/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include "utils/net_manager.h"

#include <ctype.h>
#include <inttypes.h>

#if !defined (__MINGW64__) && defined(_WIN32)
#undef max
#endif
#include <algorithm>
#include <vector>

#include "sdk_common/inno_lidar_packet.h"
#include "utils/log.h"
#include "utils/md5.h"
#include "utils/utils.h"

#if !defined (__MINGW64__) && defined(_WIN32)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <io.h>

#define close _close
#define read _read
#define write _write
#define lseek _lseek
#else
#include <unistd.h>
#endif

namespace innovusion {

const double NetManager::kDefaultReadTimeoutSec = 0.5;
const double NetManager::kMinReadTimeoutSec = 0.001;
NetManager::NetManagerInit NetManager::manager_init;

NetManager::NetManager(const char *ip_addr,
                       unsigned short port,
                       double timeout_sec) {
  inno_log_verify(ip_addr, "null ip_addr");
  inno_log_verify(strlen(ip_addr) < sizeof(ip_),
                  "Invalid ip_addr:port %s:%hu", ip_addr, port);
  const char *sep = strchr(ip_addr, ':');
  if (sep) {
    unsigned int len = sep - ip_addr;
    if (len >= sizeof(ip_)) {
      inno_log_error("Invalid ip_addr:port %s", ip_addr);
      strncpy(ip_, ip_addr, sizeof(ip_) - 1);
      ip_[sizeof(ip_) - 1] = 0;
      port_ = port;
    } else {
      strncpy(ip_, ip_addr, len);
      ip_[len] = 0;
    }
    int p = atoi(sep+1);
    if (p > 0) {
      port_ = p;
    } else {
      inno_log_error("Invalid ip_addr:port %s", ip_addr);
      port_ = port;
    }
  } else {
    strncpy(ip_, ip_addr, sizeof(ip_) - 1);
    ip_[sizeof(ip_) - 1] = 0;
    port_ = port;
  }
  default_timeout_sec_ = std::max(timeout_sec, kMinReadTimeoutSec);
}

NetManager::~NetManager() {
}

int NetManager::get_connection(double timeout_sec,
                               int recv_buffer_size) {
  int rt = get_connection(ip_, port_,
                          timeout_sec, recv_buffer_size);
  if (rt < 0) {
    inno_log_error("Error opening tcp connection (%s:%d), ret=%d",
                   ip_, port_, rt);
    return rt - 200;
  } else {
    return rt;
  }
}

int NetManager::http_get(const char *url,
                         char *buffer_out,
                         size_t buffer_size,
                         int *status_code,
                         char **content,
                         int *content_size,
                         double timeout_sec,
                         unsigned short port) {
  if (buffer_size == 0) {
    inno_log_error("cannot get_url %s with 0 size buffer", url);
    return -1;
  }
  if (buffer_size == 1) {
    inno_log_error("cannot get_url %s with 1 size buffer", url);
    *buffer_out = 0;
    return -1;
  }

  uint16_t port_connect = port == 0 ? port_ : port;
  //inno_log_info("Requesting %s from %s:%d", url, ip_, port_connect);
  int fd = get_connection(ip_, port_connect, timeout_sec);
  if (fd < 0) {
    inno_log_error_errno("Error opening HTTP connection (%s:%d)",
                   ip_, port_connect);
    return fd;
  }

  char request[64 * 1024] = {0};
  uint32_t crc32 = 0;

  if (url != NULL) {
    crc32 = InnoUtils::calculate_http_crc32(url, strlen(url), true);
  }
  snprintf(request, sizeof(request),
           "GET %s HTTP/1.1\r\n"
           "Cache-Control: no-cache\r\n"
           "Contest-length: 0\r\n"
           "Content-Type: text/html\r\n"
           "X-INNO-CRC32: %8x\r\n"
           "X-INNO-MAJOR-VERSION: %u\r\n"
           "X-INNO-MINOR-VERSION: %u\r\n"
           "Connection: Closed\r\n\r\n",
           url, crc32, kInnoMajorVersionDataPacket, kInnoMinorVersionDataPacket);
  int request_len = static_cast<int>(strlen(request));

  int send_ret;
  while (-1 == (send_ret = send(fd, request, request_len, 0)) &&
         errno == EINTR) {
  }
  if (send_ret != request_len) {
    inno_log_error("Error sending %s (%d != %d)",
                   request, send_ret, request_len);
    return send_ret;
  }

  char *cursor = buffer_out;
  char *const buffer_end = buffer_out + buffer_size - 1;
  buffer_out[buffer_size - 1] = 0;
  int read_ret;
  int ret = 0;
  while (1) {
#if !(defined(__MINGW64__) || defined(_WIN32))
    while (-1 == (read_ret = read(fd, cursor, buffer_end - cursor)) &&
           errno == EINTR) {
    }
#else
    while (-1 == (read_ret = recv(fd, cursor, buffer_end - cursor, 0)) &&
           errno == EINTR) {
    }
#endif
    if (read_ret > 0) {
      cursor += read_ret;
      *cursor = 0;
      static const char *kCLU = "Content-Length:";
      static const char *kCLL = "content-length:";
      static const char *kHeaderEnd = "\r\n\r\n";

      int res = InnoUtils::verify_lidar_version(buffer_out);
      if (res < 0) {
        inno_log_error("upgrade sdk version.");
        ret = -9;
        break;
      }
      char *hd_end = strstr(buffer_out, kHeaderEnd);
      if (content) {
        *content = NULL;
      }
      if (content_size) {
        *content_size = 0;
      }
      *status_code = 500;

      if (hd_end != NULL && hd_end < cursor) {
        // received all headers
        // find status code
        {
          int v;
          if (2 != sscanf(buffer_out, "HTTP/1.%d %d ",
                          &v, status_code)) {
            inno_log_warning("bad response from %s. "
                             "response=%s",
                             url, buffer_out);
            ret = -10;
            break;
          }
        }
        char *lp = strstr(buffer_out, kCLU);
        if (lp == NULL || lp >= hd_end) {
          lp = strstr(buffer_out, kCLL);
        }
        if (lp != NULL && lp < hd_end) {
          lp += strlen(kCLL);
          while (isspace((u_char)*lp)) {
            lp++;
          }
          int32_t content_length = 0;
          if (1 == sscanf(lp, "%d", &content_length)) {
            char *header_end = hd_end + strlen(kHeaderEnd);
            if (content) {
              *content = header_end;
            }
            if (content_size) {
              *content_size = content_length;
            }
            int received = cursor - header_end;
            int to_receive = content_length - received;
            if (to_receive > buffer_end - cursor) {
              inno_log_error("not enough buffer to get_url %s. "
                             "buffer_size=%" PRI_SIZELU " response=%s "
                             "received=%d content_length=%d",
                             url, buffer_size, buffer_out,
                             received, content_length);
              ret = -11;
              break;
            } else if (to_receive > 0) {
              ssize_t z = recv_full_buffer(fd, cursor,
                                           to_receive, 0);
              if (z < 0) {
                inno_log_warning("cannot received full %s",
                                 buffer_out);
                ret = -12;
                break;
              }
              if (z == to_receive) {
                // all received
                cursor += to_receive;
                break;
              } else {
                inno_log_warning("content-length %d to small %"
                                 PRI_SIZELD " %d",
                                 content_length,
                                 z, to_receive);
                ret = -13;
                break;
              }
            } else if (to_receive == 0) {
              // all received
              break;
            } else {
              inno_log_warning("received too much %s %d %d %d",
                               buffer_out, content_length, received,
                               to_receive);
              ret = -14;
              break;
            }
          }
        } else {
          inno_log_warning("no content-length: %s",
                           buffer_out);
          ret = -15;
          break;
        }
      }
      if (cursor >= buffer_end) {
        inno_log_error("not enough buffer to get_url %s. "
                       "buffer_size=%" PRI_SIZELU,
                       url, buffer_size);
        ret = -16;
        break;
      }
    } else {
      inno_log_warning_errno("read_ret %d, fd=%d", read_ret, fd);
      ret = -17;
      break;
    }
  }

  if ((ret == 0) && (InnoUtils::verify_http_crc32(buffer_out, NULL) < 0)) {
    inno_log_warning("crc32 check failed.");
    ret = -9;
  }
  *cursor = 0;  // zero end string
  InnoUtils::close_fd(fd);
  return ret;
}

ssize_t NetManager::recv_full_buffer(int fd, char *buffer,
                                     size_t recv_len, int flag) {
  size_t received = 0;
  inno_log_trace("recv_full_buffer total to recv %" PRI_SIZET "",
                 recv_len);
  if (recv_len <= 0) {
    return recv_len;
  }
  while (received < recv_len) {
    int n;
    if (flag == 0) {
#if !(defined(__MINGW64__) || defined(_WIN32))
      while (-1 == (n = read(fd, buffer + received,
                             recv_len - received)) &&
             errno == EINTR) {
      }
#else
      while (-1 == (n = recv(fd, buffer + received,
                             recv_len - received, 0)) &&
             errno == EINTR) {
      }
#endif
    } else {
      if (flag == -1) {
        while (-1 == (n = read(fd, buffer + received,
                               recv_len - received)) &&
             errno == EINTR) {
        }
      } else {
        while (-1 == (n = recv(fd, buffer + received,
                               recv_len - received, flag)) &&
               errno == EINTR) {
        }
      }
    }
    if (n < 0) {
      // inno_log_trace("fd=%d n=%d err=%d", fd, n, errno);
    }
    inno_log_trace("received %d %" PRI_SIZELU "/%" PRI_SIZELU,
      n, received, recv_len);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        // inno_log_error_errno("recv %d", n);
        return n;
      }
    } else if (n == 0) {
      return -1;
    } else {
      received += n;
    }
  }
  inno_log_trace("!!! all received");
  return received;
}

ssize_t NetManager::write_full_buffer(int fd, const void *buf, size_t count) {
  size_t wrote_total = 0;
  while (wrote_total < count) {
    int wrote;
#if !(defined(__MINGW64__) || defined(_WIN32))
    while (-1 == (wrote = write(fd, buf, count - wrote_total)) &&
           errno == EINTR) {
    }
#else
    if (InnoUtils::is_socket_fd(fd)) {
      while (-1 == (wrote = send(fd, (const char *)buf,
                                 count - wrote_total, 0)) &&
                                 errno == EINTR) {
      }
    } else {
      while (-1 == (wrote = write(fd, buf, count - wrote_total)) &&
             errno == EINTR) {
      }
    }
#endif
    if (wrote < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        return wrote;
      }
    } else if (wrote == 0) {
      return -1;
    } else {
      wrote_total += wrote;
    }
  }
  return wrote_total;
}

int NetManager::recv_file(int file_fd, int expect_md5,
                          const char *cmd_fmt, ...) {
  int len;
  MD5_result md5_result;
  MD5_result_str md5_result_str;
  MD5_result_str md5_result_str_remote;
  MD5_CTX md5_ctx;
  int fd, ret;

  fd = get_connection(5.0);
  if (fd < 0) {
    inno_log_error("Error: cannot send command to recv_file_a  %s", cmd_fmt);
    return fd - 100;
  }
  va_list valist;
  va_start(valist, cmd_fmt);
  ret = send_command_with_fd_v_(fd, NULL, &len, cmd_fmt, valist);
  va_end(valist);
  if (ret < 0) {
    inno_log_error("Error: cannot send command to recv_file_b %s", cmd_fmt);
    InnoUtils::close_fd(fd);
    return ret - 200;
  }

  if (recv_full_buffer(fd, reinterpret_cast<char *>(&len), sizeof(len), 0) < 0) {
    inno_log_error("Error: recv file len faild,file maybe not exist, cmd is: %s", cmd_fmt);
    InnoUtils::close_fd(fd);
    return -1;
  }

  int total_len = ntohl(len);
  int read_already = 0;
  const uint32_t recv_buff_size = 1024 * 1024;
  std::vector<char> recv_buf;
  recv_buf.resize(recv_buff_size);

  MD5_Init(&md5_ctx);
  while (read_already < total_len) {
    ssize_t n;
    ssize_t s = recv_buff_size - 1;
    ssize_t left = total_len - read_already;
    if ((n = recv_full_buffer(fd, &recv_buf[0],
                              left < s ? left : s, 0)) < 0) {
      inno_log_error("Error: recv failed a. "
                     "%" PRI_SIZELD "/%d bytes not received",
                     left, total_len);
      InnoUtils::close_fd(fd);
      return -2;
    }
     MD5_Update(&md5_ctx, &recv_buf[0], n);
    ssize_t w = write_full_buffer(file_fd, &recv_buf[0], n);
    if (w < n) {
      inno_log_error_errno("Error: write failed. "
                     "%" PRI_SIZELD " vs %" PRI_SIZELD,
                     w, n);
      InnoUtils::close_fd(fd);
      return -3;
    }
    read_already += n;
  }
  MD5_Final(md5_result.result, &md5_ctx);
  MD5_print(md5_result_str.str, sizeof(md5_result_str.str), md5_result.result);

  if (expect_md5) {
    if (recv_full_buffer(fd, reinterpret_cast<char *>(&md5_result_str_remote),
                         sizeof(md5_result_str_remote), 0) < 0) {
      inno_log_error("Error: recv failed d md5");
      InnoUtils::close_fd(fd);
      return -4;
    }
    if (0 != memcmp(md5_result_str.str, md5_result_str_remote.str,
                    sizeof(md5_result_str.str))) {
      inno_log_error("md5 mismatch a %s vs %s",
                     md5_result_str.str, md5_result_str_remote.str);
      InnoUtils::close_fd(fd);
      return -5;
    }
  }
  InnoUtils::close_fd(fd);
  return 0;
}

int NetManager::recv_length_buffer(char *buff,
                                   int buff_len, int expect_md5,
                                   const char *cmd_fmt, ...) {
  int len;
  MD5_result md5_result;
  MD5_result_str md5_result_str;
  MD5_result_str md5_result_str_remote;
  MD5_CTX md5_ctx;
  int fd;

  fd = get_connection(default_timeout_sec_);
  if (fd < 0) {
    inno_log_error("Error: cannot send command to recv_file_c %s", cmd_fmt);
    return fd - 100;
  }

  va_list valist;
  va_start(valist, cmd_fmt);
  int ret = send_command_with_fd_v_(fd, NULL, &len, cmd_fmt, valist);
  va_end(valist);
  if (ret < 0) {
    inno_log_error("Error: cannot send command to recv_file_d %s", cmd_fmt);
    InnoUtils::close_fd(fd);
    return ret - 200;
  }

  if (recv_full_buffer(fd, reinterpret_cast<char *>(&len),
                       sizeof(len), 0) < 0) {
    inno_log_error("Error: recv failed d");
    InnoUtils::close_fd(fd);
    return -1;
  }

  len = ntohl(len);

  if (len >= buff_len) {  // need to set buff[len] to 0 later
    inno_log_error("Error: file size %d is too large (>=%d)", len, buff_len);
    InnoUtils::close_fd(fd);
    return -2;
  }

  MD5_Init(&md5_ctx);
  int n = recv_full_buffer(fd, buff, len, 0);
  if (n < 0) {
    inno_log_error("Error: recv failed b. %d bytes not received", len);
    InnoUtils::close_fd(fd);
    return -3;
  }
  MD5_Update(&md5_ctx, buff, n);
  MD5_Final(md5_result.result, &md5_ctx);
  MD5_print(md5_result_str.str, sizeof(md5_result_str.str), md5_result.result);

  if (expect_md5) {
    if (recv_full_buffer(fd, reinterpret_cast<char *>(&md5_result_str_remote),
                         sizeof(md5_result_str_remote), 0) < 0) {
      inno_log_error("Error: recv failed e md5");
      InnoUtils::close_fd(fd);
      return -4;
    }
    if (0 != memcmp(md5_result_str.str, md5_result_str_remote.str,
                    sizeof(md5_result_str.str))) {
      inno_log_error("md5 mismatch b %s vs %s",
                     md5_result_str.str, md5_result_str_remote.str);
      InnoUtils::close_fd(fd);
      return -5;
    }
  }

  buff[len] = 0;
  InnoUtils::close_fd(fd);
  return len;
}

int NetManager::send_file(const char *filename, const char *cmd_fmt, ...) {
  // 1. send 4-byte length
  // 2. send content
  // 3. send 16-byte md5
  // 4. recv 4-byte length
  int err;
  int total_size_nl, total_size;
  int len;
  int ret;
  int file_fd;
  int fd;

  struct stat st;
  err = stat(filename, &st);
  if (err != 0) {
    inno_log_error("Error: cannot stat file %s", filename);
    return -1;
  }
  total_size = st.st_size;
  // inno_log_info("send file %s, size is %d bytes", filename, total_size);
  fd = get_connection(5.0);
  if (fd < 0) {
    inno_log_error("Error: cannot send command to send file_a %s", cmd_fmt);
    return fd - 100;
  }
  va_list valist;
  va_start(valist, cmd_fmt);
  ret = send_command_with_fd_v_(fd, NULL, &len, cmd_fmt, valist);
  va_end(valist);
  if (ret < 0) {
    inno_log_error("Error: cannot send command to send file_b %s", cmd_fmt);
    InnoUtils::close_fd(fd);
    return ret - 200;
  }
  total_size_nl = htonl(total_size);
  int sent = write_full_buffer(fd, reinterpret_cast<char *>(&total_size_nl),
                               sizeof(total_size_nl));
  if (sent < 0) {
    inno_log_error("Error: cannot send len %s %s", filename, cmd_fmt);
    InnoUtils::close_fd(fd);
    return -2;
  }
  if (sent != sizeof(total_size_nl)) {
    inno_log_error("Error: partial send %d vs %d", sent, total_size_nl);
    InnoUtils::close_fd(fd);
    return -3;
  }
  file_fd = InnoUtils::open_file(filename, O_RDONLY, 0);
  if (file_fd < 0) {
    inno_log_error("Error: cannot open file %s", filename);
    InnoUtils::close_fd(fd);
    return -4;
  }

  // send content
  int sent_size = 0;
  while (sent_size < total_size) {
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
    ret = sendfile(fd, file_fd, NULL, total_size - sent_size);
#else
    char buffer[64 * 1024];
    int32_t read_size = total_size - sent_size;
    if (read_size > (int32_t)sizeof(buffer)) {
      read_size = sizeof(buffer);
    }
    ret = read(file_fd, buffer, read_size);
    if (ret == read_size) {
      ret = send(fd, buffer, read_size, 0);
      if (ret != read_size) {
        ret = -1;
      }
    } else {
      ret = -2;
    }
#endif
    if (ret < 0) {
      inno_log_error("Error: cannot sendfile %s ret=%d, %d %d",
                     filename, ret, sent_size, total_size);
      InnoUtils::close_fd(fd);
      close(file_fd);
      return ret - 300;
    }
    sent_size += ret;
  }
  inno_log_verify(sent_size == total_size,
                  "Error: cannot sendfile %s %d vs %d",
                  filename, sent_size, total_size);

  // calculate local md5
  lseek(file_fd, 0, SEEK_SET);
  int read_size = 0;
  char md5_read_buf[1024];
  MD5_result md5_result;
  MD5_result_str md5_result_str;
  MD5_CTX md5_ctx;

  MD5_Init(&md5_ctx);
  while (read_size < total_size) {
    size_t left = total_size - read_size;
    int to_be_read = left < sizeof(md5_read_buf) ? left : sizeof(md5_read_buf);
    int rd;
    while (-1 == (rd = read(file_fd, md5_read_buf, to_be_read)) &&
           errno == EINTR) {
    }
    if (rd != to_be_read) {
      inno_log_error("Error: cannot read file %s %d vs %d", filename,
                     rd, to_be_read);
      InnoUtils::close_fd(fd);
      close(file_fd);
      return -5;
    }
    MD5_Update(&md5_ctx, md5_read_buf, to_be_read);
    read_size += rd;
  }
  MD5_Final(md5_result.result, &md5_ctx);
  MD5_print(md5_result_str.str, sizeof(md5_result_str.str), md5_result.result);
  // send md5
  sent = write_full_buffer(fd, reinterpret_cast<char *>(md5_result_str.str),
                           sizeof(md5_result_str.str));
  if (sent != sizeof(md5_result_str)) {
    inno_log_error("Error: cannot send md5 %s, %d", filename, sent);
    InnoUtils::close_fd(fd);
    close(file_fd);
    return -6;
  }

  // receive ack
  int ack_len = 0;
  ret = recv_full_buffer(fd, reinterpret_cast<char *>(&ack_len),
                         sizeof(ack_len), 0);
  InnoUtils::close_fd(fd);
  close(file_fd);
  if (ret < 0) {
    inno_log_error("Error: recv ack failed");
    return -7;
  } else {
    ack_len = ntohl(ack_len);
    if (total_size != ack_len) {
      inno_log_error("Error: size mismatch %d vs %d, failed to send",
                     total_size, ack_len);
      return -8;
    } else {
      return 0;
    }
  }
}

int NetManager::send_command_with_fd(int fd,
                                     char *reply,
                                     int *reply_len,
                                     const char *cmd_fmt, ...) {
  va_list valist;
  va_start(valist, cmd_fmt);
  int ret = send_command_with_fd_v_(fd, reply, reply_len,
                                    cmd_fmt, valist);
  va_end(valist);
  return ret;
}

int NetManager::send_command_with_fd_v_(int fd,
                                        char *reply,
                                        int *reply_len,
                                        const char *cmd_fmt, va_list valist) {
  int vreturn;
  char buffer[64 * 1024];
  buffer[0] = 0;

  vreturn = vsnprintf(buffer, sizeof(buffer), cmd_fmt, valist);
  if (vreturn <= 0 || (size_t)vreturn >= sizeof(buffer)) {
    inno_log_error("Error: command too long %s", cmd_fmt);
    return -1;
  }
  buffer[sizeof(buffer) - 1] = 0;

  size_t slen = strlen(buffer);
  if (slen < sizeof(buffer) - 1) {
    // append '\n'
    buffer[slen] = '\n';
    buffer[slen + 1] = 0;
  } else {
    inno_log_error("Error: command too long %s", buffer);
    return -2;
  }

  int sent;
  int buffer_len = strlen(buffer);
  if (buffer_len > 0) {
    sent = write_full_buffer(fd, buffer, buffer_len);
  } else {
    sent = 0;
  }
  if (sent < 0) {
    // send failed
    inno_log_error("Error: send failed %d", sent);
    return -3;
  } else {
    if (sent != buffer_len) {
      // should be impossible
      inno_log_error("Error: partial sent %d vs %d", sent, buffer_len);
      return -4;
    }
    // send command successful
    if (reply) {
      // read until find '\n\n'
      int already_read = 0;
      while (already_read < *reply_len) {
        int max_read = *reply_len - already_read;
        int n;
        while (-1 == (n = recv(fd, reply + already_read,
                               max_read, 0)) &&
               errno == EINTR) {
        }
        if (n < 0) {
          // inno_log_error("Error: recv failed b %d", n);
          inno_log_error_errno(" %d", n);
          return -6;
        } else if (n == 0) {
          return -8;
        } else {
          already_read += n;
          // search for '\n\n'
          for (int i = 0; i < already_read - 1; i++) {
            if (reply[i] == '\n' && reply[i+1] == '\n') {
              // found
              *reply_len = already_read;
              return fd;
            }
          }
          // not found, continue loop to read
        }
      }
      // still not found
      reply[*reply_len - 1] = 0;
      inno_log_error("Error: invalid reply %s %d %d",
                     reply, already_read, *reply_len);
      return -7;
    } else {
      // don't read, just return
      return fd;
    }
  }
}

int NetManager::send_command_return_fd(char *reply, int *reply_len,
                                       const char *cmd_fmt, ...) {
  int fd = get_connection(default_timeout_sec_);
  if (fd < 0) {
    inno_log_error("Error: cannot send command cmd=%s", cmd_fmt);
    return fd;
  }
  va_list valist;
  va_start(valist, cmd_fmt);
  int ret = send_command_with_fd_v_(fd,
                                    reply,
                                    reply_len, cmd_fmt, valist);
  va_end(valist);
  if (ret < 0) {
    InnoUtils::close_fd(fd);
    inno_log_error("Error: send command failed");
    return ret;
  } else {
    return fd;
  }
}

char *NetManager::send_command_and_get_reply(const char *cmd, ...) {
  va_list valist;
  va_start(valist, cmd);
  char *ret = send_command_and_get_reply_v_(cmd, valist);
  va_end(valist);
  return ret;
}

int NetManager::send_command_and_free_reply(const char *cmd, ...) {
  va_list valist;
  va_start(valist, cmd);
  char *reply = send_command_and_get_reply_v_(cmd, valist);
  va_end(valist);

  if (reply) {
    free(reply);
    return 0;
  } else {
    return -1;
  }
}


char *NetManager::send_command_and_get_reply_v_(const char *cmd,
                                                va_list valist) {
  char reply[64 * 1024] = {0};
  int reply_len = sizeof(reply) - 1;
  for (int i = 0; i < 2; i++) {
    int fd = get_connection(default_timeout_sec_);
    if (fd < 0) {
      inno_log_warning("get_connection failed in %lf second", default_timeout_sec_);
      continue;
    }
    int retval = send_command_with_fd_v_(fd, reply, &reply_len, cmd, valist);
    if (retval < 0) {
      InnoUtils::close_fd(fd);
      continue;
    } else {
      char *ret = NULL;
      reply[reply_len] = 0;

      // remove the last "\n"
      char *idx = strrchr(reply, '\n');
      if (idx == NULL || idx == reply || *(idx-1) != '\n') {
        inno_log_error("Error: invalid reply: %s", reply);
        ret = NULL;
      } else {
        *idx = 0;
        ret = strdup(reply);
      }
      InnoUtils::close_fd(fd);
      return ret;
    }
  }
  char full_cmd[1024];
  vsnprintf(full_cmd, sizeof(full_cmd), cmd, valist);
  full_cmd[sizeof(full_cmd) - 1] = 0;
  inno_log_error("Error: cannot send command %s", full_cmd);
  return NULL;
}

int NetManager::send_command_and_print_reply_v_(const char *cmd,
                                                va_list valist) {
  char *reply = send_command_and_get_reply_v_(cmd, valist);

  if (reply) {
    inno_fprintf(1, "%s", reply);
    free(reply);
    return 0;
  } else {
    return -1;
  }
}

int NetManager::send_command_and_print_reply(const char *cmd, ...) {
  va_list valist;
  va_start(valist, cmd);
  int ret = send_command_and_print_reply_v_(cmd, valist);
  va_end(valist);
  return ret;
}

int NetManager::inno_inet_pton(const char *src, struct in_addr *dst) {
#if !(defined(__MINGW64__) || defined(_WIN32))
  return inet_pton(AF_INET, src, dst);
#else
  static const char digits[] = "0123456789";
  int saw_digit, octets, ch;
#define NS_INADDRSZ 4
  unsigned char tmp[NS_INADDRSZ], *tp;

  saw_digit = 0;
  octets = 0;
  *(tp = tmp) = 0;
  while ((ch = *src++) != '\0') {
    const char *pch;

    if ((pch = strchr(digits, ch)) != NULL) {
      unsigned char nw = *tp * 10 + (pch - digits);

      if (saw_digit && *tp == 0) {
        return -1;
      }
      if (nw > 255) {
        return -1;
      }
      *tp = nw;
      if (!saw_digit) {
        if (++octets > 4) {
          return -1;
        }
        saw_digit = 1;
      }
    } else if (ch == '.' && saw_digit) {
      if (octets == 4) {
        return -1;
      }
      *++tp = 0;
      saw_digit = 0;
    } else {
      return -1;
    }
  }
  if (octets < 4) {
    return -1;
  }
  memcpy(dst, tmp, NS_INADDRSZ);
  return 1;
#endif
}

int NetManager::get_connection(const char *ip,
                               uint16_t port,
                               double read_timeout_sec,
                               int recv_buffer_size) {
  bool connected = false;
  int fd = -1;

  while (!connected) {
    struct sockaddr_in serv_addr;
    if (fd >= 0) {
      InnoUtils::close_fd(fd);
    }

    if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      inno_log_error_errno("Error: Could not create socket %d, fd=%d",
                           errno, fd);
      return -1;
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inno_inet_pton(ip, &serv_addr.sin_addr) <= 0) {
      inno_log_error_errno("inet_pton error %d", errno);
      return -2;
    }

    {
#if !(defined(__MINGW64__) || defined(_WIN32))
      int one = 1;
#else
      char one = 1;
#endif
      socklen_t optlen = sizeof(one);
      if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, optlen) < 0) {
        inno_log_error("%s cannot setsockopt keepalive", ip);
        InnoUtils::close_fd(fd);
        return -3;
      }
      if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
        inno_log_error("%s cannot setsockopt tcp_nodelay", ip);
        InnoUtils::close_fd(fd);
        return -4;
      }
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
      if (setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)) < 0) {
        inno_log_error("%s cannot setsockopt tcp_quickack", ip);
        InnoUtils::close_fd(fd);
        return -5;
      }
#endif

#if !(defined(__MINGW64__))  //! TODO: check if this is necessary for macos
      int rcvbuff = recv_buffer_size;
      if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcvbuff),
                     sizeof(rcvbuff)) < 0) {
        inno_log_error("%s cannot setsockopt rcvbuf", ip);
        InnoUtils::close_fd(fd);
        return -6;
      }
#endif
    }

    struct timeval tv;
#if !(defined(__MINGW64__) || defined(_WIN32))
    tv.tv_sec = static_cast<int>(read_timeout_sec);
    tv.tv_usec = (read_timeout_sec - tv.tv_sec) * 1000000;

    if (read_timeout_sec != 0) {
      int ret = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                           reinterpret_cast<void *>(&tv), sizeof(tv));
      if (ret < 0) {
        inno_log_error("Error: setsockopt");
        InnoUtils::close_fd(fd);
        return -7;
      }
    }
#else
    if (read_timeout_sec != 0) {
      int timeout = static_cast<int>(read_timeout_sec * 1000);
      int ret = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&timeout), sizeof(timeout));
      if (ret < 0) {
        inno_log_error("Error: setsockopt");
        InnoUtils::close_fd(fd);
        return -7;
      }
    }
#endif

#if !(defined(__MINGW64__) || defined(_WIN32))
    int flag;
    // set to non-blocking mode
    if ((flag = fcntl(fd, F_GETFL, NULL)) < 0) {
      inno_log_error("Error: fcntl F_GETFL");
      InnoUtils::close_fd(fd);
      return -8;
    }
    flag |= O_NONBLOCK;
    if (fcntl(fd, F_SETFL, flag) < 0) {
      inno_log_error("Error: fcntl F_SETFL");
      InnoUtils::close_fd(fd);
      return -9;
    }

#else
    // set to non-blocking mode
    u_long block = 1;
    if (ioctlsocket(fd, FIONBIO, &block) == SOCKET_ERROR) {
      inno_log_error("Error: ioctlsocket FIONBIO");
      InnoUtils::close_fd(fd);
      return -8;
    }
#endif
    int res = connect(fd, (struct sockaddr *)&serv_addr,
                      sizeof(serv_addr));
    if (res < 0) {
#ifdef _WIN32
      if (GetLastError() == WSAEWOULDBLOCK) {
        do {
          fd_set myset;
          struct timeval *tp = NULL;
          if (read_timeout_sec > 0) {
            tv.tv_sec = static_cast<int>(read_timeout_sec);
            tv.tv_usec = (read_timeout_sec - tv.tv_sec) * 1000000;
            tp = &tv;
          }
          FD_ZERO(&myset);
          FD_SET((unsigned int)fd, &myset);
          res = select(fd + 1, NULL, &myset, NULL, tp);
#elif defined(__APPLE__)
      if (errno == EINPROGRESS) {
        do {
          fd_set myset;
          struct timeval *tp = NULL;
          if (read_timeout_sec > 0) {
            tv.tv_sec = static_cast<int>(read_timeout_sec);
            tv.tv_usec = (read_timeout_sec - tv.tv_sec) * 1000000;
            tp = &tv;
          }
          FD_ZERO(&myset);
          FD_SET((unsigned int)fd, &myset);
          res = select(fd + 1, NULL, &myset, NULL, tp);
#else
      if (errno == EINPROGRESS) {
        do {
          int epoll_fd = epoll_create(1);
          inno_log_verify(epoll_fd != -1,
                          "epoll_create failed, epoll_fd = %d, "
                          "error = %d",
                          epoll_fd, errno);
          struct epoll_event event {};
          event.events = EPOLLOUT;  // Listen for writable events
          event.data.fd = fd;
          res = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event);
          inno_log_verify(res != -1,
                          "epoll_ctl failed, epoll_fd = %d, fd = %d,"
                          " error = %d",
                          epoll_fd, fd, errno);
          struct epoll_event events[1];
          res = epoll_wait(epoll_fd, events, 1, static_cast<int>(read_timeout_sec * 1000));
          close(epoll_fd);
#endif  // _WIN32

          if (res < 0) {
            if (errno != EINTR) {
              // cannot connect
              // do not print
              inno_log_error_errno("read %d", errno);
              InnoUtils::close_fd(fd);
              return -10;
            } else {
              continue;
            }
          } else if (res > 0) {
            // Socket selected for write
#if !(defined(__MINGW64__) || defined(_WIN32))
            int valopt = 0;
            socklen_t lon = sizeof(int);
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR,
                           reinterpret_cast<void*>(&valopt), &lon) < 0) {
              inno_log_error("Error: getsockopt");
              InnoUtils::close_fd(fd);
              return -11;
            }
            // Check the value returned...
            if (valopt == ECONNREFUSED) {
              usleep(10000);
              // inno_log_error("ECONNREFUSED %f", read_timeout_sec);
              read_timeout_sec -= 0.01;
              if (read_timeout_sec <= 0.00001) {
                inno_log_error("ECONNREFUSED");
                InnoUtils::close_fd(fd);
                return -17;
              }
              connected = false;
              break;
            } else if (valopt != 0) {
              // inno_log_error("Error: getsockopt return %d %s",
              //              valopt, strerror(valopt));
              inno_log_warning("connect valopt %d", valopt);
              InnoUtils::close_fd(fd);
              return -12;
            } else {
              // connected
              connected = true;
              break;
            }
#else
            connected = true;
            break;
#endif
          } else {
            inno_log_error_errno("get_connection timeout for %0.5fs, fd: %d",
                                  read_timeout_sec, fd);
            // connect timeout
            InnoUtils::close_fd(fd);
            return -13;
          }
        } while (1);
      } else {
        // cannot connect
        // do not print
        inno_log_error_errno("error=%d", errno);
        InnoUtils::close_fd(fd);
        return -14;
      }
    } else {
      // connected
      connected = true;
    }
  }

#if !(defined(__MINGW64__) || defined(_WIN32))
  int flag;
  // set to blocking mode
  if ((flag = fcntl(fd, F_GETFL, NULL)) < 0) {
    inno_log_error("Error: fcntl F_GETFL");
    InnoUtils::close_fd(fd);
    return -15;
  }
  flag &= (~O_NONBLOCK);
  if (fcntl(fd, F_SETFL, flag) < 0) {
    inno_log_error("Error: fcntl F_SETFL");
    InnoUtils::close_fd(fd);
    return -16;
  }
#else
  u_long flag = 0;
  if (ioctlsocket(fd, FIONBIO, &flag) == SOCKET_ERROR) {
    inno_log_error("Error: ioctlsocket FIONBIO");
    InnoUtils::close_fd(fd);
    return -16;
  }
#endif

  return fd;
}

}  // namespace innovusion
