/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */
#include <fcntl.h>
#include <stdarg.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(_WIN32))
#ifndef __APPLE__
#include <syscall.h>
#endif
#include <sys/uio.h>
#include <sys/time.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <chrono>

#include "utils/log.h"
#include "utils/async_log.h"

enum InnoLogLevel inno_log_level_g = INNO_LOG_LEVEL_INFO;
// Nice log level. the fatal and critical do the sync
// and the other do async

const char *inno_log_header_g[] = {
  "[FATAL]",
  "[CRITI]",  // critical
  "[ERROR]",
  "[ TEMP]",
  "[ WARN]",
  "[DEBUG]",
  "[ INFO]",
  "[TRACE]",
  "[DETAL]",  // detail
  "[     ]",
};

namespace innovusion {

#if !(defined(__MINGW64__) || defined(_WIN32))
struct sigaction InnoLog::olddisp_segv, InnoLog::olddisp_fpe;
struct sigaction InnoLog::olddisp_bus, InnoLog::newdisp;
#endif

/**
 * @brief : the file writing function.
 * the file  **.txt newer than **1.txt, and **1.txt is newer than **2.txt
 * @param  info  : logContextInfo
 */
void RotateLogFile::write(const logContextInfo &info) {
  // write to external files with size limit and rotate
  if (rotate_file_number_ > 0 &&
      rotate_file_size_limit_ > 0 &&
      rotate_file_base_file_.size() > 0) {
    if (log_file_fd_ < 0) {
      log_rotate_();
    }

    if (log_file_fd_ >= 0) {
      int ret = InnoLog::log_printv(log_file_fd_, info);
      if (ret >= 0) {
        current_file_size_ += ret;
        if (current_file_size_ >= rotate_file_size_limit_) {
          close(log_file_fd_);
          log_file_fd_ = -1;
          current_file_size_ = 0;
        }
      }
    }
  }
  return;
}

/**
 * @brief : log rotate
 * @return int  : 0 succeed -1 failed
 */
int RotateLogFile::log_rotate_() {
  if (rotate_file_base_file_.size() == 0 ||
      rotate_file_size_limit_ == 0 ||
      rotate_file_number_ < 1) {
    return -1;
  }

  if (log_file_fd_ >= 0) {
    // still use the existing file
    return 0;
  }

  int fd = open(rotate_file_base_file_.c_str(),
                O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    return -2;
  }

  off_t sz = lseek(fd, 0, SEEK_END);
  if (sz < 0) {
    close(fd);
    return -3;
  }

  if (sz < (ssize_t)rotate_file_size_limit_) {
    // still have some space, just use this fd
    current_file_size_ = sz;
    log_file_fd_ = fd;
    return 0;
  }

  close(fd);

  // rotate files
  std::string base_name(rotate_file_base_file_);
  for (uint32_t dst = rotate_file_number_;
       dst > 0;
       dst--) {
    uint32_t src = dst - 1;
    std::string src_name = base_name;
    std::string dst_name = base_name + "." + std::to_string(dst);
    if (src > 0) {
      src_name += "." + std::to_string(src);
    }

    if (dst == rotate_file_number_) {
      remove(dst_name.c_str());
    } else {
      rename(src_name.c_str(), dst_name.c_str());
    }
  }

  // open new file
  fd = open(rotate_file_base_file_.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return -2;
  }

  current_file_size_ = 0;
  log_file_fd_ = fd;
  return 0;
}

InnoLog::InnoLog() {
  log_out_fd_ = 1;    // stdout
  log_error_fd_ = 2;  // stderr
  rotate_out_ = NULL;
  rotate_error_ = NULL;
  log_callback_ = NULL;
  log_callback_ctx_ = NULL;
  // delay to create the thread
  // init the rw lock

  asynclog_manager_p_ = NULL;
  asynclog_exist_ = false;
}

InnoLog::~InnoLog() {
  shutdown_async_log_thread_();


  if (rotate_out_) {
    delete rotate_out_;
    rotate_out_ = NULL;
  }
  if (rotate_error_) {
    delete rotate_error_;
    rotate_error_ = NULL;
  }
  // delete the asyncJobThread
  log_callback_ = NULL;
  log_callback_ctx_ = NULL;

  if (asynclog_manager_p_) {
    delete asynclog_manager_p_;
    asynclog_manager_p_ = NULL;
    asynclog_exist_ = false;
  }
}

/**
 * @brief :create and start the async log
 */
void InnoLog::start_async_log_thread_() {
  // delay to create the thread
  // bc the thread need use log id
  // make sure the pool size > queue size
  asynclog_manager_p_ = new AsyncLogManager(
                        "AsyncLogThread_Manager",
                        0, InnoLog::process,
                        &get_instance(),
                        60, 90, 4096);

  // enable the asyncLogExist
  asynclog_lock_.acquire_write();
  asynclog_exist_ = true;
  asynclog_lock_.release_write();

  inno_log_verify_no_print(asynclog_manager_p_);
}

/**
 * @brief :shutdown the async log thread
 */
void InnoLog::shutdown_async_log_thread_() {
  if (asynclog_manager_p_) {
    asynclog_lock_.acquire_write();
    asynclog_exist_ = false;
    asynclog_lock_.release_write();
    asynclog_manager_p_->shutdown();
    delete asynclog_manager_p_;
    asynclog_manager_p_ = NULL;
  }
}

#if !(defined(__MINGW64__) || defined(_WIN32))
void InnoLog::print_backtrace_(int sig, struct sigcontext *ctx) {
// #define TRACE_SIZE 100
//   int j, nptrs;
//   void *buffer[TRACE_SIZE];
//   char **strings;

  inno_log_error("pid=%d: got signal %d",
                  getpid(), sig);
  // nptrs = backtrace(buffer, TRACE_SIZE);

  // for (j = 0; j < nptrs; j++) {
  //    inno_log_error("backtrace: [%d]: %s", j, buffer[j]);
  //}

  // strings = backtrace_symbols(buffer, nptrs);
  // if (strings == NULL) {
  //   inno_log_error_errno("backtrace_symbols %s", "");
  //   return;
  // }

  // for (j = 0; j < nptrs; j++) {
  //   inno_log_error("backtrace: [%d]: %s", j, strings[j]);
  // }
  // to avoid possible failure, don't free
  // free(strings);
}

/**
 * @brief :  exception hook
 * @param  sig : sig id
 * @param  sinfo : sig info
 * @param  cont :  context
 */
void InnoLog::sighandler_(int sig, siginfo_t *sinfo, void *cont) {
  // struct sigcontext *ctx = (struct sigcontext *)cont;
  print_backtrace_(sig, (struct sigcontext *)cont);
  if (sig == SIGSEGV) {
    abort();
    // (*olddisp_segv.sa_sigaction)(sig, sinfo, cont);
  } else if (sig == SIGBUS) {
    (*olddisp_bus.sa_sigaction)(sig, sinfo, cont);
  } else {
    (*olddisp_fpe.sa_sigaction)(sig, sinfo, cont);
  }
  inno_log_info("done signal %d", sig);
  // exit(EXIT_FAILURE);
}
#endif  // !__MINGW64__

/**
 * @brief :
 * @param  fd   : FD
 * @param  info : loginfo
 * @return int  : writing len
 */
int InnoLog::log_printv(int fd, const logContextInfo &info) {
  int written;
#if !(defined(__MINGW64__) || defined(_WIN32))
  char space = ' ';
  char ret = '\n';
  struct iovec ios[6];
  ios[0].iov_base = info.head1_p;
  ios[0].iov_len = info.head1_len;
  ios[1].iov_base = &space;
  ios[1].iov_len = sizeof(space);
  ios[2].iov_base = info.head2_p;
  ios[2].iov_len = info.head2_len;
  ios[3].iov_base = &space;
  ios[3].iov_len = sizeof(space);
  ios[4].iov_base = info.body_p;
  ios[4].iov_len = info.body_len;
  ios[5].iov_base = &ret;
  ios[5].iov_len = sizeof(ret);
  written = writev(fd, ios, 6);
  if (written < 0) {
    perror("writev");
  }
#else
  char buffer[10240];
  snprintf(buffer, sizeof(buffer),
           "%s %s %s\n", info.head1_p, info.head2_p, info.body_p);
  buffer[sizeof(buffer) - 2] = '\n';
  buffer[sizeof(buffer) - 1] = 0;
  written = write(fd, buffer, strlen(buffer));
  if (written < 0) {
    perror("dprintf");
  }
#endif
  return written;
}

/**
 * @brief : Log_v
 * @param level -name log level
 * @param file  -name lode file
 * @param line  -name code line number
 * @param fmt     - name  format valist
 * @param valist  - values
 */
int InnoLog::log_v(enum InnoLogLevel level,
                   bool discardable,
                   const char *file, int line,
                   const char *fmt,
                   va_list valist) {
  // define max len
  static const int Kmax_head1_len = 64;
  char tbuffer[32];
  char head1_buff[Kmax_head1_len];
  struct tm* tm_info = NULL;
  struct tm result_time;

  auto duration = std::chrono::system_clock::now().time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  time_t now_sec = seconds.count();
  int milli = milliseconds.count() % 1000;

#if !(defined(__MINGW64__) || defined(_WIN32))
  tm_info = localtime_r(&now_sec, &result_time);
#else
  tm_info = localtime_s(&result_time, &now_sec) == 0 ?
            &result_time : NULL;
#endif
  if (tm_info) {
    strftime(tbuffer, sizeof(tbuffer) - 1, "%Y-%m-%d %H:%M:%S", tm_info);
    tbuffer[sizeof(tbuffer) - 1] = 0;
  } else {
    tbuffer[0] = 0;
  }

  int head1_len = snprintf(head1_buff, Kmax_head1_len, "%s.%03d",
                          tbuffer, milli);
  // trim
  if (head1_len > (Kmax_head1_len - 1)) {
    head1_len = Kmax_head1_len - 1;
  }
  head1_buff[head1_len] = 0;

  // get head2
  uint32_t tid = 0;
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
  tid = syscall(SYS_gettid);
#endif
  static const int kMaxHeaderSize = 100;
  char header2_buff[kMaxHeaderSize];
  int head2_len = snprintf(header2_buff,
                      kMaxHeaderSize, "%s %d %s:%d",
                      inno_log_header_g[level], tid, file, line);
  if (head2_len > (kMaxHeaderSize - 1)) {
    head2_len = kMaxHeaderSize - 1;
  }
  header2_buff[head2_len] = 0;

  // offset for simple head2
  int levellen = strlen(inno_log_header_g[level]);

  char *buffer = NULL;
  char *buffer_alloc = NULL;
  static const ssize_t kMaxSize = 10000;
  char buffer_stack[kMaxSize];
  // try to write to buffer_stack
  va_list valist_cpy;
  va_copy(valist_cpy, valist);
  int snp_size = vsnprintf(buffer_stack, kMaxSize, fmt, valist_cpy);
  va_end(valist_cpy);
  // data is larger than kMaxSize
  if (snp_size >= kMaxSize) {
    buffer_alloc = reinterpret_cast<char *>(malloc(snp_size + 1));
    if (buffer_alloc) {
      va_copy(valist_cpy, valist);
      snp_size = vsnprintf(buffer_alloc, (snp_size + 1), fmt, valist_cpy);
      va_end(valist_cpy);
      buffer_alloc[snp_size] = 0;
      buffer = buffer_alloc;
    } else {
      return -1;
    }
  } else {
    buffer = buffer_stack;
  }
  if ((snp_size > 0) && (buffer[snp_size - 1] == '\n')) {
    // remove unnecessary \n
    buffer[snp_size - 1] = 0;
    // update the len
    snp_size = snp_size - 1;
  }

  logContextInfo logInfo;
  // level
  logInfo.level = level;
  // head1
  logInfo.head1_p = head1_buff;
  logInfo.head1_len = head1_len;
  // head2
  logInfo.head2_p = header2_buff;
  logInfo.head2_len = head2_len;
  // offset
  logInfo.level_len = levellen;
  // body
  logInfo.body_p = buffer;
  logInfo.body_len = snp_size;

  // add the rd lock
  asynclog_lock_.acquire_read();
  if ((asynclog_exist_ == true) &&\
      (asynclog_manager_p_ != NULL) &&\
      (true != asynclog_manager_p_->is_log_worker())) {
    asynclog_manager_p_->add_log_job(logInfo, discardable);
    asynclog_lock_.release_read();
  } else {
    asynclog_lock_.release_read();
#ifndef ASYCLOG_UNITEST_ENABLE
    // log to terminal
    log2terminal_(logInfo);
#endif
    // log to file
    log2file_(logInfo);
    // log to callback
    log2Callback_(logInfo);
  }

  // free the buffer
  if (buffer_alloc) {
    free(buffer_alloc);
  }

  return 0;
}

/**
 * @brief : log2Callback_
 * @param  info ：Ref logContextInfo
 */
void InnoLog::log2Callback_(const logContextInfo &info) {
  asynclog_lock_.acquire_read();
  if (log_callback_) {
    log_callback_(log_callback_ctx_,
                  info.level, info.head1_p,
                  (info.head2_p + info.level_len),
                  info.body_p);
  }
  asynclog_lock_.release_read();
}

/**
 * @brief : log2file_
 * @param  info : ref logContextInfo
 */
void InnoLog::log2file_(const logContextInfo &info) {
  if (rotate_out_) {
    rotate_out_->write(info);
  }

  if (info.level <= INNO_LOG_LEVEL_DEBUG) {
    if (rotate_error_) {
      rotate_error_->write(info);
    }
  }
}

/**
 * @brief : log2terminal_
 * @param  info : ref logContextInfo
 */
void InnoLog::log2terminal_(const logContextInfo &info) {
  if (info.level <= INNO_LOG_LEVEL_DEBUG) {
    if (log_error_fd_ >= 0) {
      std::unique_lock<std::mutex> lck(log_out_mutex_);
      log_printv(log_error_fd_, info);
    }
  } else {
    // write non-error
    if (log_out_fd_ >= 0) {
      std::unique_lock<std::mutex> lck(log_err_mutex_);
      log_printv(log_out_fd_, info);
    }
  }
}

/**
 * @brief :  process the log from other threads
 * @param  in_job   :  the log packet
 * @param  ctx      :  single inno log instance
 * @param  prefer   :  High priority or low
 * @return int      :  the job ID
 */
int InnoLog::process(void *in_job, void *ctx,
                     bool prefer) {
  InnoLog *s = reinterpret_cast<InnoLog *>(ctx);
  // process job
  return s->process_job_(reinterpret_cast<logContextInfo *>(in_job), prefer);
}

/**
 * @brief :  process_job_
 * @param  job    : log packet
 * @param  prefer  :
 * @return int  : 0: ok -1: error
 */
int InnoLog::process_job_(logContextInfo *job_p, bool prefer) {
  // check job
  inno_log_verify_no_print(job_p, "job");
#ifdef ASYCLOG_UNITEST_ENABLE
  if (process_job_hook_) {
    process_job_hook_(job_p, prefer);
  }
#endif
  if (prefer == true) {
  // get level
#ifndef ASYCLOG_UNITEST_ENABLE
    log2terminal_(*job_p);
#endif
    log2file_(*job_p);

    log2Callback_(*job_p);
  }
  // always free buffer
  asynclog_manager_p_->free_buffer(job_p);
  return 0;
}

/**
 * @brief : asynclog_info
 */
void InnoLog::asynclog_info() {
  if (asynclog_manager_p_) {
    asynclog_manager_p_->print_stats();
  }
}

/**
 * @brief : set_logs
 */
void InnoLog::set_logs(int out_fd, int error_fd,
                       const char *rotate_file_base_file,
                       uint32_t rotate_file_number,
                       uint64_t rotate_file_size_limit,
                       const char *rotate_file_base_file_err,
                       uint32_t rotate_file_number_err,
                       uint64_t rotate_file_size_limit_err,
                       InnoLogCallback log_callback, void *ctx,
                       bool do_async) {
  inno_log_verify(asynclog_exist_ == false &&
                  asynclog_manager_p_ == NULL,
                  "cannot do set_logs() more than once for async logger");

  log_out_fd_ = out_fd;
  log_error_fd_ = error_fd;

  if (rotate_file_base_file &&
    strlen(rotate_file_base_file) > 0) {
    // set_logs may be called multi times, avoid fd/mem leaking
    if (rotate_out_) {
      delete rotate_out_;
      rotate_out_ = nullptr;
    }

    // create rotate file
    rotate_out_ = new RotateLogFile(rotate_file_base_file,
                                    rotate_file_number,
                                    rotate_file_size_limit);
    inno_log_verify(rotate_out_, "rotate_out_");
  }

  if (rotate_file_base_file_err && strlen(rotate_file_base_file_err) > 0) {
    if (rotate_error_) {
      delete rotate_error_;
      rotate_error_ = nullptr;
    }

    // create rotate log error
    rotate_error_ = new RotateLogFile(rotate_file_base_file_err,
                                      rotate_file_number_err,
                                      rotate_file_size_limit_err);
    inno_log_verify(rotate_error_, "rotate_error_");
  }

  set_logs_callback(log_callback, ctx);

  if (do_async) {
    start_async_log_thread_();
  }
}

/**
 * @brief : set_logs_callback
 */
void InnoLog::set_logs_callback(InnoLogCallback log_callback, void *ctx) {
  if (asynclog_manager_p_) {
    // make sure there is no un-processed job and pause Q
    asynclog_manager_p_->flush_and_pause();
  }
  log_callback_lock_.acquire_write();
  log_callback_ = log_callback;
  log_callback_ctx_ = ctx;
  log_callback_lock_.release_write();
  if (asynclog_manager_p_) {
    asynclog_manager_p_->resume();
  }
}

/**
 * @brief : setup_sig_handler
 */
void InnoLog::setup_sig_handler() {
#if !(defined(__MINGW64__) || defined(_WIN32))
  int status;

  memset(&newdisp, 0, sizeof(struct sigaction));
  newdisp.sa_sigaction = sighandler_;
  newdisp.sa_flags = SA_SIGINFO;

  status = sigaction(SIGSEGV, &newdisp, &olddisp_segv);
  if (status == -1) {
    inno_log_error_errno("sigaction SIGSEVG error: %s", "");
    exit(-1);
  }

  status = sigaction(SIGBUS, &newdisp, &olddisp_bus);
  if (status == -1) {
    inno_log_error_errno("sigaction SIGBUS error: %s", "");
    exit(-2);
  }

  status = sigaction(SIGFPE, &newdisp, &olddisp_fpe);
  if (status == -1) {
    inno_log_error_errno("sigaction SIGFPE error: %s", "");
    exit(-3);
  }

  inno_log_info("setup_sig_handler ready");
#else
  signal(SIGINT, [](int code) {
    exit(EXIT_FAILURE);
  });
#endif
}

}  // namespace innovusion

void inno_log_print(enum InnoLogLevel level,
                    bool discardable,
                    const char *file, int line,
                    const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  innovusion::InnoLog::get_instance().\
                       log_v(level, discardable,
                             file, line, fmt, valist);
  va_end(valist);
  return;
}

void inno_fprintf(int fd, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  char buffer[4096];
  int w = vsnprintf(buffer, sizeof(buffer), fmt, valist);
  inno_ignore_variable(w);
  buffer[sizeof(buffer) - 1] = 0;
  int r = write(fd, buffer, strlen(buffer));
  inno_ignore_variable(r);
  va_end(valist);
  return;
}
