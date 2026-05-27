/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_

#if !defined (__MINGW64__) && defined(_WIN32)
#include <io.h>
#define ssize_t __int64
#else
#include <unistd.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef _WIN32
#include <pthread.h>
#endif

#if !(defined(_QNX_) || defined(__MINGW64__) || defined(_WIN32))
#include <execinfo.h>
#endif

#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <mutex>
#include <condition_variable>
#include "utils/inno_lidar_log.h"
#include "utils/consumer_producer.h"

#ifdef __APPLE__
#include <os/signpost.h>
#define INNER_BEGIN_LOG(subsystem, category, name) \
os_log_t m_log_##name = os_log_create((#subsystem), (#category));\
os_signpost_id_t m_spid_##name = os_signpost_id_generate(m_log_##name);\
os_signpost_interval_begin(m_log_##name, m_spid_##name, (#name));

#define INNER_END_LOG(name) \
os_signpost_interval_end(m_log_##name, m_spid_##name, (#name));
#endif

namespace innovusion {
// define the format of log
// cross include and earlier declare
class AsyncLogManager;
typedef struct {
  enum InnoLogLevel level;
  char *head1_p;
  int   head1_len;
  char *head2_p;
  int   level_len;
  int   head2_len;
  char *body_p;
  int   body_len;
  char  payload[0];
} logContextInfo;

/**
 * @brief RWLock
 */
#ifdef __MINGW64__
class RWLock {
 public:
  RWLock() : readers_count_(0), writer_active_(false), write_requests_(0) {
  }

  void acquire_read() {
    std::unique_lock<std::mutex> lock(mutex_);
    read_cv_.wait(lock, [this]() { return !writer_active_ && write_requests_ == 0; });
    ++readers_count_;
  }

  void release_read() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (--readers_count_ == 0) {
      if (write_requests_ > 0) {
        write_cv_.notify_one();
      } else {
        read_cv_.notify_all();
      }
    }
  }

  void acquire_write() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++write_requests_;
    write_cv_.wait(lock, [this]() { return !writer_active_ && readers_count_ == 0; });
    --write_requests_;
    writer_active_ = true;
  }

  void release_write() {
    std::unique_lock<std::mutex> lock(mutex_);
    writer_active_ = false;
    if (write_requests_ > 0) {
      write_cv_.notify_one();
    } else {
      read_cv_.notify_all();
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable read_cv_;
  std::condition_variable write_cv_;
  int readers_count_;
  bool writer_active_;
  int write_requests_;
};
#else
class RWLock {
 public:
  RWLock() {
#ifndef _WIN32
    rw_lock_ = PTHREAD_RWLOCK_INITIALIZER;
#endif
  }

  /**
   * @brief Acquire the read lock
   */
  void acquire_read() {
#ifdef _WIN32
    mutex_.lock();
#else
    pthread_rwlock_rdlock(&rw_lock_);
#endif
  }

  /**
   * @brief Release the read lock
   */
  void release_read() {
#ifdef _WIN32
    mutex_.unlock();
#else
    pthread_rwlock_unlock(&rw_lock_);
#endif
  }
  /**
   * @brief Acquire the write lock
   */
  void acquire_write() {
#ifdef _WIN32
    mutex_.lock();
#else
    pthread_rwlock_wrlock(&rw_lock_);
#endif
  }

  /**
   * @brief Release the write lock
   */
  void release_write() {
#ifdef _WIN32
    mutex_.unlock();
#else
    pthread_rwlock_unlock(&rw_lock_);
#endif
  }

 private:
#ifdef _WIN32
  std::mutex mutex_;
#else
  pthread_rwlock_t rw_lock_;
#endif
};
#endif

typedef int (*process_job)(logContextInfo *job, bool prefer);
/**
 * @class RotateLogFile :
 * When the log file is larger than the max length(rotate_file_size_limit),
 * a new file will be created to do this.  the name of log file will do the rotation.
 * this class only is used by the InnoLog.
 */
class RotateLogFile {
 public:
// constructor :  init properties

  /**
   * @brief RotateLogFile constructor
   * @param rotate_file_base_file  The base name of rotation file
   * @param rotate_file_number     The max number of rotation files
   * @param rotate_file_size_limit The max size of file
   */
  RotateLogFile(const char *rotate_file_base_file,
             uint32_t rotate_file_number,
             uint64_t rotate_file_size_limit)
      : log_file_fd_(-1)
      , current_file_size_(0)
      , rotate_file_base_file_(rotate_file_base_file)
      , rotate_file_number_(rotate_file_number)
      , rotate_file_size_limit_(rotate_file_size_limit) {
  }
// Destructor
  ~RotateLogFile() {
    if (log_file_fd_ >= 0) {
    #if !defined (__MINGW64__) && defined(_WIN32)
      _close(log_file_fd_);
    #else
      close(log_file_fd_);
    #endif
      log_file_fd_ = -1;
    }
  }

  /**
   * @brief Write the log info to the file
   * @param info  The log info
   */
  void write(const logContextInfo &info);

 private:
  /**
   * @brief Rotate the log file
   *        Keep the latest log filename is the base name
   *        and the old log file will be labeled to the next number
   *        and remove the oldest log file if the number of log files is larger than the max number
   * @return Return 0 if success, others for error
   */
  int log_rotate_();

 private:
// Do the protection for rotation action
  std::mutex mutex_;
  // Current log ID.  >0 means valid  -1 : invalid
  int log_file_fd_ = -1;
  // Current file's size
  size_t current_file_size_;
// base name of rotation file
  std::string rotate_file_base_file_;
// the max number of rotation files
  uint32_t rotate_file_number_;
// the max size of file
  size_t rotate_file_size_limit_;
};

/**
 * @class : InnoLog
 * the InnLog will be a single instance per process.
 * it will be as the uniform API and provide to other process
 * to call the log function.
 */
class InnoLog {
 private:
// single instance
  InnoLog();
  ~InnoLog();
  /**
   * @brief Create and start the async log thread
   */
  void start_async_log_thread_(void);
  /**
   * @brief Shutdown and delete the async log thread
   */
  void shutdown_async_log_thread_(void);
// for the exception call back
#if !(defined(__MINGW64__) || defined(_WIN32))
  /**
   * @brief Signal handler
   * @param sig   signal
   * @param sinfo signal info
   * @param cont  context
   */
  static void sighandler_(int sig, siginfo_t *sinfo, void *cont);
  /**
   * @brief Print the backtrace
   *        It's temporary disabled
   * @param sig signal
   * @param ctx context
   */
  static void print_backtrace_(int sig, struct sigcontext *ctx);
#endif

 public:
  // disable the default assign
  InnoLog(InnoLog const&) = delete;
  void operator=(InnoLog const&) = delete;

  /**
   * @brief Handle signal interruption
   */
  static void setup_sig_handler();

  /**
   * @brief Get InnLog instance
   * @return Return the InnLog instance
   */
  static InnoLog &get_instance(void) {
    static InnoLog instance;
    return instance;
  }
  /**
   * @brief Write formated log info to the log file
   *        Send log info to the async log thread if the async log thread is running
   *        Otherwise, write the log info to the log file directly
   * @param  level : log level
   * @param  discardable : discardable flag
   * @param  file  : code file name
   * @param  line  : code line number
   * @param  fmt   : format string
   * @param  valist: params list
   * @return int : 0 : succeed -1: failed
   */
  int log_v(enum InnoLogLevel level,
            bool discardable,
            const char *file, int line,
            const char *fmt,
            va_list valist);

  /**
   * @brief Set inno log properties
   * @param out_fd                      Normal log file descriptor
   * @param error_fd                    Error log file descriptor
   * @param rotate_file_base_file       The base name of normal rotation file
   * @param rotate_file_number          The max number of normal rotation files
   * @param rotate_file_size_limit      The max size of normal file
   * @param rotate_file_base_file_err   The base name of error roration file
   * @param rotate_file_number_err      The max number of error rotation files
   * @param rotate_file_size_limit_err  The max size of error file
   * @param log_callback                The callback function for log
   * @param ctx                         The context for callback function
   * @param do_async                    Whether do the async log
   */
  void set_logs(int out_fd, int error_fd,
                const char *rotate_file_base_file,
                uint32_t rotate_file_number,
                uint64_t rotate_file_size_limit,
                const char *rotate_file_base_file_err,
                uint32_t rotate_file_number_err,
                uint64_t rotate_file_size_limit_err,
                InnoLogCallback log_callback,
                void *ctx,
                bool do_async);

  /**
   * @brief Set the log callback function
   * @param log_callback The callback function for log
   * @param ctx          The context for callback function
   */
  void set_logs_callback(InnoLogCallback log_callback,
                void *ctx);

  /**
   * @brief Write formated log info to the given file descriptor
   * @param fd   File descriptor
   * @param info Log info
   * @return Return positive number for written bytes, others for error
   */
  static int log_printv(int fd, const logContextInfo &info);

  // add the static process function to the consumer of async log
  /**
   * @brief The process function for async log.
   *        Write the log info to the log file.
   *        Output the log info to terminal if ASYCLOG_UNITEST_ENABLE not defined.
   *        Call the callback function if it's set.
   * @param job    Address of logContextInfo
   * @param ctx    Address of AsyncLogManager
   * @param prefer Whether prefer to process the job
   * @return Return 0 if success, others for error
   */
  static int process(void *job, void *ctx, bool prefer);
  // print the status
  /**
   * @brief Print the status of async log
   */
  void asynclog_info();
  // add the for debug
  /**
   * @brief  Get the async log manager
   * @return Return the async log manager
   */
  inline AsyncLogManager* get_asynclog_manager() {
    return asynclog_manager_p_;
  }
#ifdef ASYCLOG_UNITEST_ENABLE
  // add this for debug
  /**
   * @brief Set the hook function for async log process
   * @param process_job_hook The hook function
   */
  void set_process_hook_for_asynclog(process_job process_job_hook) {
    process_job_hook_ = process_job_hook;
  }
#endif

 public:
#if !(defined(__MINGW64__) || defined(_WIN32))
  static struct sigaction olddisp_segv, olddisp_fpe, olddisp_bus, newdisp;
#endif

 private:
  // terminal log
  int log_out_fd_;
  int log_error_fd_;
  std::mutex log_out_mutex_;
  std::mutex log_err_mutex_;
  void log2terminal_(const logContextInfo &info);
  // file log
  RotateLogFile *rotate_out_;
  RotateLogFile *rotate_error_;
  void log2file_(const logContextInfo &info);
  // callback log
  InnoLogCallback log_callback_;
  void *log_callback_ctx_;
  RWLock log_callback_lock_;

  void log2Callback_(const logContextInfo &info);
  // AsyncLogManager entry
  AsyncLogManager  * asynclog_manager_p_;
  RWLock asynclog_lock_;
  bool asynclog_exist_ = false;
  int process_job_(logContextInfo *job, bool prefer);
#ifdef ASYCLOG_UNITEST_ENABLE
  // add this for debug
  process_job process_job_hook_ = NULL;
#endif
};

}  // namespace innovusion

#endif  // UTILS_LOG_H_
