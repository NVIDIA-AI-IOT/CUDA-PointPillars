/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_INNO_THREAD_H_
#define UTILS_INNO_THREAD_H_

#include <stdint.h>
#include <time.h>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>
#if !defined(__MINGW64__) && defined(_WIN32)
typedef char cpu_set_t;
#elif defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__)
struct cpu_set_t;
#endif

namespace innovusion {
/**
 * @brief InnoThread
 */
class InnoThread {
 public:
  /**
   * @brief InnopThread constructor
   * @param name          Name of the thread
   * @param priority      Priority of the thread
   * @param worker_num    Number of threads to create
   * @param func_main     Process function of the thread
   * @param func_context  Context of the process function
   * @param cpusetsize    Size of the cpu set
   * @param cpuset        CPU set
   */
  InnoThread(const char *name, int priority,
             int worker_num,
             void *(*func_main)(void *),
             void *func_context,
             size_t cpusetsize,
             const cpu_set_t *cpuset);
  ~InnoThread();
  /**
   * @brief Create and start the threads
   */
  void start();
  /**
   * @brief Stop the threads and block until all threads are stopped
   */
  void shutdown();
  /**
   * @brief Check if the shutdown flag is set
   * @return Return true if the shutdown flag is set, otherwise false
   */
  bool has_shutdown() const {
    return shutdown_.load();
  }
  /**
   * @brief Thread wait for useconds
   * @param useconds Time to wait in microseconds
   */
  void timed_wait(uint32_t useconds);

 private:
  /**
   * @brief Thread function, it will call the main process function
   * @param context Address of InnoThread
   * @return Return NULL
   */
  static void *inno_thread_func_(void *context);

 private:
  char *name_;
  int priority_;
  int worker_num_;
  std::vector<std::thread> threads_;
  void *(*func_main_)(void *);
  void *func_context_;
  size_t cpusetsize_;
  const cpu_set_t *cpuset_ = NULL;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::atomic_bool shutdown_;
  size_t start_time_;
};
}  // namespace innovusion

#endif  // UTILS_INNO_THREAD_H_
