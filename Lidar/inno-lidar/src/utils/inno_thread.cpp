/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/inno_thread.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if !(defined(_QNX_) || defined(__MINGW64__) || defined(_WIN32))
#include <sys/syscall.h>
#endif
#include <sys/types.h>

#include "utils/log.h"
#include "utils/utils.h"

namespace innovusion {
InnoThread::InnoThread(const char *name, int priority,
                       int worker_num,
                       void *(*func_main)(void *),
                       void *func_context,
                       size_t cpusetsize,
                       const cpu_set_t *cpuset)
    : priority_(priority)
    , worker_num_(worker_num)
    , func_main_(func_main)
    , func_context_(func_context)
    , cpusetsize_(cpusetsize)
    , cpuset_(cpuset)
    , shutdown_(true) {
  name_ = strdup(name);
  inno_log_verify(name_, "%s name", name);
  start_time_ = InnoUtils::get_time_ns();
}

InnoThread::~InnoThread() {
  inno_log_verify(shutdown_.load(), "%s shutdown = %d", name_, shutdown_.load());
  free(name_);
  name_ = NULL;
}

void *InnoThread::inno_thread_func_(void *context) {
  InnoThread *cp = reinterpret_cast<InnoThread *>(context);
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
  pid_t pid = 0;
  if (cp->cpuset_ && cp->cpusetsize_) {
    const pthread_t tid = pthread_self();
    pthread_setaffinity_np(tid, cp->cpusetsize_, cp->cpuset_);
  }
  pid = syscall(SYS_gettid);
#else
  (void)cp->cpuset_;
  (void)cp->cpusetsize_;
  uint32_t pid = 0;
#endif

  inno_log_info("thread %s starts. pid=%d target_priority=%d", cp->name_, pid, cp->priority_);
  InnoUtils::set_self_thread_priority(cp->priority_);

  cp->func_main_(cp->func_context_);

  return NULL;
}

void InnoThread::start() {
  shutdown_ = false;
  threads_.resize(worker_num_);
  for (int i = 0; i < worker_num_; i++) {
    threads_.emplace_back(inno_thread_func_, this);
  }
}

void InnoThread::shutdown() {
  inno_log_verify(!shutdown_.load(), "%s shutdown = %d", name_, shutdown_.load());
  shutdown_ = true;
  cond_.notify_one();
  // join threads
  for (std::thread &t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

/**
 * Should ONLY be called by work thread itself.
 * @param useconds
 */
void InnoThread::timed_wait(uint32_t useconds) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto timeout = std::chrono::microseconds(useconds);
  cond_.wait_for(lock, timeout);
}

}  // namespace innovusion
