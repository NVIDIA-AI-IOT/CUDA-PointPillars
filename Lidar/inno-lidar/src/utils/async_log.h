/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 * @File Name: async_log.h
 * @brief :  This log module  will provide the async log.
 *           The caller can choose the mode of sync or  async
 * @Version : 1.0
 * @Creat Date : 2022-03-18
 */

#ifndef UTILS_ASYNC_LOG_H_
#define UTILS_ASYNC_LOG_H_

#include <limits.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <set>

#include "utils/utils.h"
#include "utils/consumer_producer.h"
#include "utils/mem_pool_manager.h"
#include "utils/log.h"

namespace innovusion {
typedef struct {
  size_t  normal_ret;
  size_t  block_ret;
  size_t  giveUp_ret;
  size_t  dropHead_ret;
  size_t  error_ret;
} AsyncLogDebugInfo;

// cross include and earlier declare
class MemPool;

/**
 * @class : Async Job Thread Manager
 * This async thread will do the job for log and
 * packet deliver in callback module
 * Maybe will be used in other module.
 */
class AsyncLogManager {
 public:
  /**
   * @brief AsyncLogManager constructor
   * @param name            Thread name
   * @param priority        Thread priority
   * @param consume_func    Consume function
   * @param consume_context Consume funciton context
   * @param queue_size      Consume queue size
   * @param bufNum          Mempool buffer number
   * @param bufSize         Mempool buffer size
   */
  AsyncLogManager(const char *name,
                  int priority,
                  ConsumeFunc consume_func,
                  void *consume_context,
                  int queue_size,
                  size_t bufNum = 50,
                  size_t bufSize = 1024*4);
  // Destroy
  virtual ~AsyncLogManager();

  /**
   * @brief Alloc buffer from mempool or malloc if mempool has something wrong
   * @param len Buffer size
   * @return Buffer pointer in logContextInfo format
   */
  logContextInfo* alloc_buffer(size_t len);

  /**
   * @brief Free the buffer to mempool or just free it if it's not a buffer from mempool
   * @param buffer_p Buffer to be freed
   */
  void free_buffer(logContextInfo *buffer_p);

  /**
   * @brief Start the async thread and add job to queue
   * @param log_info_p  Log info to be added to thread job
   * @param discardable Whether the log is discardable
   * @return Return 0 for success, others for error
   */
  int add_log_job(const logContextInfo &log_info_p,
                  bool discardable);

  /**
   * @brief Check if current thread a log worker
   * @return Return true if current thread is a log worker, false otherwise
   */
  bool is_log_worker();

  // get debug info
  // AsyncLogDebugInfo* get_debug_info();

  /**
   * @brief Pause adding jobs to log worker and wait until all jobs are processed or dropped
   */
  void flush_and_pause() {
    cp_async_job_thread_->flush_and_pause();
  }

  /**
   * @brief Resume adding jobs to log worker
   */
  void resume() {
    cp_async_job_thread_->resume();
  }

  // print stats

  /**
   * @brief Print stats if it's a async log worker
   */
  void print_stats() {
    if (cp_async_job_thread_) {
      cp_async_job_thread_->print_stats();
    }
  }

#ifdef ASYCLOG_UNITEST_ENABLE
  // add this for debug
  MemPool * get_mempool() {
    return poolP_;
  }
#endif

  // shutdown

  /**
   * @brief Shutdown the async log worker and wait until all workers are stopped
   */
  void shutdown();
  // add the static process function to the consumer of async log

  /**
   * @brief Process the job in async log worker
   * @param job    Job to be processed
   * @param ctx    Address of InnoLog
   * @param prefer Whether the job is prefered
   * @return Return 0 for success, others for error
   */
  static int process(void *job, void *ctx, bool prefer);
//
 private:
  // add the default value
  const int worker_num_ = 1;
  const int hi_cp_queue_size_ = 0;
  const size_t cpusetsize_ = 0;
  const int alignment_ = 32;
  // handle for job thread
  ConsumerProducer *cp_async_job_thread_;
  // Property
  // The buffer pool  number
  size_t async_job_bufnumber_;  // default 50
  // Single buffer size
  size_t async_job_bufsize_;  // default 4k
  // handle for buffer pool
  MemPool * poolP_;
  // worker id
  std::thread *threads_p_;
  // default status is false
  bool async_running_ = false;
  // mutex
  // AsyncLogDebugInfo  debug_info_;
};

}  // namespace innovusion
#endif  // UTILS_ASYNC_LOG_H_
