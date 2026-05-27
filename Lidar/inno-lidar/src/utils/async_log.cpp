/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 * @File Name: async_log.cpp
 * @Version : 1.0
 * @Creat Date : 2022-03-18
 */
#include <utility>

#include "utils/async_log.h"
#include "utils/config.h"
#include "utils/log.h"
#include "utils/utils.h"
#include "utils/consumer_producer.h"

// this Async job thread will be used by
// log and callback
namespace innovusion {

/**
 * @brief : contructor for AsyncLogManager
 * as a lib module, it will be create by multiple instance.
 * the creator will config some parameters
 */
AsyncLogManager::AsyncLogManager(const char *name,
                      int priority,
                      ConsumeFunc consume_func,
                      void *consume_context,
                      int queue_size,
                      size_t buf_num,
                      size_t buf_size):
                      async_job_bufnumber_(buf_num),
                      async_job_bufsize_(buf_size) {
  // Now the buffer pool size ==  ConsumerProducer queue size
  // create the buffer pool
  poolP_ = new MemPool("AsyncJob_memory_pool",
                        async_job_bufsize_,
                        async_job_bufnumber_,
                        alignment_,
                        true);
  // check the pool pointer valid
  inno_log_verify(poolP_, "AsyncJob_memory_pool");

  static const uint32_t kMaxCurrentThreads = 20;
  inno_log_verify((ssize_t)buf_num >=
                  queue_size + worker_num_ + kMaxCurrentThreads,
                  "buf_number too small");

  // create the sync thread
  // make sure the prefer_queue_size >= 1
  // and make sure the job can be job.
  cp_async_job_thread_ = new ConsumerProducer(
      name, priority,
      worker_num_, consume_func, consume_context,
      queue_size, queue_size,
      hi_cp_queue_size_, cpusetsize_, NULL,
      false /* no log */);

  // check the cp_async_job_thread_ valid
  inno_log_verify(cp_async_job_thread_, "cp_async_job_thread_");
  // init sync id to 0
  // start the thread;
  cp_async_job_thread_->start();
  // update the thread
  threads_p_ = cp_async_job_thread_->get_threads();

  inno_log_with_level(INNO_LOG_LEVEL_INFO,
              "LIDAR Log Async Thread work : %s",
              "True");

  // debug info init
  // memset(&debug_info_, 0, sizeof(AsyncLogDebugInfo));
  // ready
  async_running_ = true;
}

/**
 * @brief : Destructor Async Log Manager object
 */
AsyncLogManager::~AsyncLogManager() {
  // disable the async log
  async_running_ = false;

  // destructor the thread
  if (cp_async_job_thread_) {
    // do the shutdown
    delete cp_async_job_thread_;
    cp_async_job_thread_ = NULL;
  }

  // destructor the buffer pool
  if (poolP_) {
    delete poolP_;
    poolP_ = NULL;
  }
}

/**
 * @brief : asyncLog_shutdown
 */
void AsyncLogManager::shutdown() {
  cp_async_job_thread_->shutdown();
}

/**
 * @brief : add the log  async enqueue failed and need release buffer
 * @param  log_info_p  : log
 * @return int :0: normal enqueue 1: block enqueue 2: give up enqueue
 *         3: discard head and enqueue
 */
int AsyncLogManager::add_log_job(const logContextInfo &log_info_p,
                                 bool discardable) {
  int ret = 0;
  logContextInfo *buffer_p = 0;
  // calculate the context
  int head1_len = log_info_p.head1_len;
  int head2_len = log_info_p.head2_len;
  int message_len = log_info_p.body_len;
  int total_context_len = head1_len + head2_len + message_len;

  // the buffer len =  len of logContextInfo + len of context + 3*(0-end)
  uint64_t base_size = sizeof(logContextInfo) + 3;
  buffer_p = alloc_buffer(base_size + total_context_len);

  if (buffer_p != NULL) {
    // clear header
    memset(buffer_p, 0, base_size);
    // encode level
    buffer_p->level = log_info_p.level;
    buffer_p->level_len = log_info_p.level_len;
    // update the info (level,head1,head2,body and lens)
    memcpy(buffer_p->payload,
            log_info_p.head1_p, head1_len);
    // +1 '\n'
    buffer_p->payload[head1_len] = 0;
    buffer_p->head1_len = head1_len;
    buffer_p->head1_p = buffer_p->payload;
    // encode head2
    memcpy((buffer_p->payload + head1_len + 1),
            log_info_p.head2_p, head2_len);
    // +1 '\n'
    buffer_p->payload[head1_len + 1 + head2_len] = 0;
    buffer_p->head2_p = buffer_p->payload + 1 + head1_len;
    buffer_p->head2_len = head2_len;
    // encode body
    memcpy((buffer_p->payload + head1_len + head2_len + 2),
            log_info_p.body_p, message_len);
    // +1 '\n'
    buffer_p->payload[head1_len + head2_len + message_len + 2] = 0;
    buffer_p->body_p = buffer_p->payload + head1_len + head2_len + 2;
    buffer_p->body_len = message_len;
    // Sync log for INNO_LOG_LEVEL_CRITICAL and below
    if (buffer_p->level <= INNO_LOG_LEVEL_CRITICAL) {
      ret = cp_async_job_thread_->add_job_wait_done(
          reinterpret_cast<void*>(buffer_p), false);
    } else {
      // error cannot be dropped
      ret = cp_async_job_thread_->add_job(
            reinterpret_cast<void*>(buffer_p), false,
            buffer_p->level <= INNO_LOG_LEVEL_ERROR ||
            !discardable);
    }
  } else {
    ret = -1;
  }
  /* add the debug log
  switch (ret) {
  case 0:
    debug_info_.normal_ret++;
    break;
  case 1:
    debug_info_.block_ret++;
    break;
  case 2:
    debug_info_.giveUp_ret++;
    break;
  case 3:
    debug_info_.dropHead_ret++;
    break;
  default:
    debug_info_.error_ret++;
    break;
  }
  */
  return ret;
}

/**
 * @brief :Get debug info
 */
// AsyncLogDebugInfo * AsyncLogManager::get_debug_info() {
//   return &debug_info_;
// }

/**
 * @brief :  alloc the buffer
 * @return void*       : return the buffer pointer
 */
logContextInfo *AsyncLogManager::alloc_buffer(size_t len) {
  void * buffer_p = NULL;

  if (len <= async_job_bufsize_) {
    // alloc buffer from pool
    buffer_p = poolP_->alloc();
  }

  if (buffer_p == NULL) {
    // either pool is empty or len is too big
    buffer_p = malloc(len);
  }

  // buffer from malloc
  return reinterpret_cast<logContextInfo *>(buffer_p);
}

/**
 * @brief : delete the  buffer
 * @param  buffer_p  :  logContextInfo *
 */
void AsyncLogManager::free_buffer(logContextInfo *buffer_p) {
  //
  void * buffer_p_ = reinterpret_cast<void *>(buffer_p);
  if (poolP_->is_manager_of(buffer_p_) == true) {
    poolP_->free(buffer_p_);
  } else {
    ::free(buffer_p_);
  }
}

/**
 * @brief : is_log_worker
 * @return true    : if the caller == async thread
 * @return false   : log to async thread
 */
bool AsyncLogManager::is_log_worker() {
  // async thread work
  if (async_running_ == true) {
    auto id = std::this_thread::get_id();
    // if the pthread is async log thread.
    // write the log to file directly
    for (int index = 0; index < worker_num_; index++) {
      if (threads_p_[index].get_id() == id) {
        return true;
      }
    }
    // use async thread
    return false;
  } else {
    // log directly
    return true;
  }
}

}  // namespace innovusion
