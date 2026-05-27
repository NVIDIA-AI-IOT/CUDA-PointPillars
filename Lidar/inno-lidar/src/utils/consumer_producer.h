/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_CONSUMER_PRODUCER_H_
#define UTILS_CONSUMER_PRODUCER_H_


#include <stdint.h>
#include <time.h>

#include <thread>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <string>

#include "utils/inno_thread.h"

namespace innovusion {
/**
 * @brief ConsumerProducer process callback function
 * @param job       Job to process
 * @param context   Context of the process function
 * @param prefer    True if the job is preferred, otherwise false
 */
typedef int (*ConsumeFunc)(void *job, void *context, bool prefer);

class ConsumerProducer {
 private:
  enum Priority {
    PRIORITY_LOW = 0,
    PRIORITY_HIGH = 1,
    PRIORITY_MAX = 2,
  };

 private:
  class Job {
   public:
    Job()
        : job(NULL)
        , timestamp(0)
        , job_id(0)
        , not_discardable(false) {
    }
    Job(void *j, size_t t, uint64_t i, bool d)
        : job(j)
        , timestamp(t)
        , job_id(i)
        , not_discardable(d) {
    }

    void *job;
    size_t timestamp;
    uint64_t job_id;
    bool not_discardable;
  };

  class CpQueue {
   public:
    CpQueue(const char *name, int queue_size, bool allow_log);
    ~CpQueue();
    bool is_full() const {
      return full_;
    }
    bool is_empty() const {
      return empty_;
    }
    int queue_length() const {
      return count_;
    }
    void add(const Job &job);
    int pop(Job *job, bool peek_only = false);
    inline int peek(Job *job) {
      return pop(job, true);
    }

   private:
    inline bool do_log_() const {
      return allow_log_;
    }

    char *name_;
    int head_, tail_;
    bool full_, empty_;
    int queue_size_, count_;
    int64_t done_;
    Job *jobs_;
    bool allow_log_;
  };

 public:
  static void *consumer_thread_func_(void *context);

 public:
  /**
   * @brief ConsumerProducer constructor.
   *        Job in high priority queue will be processed first, and will be marked as preferred.
   *        Job in low priority queue will be marked as preferred
   *        if the size of low priority queue is less or equal than prefer_queue_size,
   *        others will be marked as not preferred.
   * @param name              Name of the ConsumerProducer
   * @param priority          Priority of the threads
   * @param worker_num        Number of threads to create
   * @param consume_func      Process function of the threads
   * @param consume_context   Context of the process function
   * @param queue_size        Size of the queue for jobs
   * @param prefer_queue_size Size of the prefered jobs in the low priority queue
   * @param hi_cp_queue_size  Size of the queue for high priority jobs
   * @param cpusetsize        Size of the CPU set
   * @param cpuset            CPU set to bind the threads
   * @param allow_log_        Allow logging
   */
  ConsumerProducer(const char *name, int priority,
                   int worker_num, ConsumeFunc consume_func,
                   void *consume_context,
                   int queue_size, int prefer_queue_size,
                   int hi_cp_queue_size,
                   size_t cpusetsize,
                   const cpu_set_t *cpuset,
                   bool allow_log_ = true);
  ~ConsumerProducer();
  /**
   * @brief Create threads and start processing jobs
   */
  void start();
  /**
   * @brief Add job to the queue
   * @param in              Job to add
   * @param high_priority   True if the job is high priority, otherwise false.
   *                        High priority job will be added to high priority queue,
   *                        otherwise low priority queue.
   *                        If the high priority queue is full, the high priority job will wait until
   *                        the queue is not full.
   * @param not_discardable True if the job is not discardable, otherwise false.
   *                        If the low priority queue is full, and not_discardable is true,
   *                        the low priority job will wait until the queue is not full.
   *                        If the low priority queue is full, and not_discardable is false,
   *                        then will comsume the oldest job if possible, otherwise will cosume the
   *                        current job immediately.
   * @return Return 0: normal enqueue
   *                1: block enqueue
   *                2: give up enqueue
   *                3: discard head and enqueue
   */
  inline int add_job(void *in, bool high_priority = false,
                     bool not_discardable = false) {
    return add_job_do_(in, high_priority, NULL, not_discardable);
  }
  /**
   * @brief Add job to the quque and wait until the job is processed
   * @param in            Job to add
   * @param high_priority True if the job is high priority, otherwise false.
   * @return Return 0: normal enqueue
   *                1: block enqueue
   *                2: give up enqueue
   *                3: discard head and enqueue
   */
  int add_job_wait_done(void *in, bool high_priority = false);
  /**
   * @brief Shutdown the threads and wait until all jobs are stopped
   */
  void shutdown();
  /**
   * @brief Pause adding jobs to the queue, and wait until all jobs are processed or dropped
   */
  void flush_and_pause();
  /**
   * @brief Resume adding jobs to the queue
   */
  void resume();
  /**
   * @brief Get the normal queue length
   * @return Return the normal queue length
   */
  inline int queue_length() const {
    return queue_.queue_length();
  }
  /**
   * @brief Get the normal queue max length
   * @return Return the normal queue max length
   */
  inline int max_queue_length() const {
    return max_queue_len_;
  }
  /**
   * @brief Get the work threads vector pointer
   * @return Return the work threads vector pointer
   */
  inline std::thread *get_threads() {
    return &threads_[0];
  }
  /**
   * @brief  Get the total number of dropped jobs
   * @return Return the total number of dropped jobs
   */
  inline size_t dropped_job_count() const {
    size_t r = 0;
    for (int i = 0; i < PRIORITY_MAX; i++) {
      r += dropped_job_[i];
    }
    return r;
  }
  /**
   * @brief  Get the total number of blocked jobs
   * @return Return the total number of blocked jobs
   */
  inline size_t blocked_job_count() const {
    size_t r = 0;
    for (int i = 0; i < PRIORITY_MAX; i++) {
      r += blocked_job_[i];
    }
    return r;
  }
  /**
   * @brief Print current status
   */
  void print_stats(void);
  /**
   * @brief Get current status string
   * @param buf       Buffer to store the status string
   * @param buf_size  Size of the buffer
   */
  void get_stats_string(char *buf, size_t buf_size);

 private:
  /**
   * @brief Get the job id of corresponding priority
   * @param idx Index of the priority
   * @return Return the job id of corresponding priority
   */
  uint64_t assign_job_id_(int idx) {
    return ++job_id_[idx];
  }
  /**
   * @brief Mark the last finished job id of corresponding priority
   * @param idx Index of the priority
   * @param id  Job id to mark
   */
  void update_done_job_id_(int idx, uint64_t id) {
    finished_job_id_[idx] = id;
  }
  /**
   * @brief Add job to the queue
   * @param in                Job to add
   * @param high_priority     True if the job is high priority, otherwise false.
   * @param job_id_out        Buffer to store job id of the added job
   * @param not_discardable   True if the job is not discardable, otherwise false.
   * @return  Return 0: normal enqueue
   *                 1: block enqueue
   *                 2: give up enqueue
   *                 3: discard head and enqueue
   */
  int add_job_do_(void *in, bool high_priority,
                  uint64_t *job_id_out, bool not_discardable);
  /**
   * @brief Get a job from high priority queue or low priority queue
   * @param job      Buffer to store the job
   * @param priorty  Buffer to store the priority of the job
   * @return Return the queue size where the job is from
   */
  int get_job_(Job *job,
               enum Priority *priorty);
  /**
   * @brief Should print log
   * @return Return true if should print log, false for not
   */
  inline bool do_log_() const {
    return allow_log_;
  }
  /**
   * @brief Get the total number of added jobs
   * @return Return the total number of added jobs
   */
  uint64_t added_job_total_();
  /**
   * @brief Get the total number of finished jobs
   * @return Return the total number of finished jobs
   */
  uint64_t finished_job_total_();
  /**
   * @brief Get the total number of dropped jobs
   * @return Return the total number of dropped jobs
   */
  uint64_t dropped_job_total_();

 private:
  char *name_;
  int priority_;
  CpQueue queue_;
  std::atomic_bool blocking_;
  std::atomic_bool paused_;
  int prefer_queue_size_;
  CpQueue hi_queue_;
  int worker_num_;
  std::vector<std::thread> threads_;
  ConsumeFunc consume_func_;
  void *consume_context_;
  std::mutex mutex_;
  std::condition_variable not_full_, not_empty_, job_done_cond_;
  std::atomic_int started_;
  std::atomic_bool shutdown_;
  size_t cpusetsize_;
  const cpu_set_t *cpuset_ = NULL;
  bool allow_log_;

  size_t start_time_;
  size_t total_wait_time_[PRIORITY_MAX];
  size_t total_process_time_[PRIORITY_MAX];
  size_t total_drop_time_[PRIORITY_MAX];
  size_t added_job_[PRIORITY_MAX];
  size_t finished_job_[PRIORITY_MAX];
  size_t blocked_job_[PRIORITY_MAX];
  size_t dropped_job_[PRIORITY_MAX];
  uint64_t job_id_[PRIORITY_MAX];
  uint64_t finished_job_id_[PRIORITY_MAX];
  int32_t max_queue_len_;
  size_t last_active_time_;
  size_t last_elapse_time_;
  int pid_{0};
};
}  // namespace innovusion

#endif  // UTILS_CONSUMER_PRODUCER_H_
