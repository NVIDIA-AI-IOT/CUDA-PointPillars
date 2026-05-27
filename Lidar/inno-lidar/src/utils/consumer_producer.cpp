/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/consumer_producer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#if !(defined(_QNX_) || defined(__MINGW64__) || defined(_WIN32))
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#include <chrono>
#include <string>

#include "utils/log.h"
#include "utils/utils.h"

namespace innovusion {
ConsumerProducer::CpQueue::CpQueue(const char *name,
                                   int queue_size,
                                   bool allow_log) {
  name_ = strdup(name);
  done_ = 0;
  empty_ = true;
  full_ = false;
  count_ = 0;
  head_ = 0;
  tail_ = 0;
  queue_size_ = queue_size;
  jobs_ = new Job[queue_size];
  allow_log_ = allow_log;
  if (do_log_()) {
    inno_log_verify(name_, "name %s", name);
    inno_log_verify(jobs_, "%s jobs", name_);
  }
}

ConsumerProducer::CpQueue::~CpQueue() {
  if (do_log_()) {
    inno_log_verify(count_ == 0, "%s count = %d", name_, count_);
  }
  if (name_) {
    free(name_);
    name_ = NULL;
  }
  delete [] jobs_;
}

void ConsumerProducer::CpQueue::add(const Job &job) {
  jobs_[tail_] = job;

  tail_++;
  if (tail_ == queue_size_) {
    tail_ = 0;
  }
  if (tail_ == head_) {
    full_ = true;
  }
  empty_ = false;
  count_++;
  if (do_log_()) {
    inno_log_verify(count_ <= queue_size_,
                    "%s count = %d queue_size_ = %d",
                    name_, count_, queue_size_);
  }

  return;
}

int ConsumerProducer::CpQueue::pop(Job *job,
                                   bool peek_only) {
  int ret  = count_;
  if (empty_) {
    if (do_log_()) {
      inno_log_verify(count_ == 0,
                      "%s count = %d",
                      name_, count_);
    }
    return 0;
  } else {
    if (do_log_()) {
      inno_log_verify(count_ > 0,
                      "%s count = %d",
                      name_, count_);
    }
  }
  *job = jobs_[head_];
  if (peek_only) {
    return ret;
  }

  head_++;
  if (head_ == queue_size_) {
    head_ = 0;
  }
  count_--;
  if (head_ == tail_) {
    empty_ = true;
    if (do_log_()) {
      inno_log_verify(count_ == 0,
                      "%s count = %d",
                      name_, count_);
    }
  }
  full_ = false;

  return ret;
}

ConsumerProducer::ConsumerProducer(const char *name,
                                   int priority,
                                   int worker_num, ConsumeFunc consume_func,
                                   void *consume_context,
                                   int queue_size, int prefer_queue_size,
                                   int hi_cp_queue_size,
                                   size_t cpusetsize,
                                   const cpu_set_t *cpuset,
                                   bool allow_log)
    : priority_(priority)
    , queue_(name, queue_size, allow_log)
    , prefer_queue_size_(prefer_queue_size)
    , hi_queue_(name, hi_cp_queue_size, allow_log)
    , worker_num_(worker_num)
    , consume_func_(consume_func)
    , consume_context_(consume_context)
    , started_(0)
    , shutdown_(true)
    , cpusetsize_(cpusetsize)
    , cpuset_(cpuset)
    , allow_log_(allow_log) {
  name_ = strdup(name);
  if (do_log_()) {
    inno_log_verify(name_, "%s name", name);
  }
  if (prefer_queue_size_ <= 0) {
    prefer_queue_size_ = queue_size;
    blocking_ = true;
  } else {
    blocking_ = false;
  }
  paused_ = false;
  for (int i = 0; i < PRIORITY_MAX; i++) {
    total_wait_time_[i] = 0;
    total_process_time_[i] = 0;
    total_drop_time_[i] = 0;
    added_job_[i] = 0;
    finished_job_[i] = 0;
    blocked_job_[i] = 0;
    dropped_job_[i] = 0;
    job_id_[i] = 0;
    finished_job_id_[i] = 0;
  }
  max_queue_len_ = 0;
  start_time_ = InnoUtils::get_time_ns();
  last_active_time_ = 0;
  last_elapse_time_ = 0;
}

ConsumerProducer::~ConsumerProducer() {
  if (do_log_()) {
    inno_log_verify(shutdown_.load() && !paused_.load(),
                    "%s shutdown=%d paused=%d started=%d",
                    name_, shutdown_.load(), paused_.load(), started_.load());
  }
  free(name_);
  name_ = NULL;
}

void *ConsumerProducer::consumer_thread_func_(void *context) {
  ConsumerProducer *cp = reinterpret_cast<ConsumerProducer *>(context);
#if !(defined(_QNX_) || defined(__MINGW64__) || defined(__APPLE__) || defined(_WIN32))
  pid_t pid = 0;
  if (cp->cpuset_ && cp->cpusetsize_) {
    const pthread_t tid = pthread_self();
    pthread_setaffinity_np(tid, cp->cpusetsize_,
                           cp->cpuset_);
  }
  pid = syscall(SYS_gettid);
#else
  (void)cp->cpusetsize_;
  (void)cp->cpuset_;
  uint32_t pid = 0;
#endif
  inno_log_info("thread %s starts. pid=%d target_priority=%d",
                cp->name_, pid, cp->priority_);
  cp->pid_ = pid;
  InnoUtils::set_self_thread_priority(cp->priority_);

  while (1) {
    enum Priority priority;
    Job job;
    int r = cp->get_job_(&job, &priority);
    auto timestamp_poped = InnoUtils::get_time_ns();
    bool is_prefered;
    int idx = 0;
    if (r == 0) {
      break;
    }
    if (priority == PRIORITY_LOW) {
      is_prefered = (r <= cp->prefer_queue_size_);
      idx = 0;
    } else {
      // high priority queue
      is_prefered = true;
      idx = 1;
    }
    cp->consume_func_(job.job, cp->consume_context_, is_prefered);

    auto timestamp_done = InnoUtils::get_time_ns();
    std::unique_lock<std::mutex> lck(cp->mutex_);
    cp->total_wait_time_[priority] += (timestamp_poped - job.timestamp);
    if (is_prefered) {
      cp->finished_job_[priority]++;
      cp->total_process_time_[priority] += (timestamp_done - timestamp_poped);
    } else {
      cp->dropped_job_[priority]++;
      cp->total_drop_time_[priority] += (timestamp_done - timestamp_poped);
    }
    cp->update_done_job_id_(idx, job.job_id);
    lck.unlock();
    cp->job_done_cond_.notify_all();
    /*
    inno_log_debug("%s %p out enq=%.4f deq=%.4f done=%.4f", cp->name_,
                   job.job,
                   job.timestamp/1000000000.0,
                   timestamp_poped/1000000000.0,
                   timestamp_done/1000000000.0);
    */
    // yield cpu that firmware can switch mode in time
    // "Use of sched_yield()
    //  with nondeterministic scheduling policies such as SCHED_OTHER is
    //  unspecified and very likely means your application design is
    //  broken."
    if (cp->priority_ > 0) {
      std::this_thread::yield();
    }
  }
  return NULL;
}

void ConsumerProducer::start() {
  started_++;
  shutdown_ = false;
  // creates threads
  for (int i = 0; i < worker_num_; i++) {
    threads_.emplace_back(consumer_thread_func_, this);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void ConsumerProducer::shutdown() {
  print_stats();
  if (do_log_()) {
    inno_log_verify(!shutdown_.load(), "%s shutdown=%d started=%d",
                    name_, shutdown_.load(), started_.load());
  }
  {
    std::unique_lock<std::mutex> lck(mutex_);
    shutdown_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }
  // join threads
  for (std::thread& t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  threads_.clear();
}

uint64_t ConsumerProducer::added_job_total_() {
  uint64_t ret = 0;
  for (int i = 0; i < PRIORITY_MAX; i++) {
    ret += added_job_[i];
  }
  return ret;
}

uint64_t ConsumerProducer::finished_job_total_() {
  uint64_t ret = 0;
  for (int i = 0; i < PRIORITY_MAX; i++) {
    ret += finished_job_[i];
  }
  return ret;
}

uint64_t ConsumerProducer::dropped_job_total_() {
  uint64_t ret = 0;
  for (int i = 0; i < PRIORITY_MAX; i++) {
    ret += dropped_job_[i];
  }
  return ret;
}

void ConsumerProducer::flush_and_pause() {
  std::unique_lock<std::mutex> lck(mutex_);
  if (do_log_()) {
    inno_log_verify(!paused_, "paused_");
  }
  paused_ = true;
  while (added_job_total_() !=
         finished_job_total_() +
         dropped_job_total_()) {
    if (do_log_()) {
      inno_log_verify(added_job_total_() >
                      finished_job_total_() +
                      dropped_job_total_(),
                      "%" PRI_SIZEU " should be bigger %"
                      PRI_SIZEU " + %" PRI_SIZEU,
                      added_job_total_(),
                      finished_job_total_(),
                      dropped_job_total_());
    }
    job_done_cond_.wait(lck);
  }
  // return with mutex locked
  return;
}

void ConsumerProducer::resume() {
  if (do_log_()) {
    inno_log_verify(paused_, "paused_");
  }
  paused_ = false;
  // wake up producer
  not_full_.notify_all();
}

int ConsumerProducer::add_job_wait_done(void *in,
                                        bool high_priority) {
  uint64_t job_id;
  if (do_log_()) {
    inno_log_verify(worker_num_ == 1,
                    "add_job_until_done only support 1 worker");
  }
  int r = add_job_do_(in, high_priority, &job_id, true);
  std::unique_lock<std::mutex> lck(mutex_);
  int idx = high_priority ? 1 : 0;
  while (job_id > finished_job_id_[idx]) {
    job_done_cond_.wait(lck);
  }
  return r;
}
// 0: normal enqueue 1: block enqueue 2: give up enqueue
// 3: discard head and enqueue
int ConsumerProducer::add_job_do_(void *job, bool high_priority,
                                  uint64_t *job_id_out,
                                  bool not_discardable) {
  bool done = false;
  int ret = 0;
  CpQueue &queue = high_priority ? hi_queue_ : queue_;
  int idx;
  bool blocking;
  size_t timestamp_in = InnoUtils::get_time_ns();
  uint64_t job_id = 0;
  /*
  if (do_log_()) {
    inno_log_debug("%s %p in enq=%.4f", name_,
      job, timestamp_in/1000000000.0);
  }
  */
  if (high_priority) {
    idx = 1;
    blocking = true;
  } else {
    idx = 0;
    blocking = blocking_ || not_discardable;
  }
  while (!done) {
    std::unique_lock<std::mutex> lck(mutex_);
    if (paused_) {
      not_full_.wait(lck);
      lck.unlock();
      continue;
    }
    // cannot be shutdown while any producer is still alive
    if (do_log_()) {
      inno_log_verify(!shutdown_.load(), "%s shutdown=%d started=%d",
                      name_, shutdown_.load(), started_.load());
    }
    if (queue.is_full()) {
      if (blocking) {
        ret = 1;
        not_full_.wait(lck);
        lck.unlock();
        continue;
      } else {
        Job discard_job;
        int r = queue.peek(&discard_job);
        if (discard_job.not_discardable) {
          // cannot drop the one in the queue
          // have to discard itself
          added_job_[idx]++;
          dropped_job_[idx]++;
          lck.unlock();
          consume_func_(job, consume_context_, false);
          // don't need to signal job_done_cond_
          // give up enqueue
          ret = 2;
          done = true;
        } else {
          r = queue.pop(&discard_job);
          dropped_job_[idx]++;
          if (do_log_()) {
            inno_log_verify(r,
                            "%s r = %d",
                            name_, r);
          }
          update_done_job_id_(idx, discard_job.job_id);
          lck.unlock();
          consume_func_(discard_job.job, consume_context_, false);
          job_done_cond_.notify_all();
          // discard the head and enqueue
          ret = 3;
        }
      }
    } else {
      job_id = assign_job_id_(idx);
      Job j(job, timestamp_in, job_id, not_discardable);
      queue.add(j);
      int l = queue_.queue_length();
      if (max_queue_len_ < l) {
        max_queue_len_ = l;
      }
      added_job_[idx]++;
      lck.unlock();
      not_empty_.notify_all();
      done = true;
    }
  }
  // only update the block case
  if (ret == 1) {
    blocked_job_[idx] += ret;
  }

  if (job_id_out) {
    *job_id_out = job_id;
  }
  return ret;
}

int ConsumerProducer::get_job_(Job *job,
                               enum Priority *priority) {
  for (;;) {
    std::unique_lock<std::mutex> lck(mutex_);
    if (queue_.is_empty() && hi_queue_.is_empty() && !shutdown_) {
      not_empty_.wait(lck);
    } else {
    }
    int ret = 0;
    ret = hi_queue_.pop(job);
    if (ret == 0) {
      *priority = PRIORITY_LOW;
      ret = queue_.pop(job);
    } else {
      *priority = PRIORITY_HIGH;
    }
    if (ret != 0) {
      lck.unlock();
      not_full_.notify_all();
      return ret;
    } else {
      if (shutdown_) {
        lck.unlock();
        not_empty_.notify_all();
        return 0;
      } else {
        lck.unlock();
      }
    }
    // fprintf(stderr, "again %d\n", a++);
  }
}

void ConsumerProducer::print_stats(void) {
  char buf[1024];
  //get_stats_string(buf, sizeof(buf));
  //inno_log_info("%s", buf);
  return;
}

void ConsumerProducer::get_stats_string(char *buf, size_t buf_size) {
  // xxx todo: this is not protected by mutex, so the number may off a little
  size_t active_time = 0;
  size_t len = 0;
  buf[0] = 0;
  for (int i = 0; i < PRIORITY_MAX; i++) {
    if (added_job_[i] == 0) {
      continue;
    }
    int ret = snprintf(buf + len, buf_size - len,
                       "%s queue#%d added=%" PRI_SIZET
                       " finished=%" PRI_SIZET
                       " dropped=%" PRI_SIZET
                       " blocked=%" PRI_SIZET
                       " wait=%" PRI_SIZET
                       "us process=%" PRI_SIZET
                       "us drop=%" PRI_SIZET "us "
                       "pid=%d ",
                       name_, i,
                       added_job_[i], finished_job_[i], dropped_job_[i],
                       blocked_job_[i],
                       (finished_job_[i] + dropped_job_[i]) > 0 ?
                       total_wait_time_[i] /
                       (finished_job_[i]+ dropped_job_[i]) /
                       1000 :
                       0,
                       finished_job_[i] > 0 ?
                       total_process_time_[i] / finished_job_[i] / 1000 : 0,
                       total_drop_time_[i] > 0 ?
                       total_drop_time_[i] / dropped_job_[i] / 1000 : 0,
                       pid_);
    if (ret >= ssize_t(buf_size - len)) {
      buf[buf_size - 1] = 0;
      return;
    }
    len += ret;
    active_time += total_process_time_[i] + total_drop_time_[i];
  }

  size_t end_time = InnoUtils::get_time_ns();
  size_t elapse_time = end_time - start_time_;

  if (elapse_time > 0 && active_time > 0) {
    size_t elapse_delta = elapse_time - last_elapse_time_;
    size_t active_delta = active_time - last_active_time_;
    double ratio = 0;
    if (elapse_delta > 0 && active_delta > 0) {
      ratio = active_delta * 100.0 / elapse_delta;
    }
    int ret = snprintf(buf + len, buf_size - len,
                       "elapsed_time=%" PRI_SIZET "/%" PRI_SIZET
                       "ms active_time=%" PRI_SIZET
                       "/%" PRI_SIZET "ms "
                       "ratio=%.2f%%/%.2f%%",
                       elapse_delta / 1000000,
                       (end_time - start_time_) / 1000000,
                       active_delta / 1000000,
                       active_time / 1000000,
                       ratio,
                       active_time * 100.0 / (end_time - start_time_));
    if (ret >= ssize_t(buf_size - len)) {
      buf[buf_size - 1] = 0;
      return;
    }
    if (elapse_delta > 0 && active_delta > 0) {
      last_elapse_time_ = elapse_time;
      last_active_time_ = active_time;
    }
  }
  return;
}

}  // namespace innovusion
