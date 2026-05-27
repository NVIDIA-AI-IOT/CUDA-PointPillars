/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/mem_pool_manager.h"

#include "utils/mem_allocator.h"

namespace innovusion {
MemPoolManager::MemPoolManager(const char *name, void *buffer, unsigned int unit_size, unsigned int unit_count) {
  name_ = strdup(name);
  pool_ = buffer;
  unit_size_ = unit_size;
  unit_count_ = unit_count;
  free_count_ = unit_count_;
  alloc_call_count_ = 0;
  request_too_big_ = 0;
  return_null_count_ = 0;
  in_use_ = reinterpret_cast<bool *>(malloc(unit_count * sizeof(bool)));
  next_free_ = reinterpret_cast<int *>(malloc(unit_count * sizeof(int)));
  for (unsigned int i = 0; i < unit_count; i++) {
    in_use_[i] = false;
    next_free_[i] = i + 1;
  }
  first_free_ = 0;
  last_free_ = unit_count_ - 1;
  next_free_[last_free_] = -1;
  inno_log_info(
      "MemPoolManager [%s] %p created pool=%p, "
      "unit_size=%u, unit_count=%u, allocator=%s",
      name_, this, pool_, unit_size, unit_count, MemAllocDelegate::get_instance()->get_allocator_name(pool_));
}

MemPoolManager::~MemPoolManager() {
  std::unique_lock<std::mutex> lck(mutex_);
  int f = first_free_;
  for (unsigned int i = 0; i < unit_count_; i++) {
    inno_log_panic_if_not(!in_use_[i], "%s %uth unit still in use. %u/%u", name_, i, free_count_, unit_count_);
    inno_log_panic_if_not(f >= 0 && f < static_cast<int>(unit_count_), "%s invalid free idx %d", name_, f);
    f = next_free_[f];
  }
  inno_log_panic_if_not(f == -1, "%s invalid free idx %d", name_, f);

  inno_log_panic_if_not(free_count_ == unit_count_, "%s not all external buffer are freed. %u %u", name_, free_count_,
                        unit_count_);
  lck.unlock();
  inno_log_info(
      "%s delete MemPoolManager %p pool=%p, "
      "called=%" PRI_SIZEU ", return-null=%" PRI_SIZEU " request_too_big=%" PRI_SIZEU,
      name_, this, pool_, alloc_call_count_, return_null_count_, request_too_big_);
  ::free(in_use_);
  ::free(next_free_);
  ::free(name_);
  in_use_ = NULL;
}

void *MemPoolManager::alloc(unsigned int size) {
  void *ret = NULL;
  std::unique_lock<std::mutex> lck(mutex_);
  alloc_call_count_++;
  if (size > unit_size_) {
    request_too_big_++;
    return_null_count_++;
    inno_log_error("%s external mem pool unit_size too small %u < %u", name_, unit_size_, size);
    lck.unlock();
    return NULL;
  }
  if (free_count_ <= 0) {
    return_null_count_++;
    ret = NULL;
    inno_log_error("no free slot in %s job pool, free_count_=%u", name_, free_count_);
  } else {
    // find the first free
    int f = first_free_;
    inno_log_panic_if_not(f >= 0 && f < static_cast<int>(unit_count_), "%s invalid free idx %d", name_, f);
    inno_log_panic_if_not(!in_use_[f], "%s invalid free idx %d, in_use %d", name_, f, in_use_[f]);

    first_free_ = next_free_[f];
    next_free_[f] = -1;
    if (last_free_ == f) {
      last_free_ = -1;
    }
    in_use_[f] = true;
    ret = reinterpret_cast<char *>(pool_) + unit_size_ * f;
    // memset(ret, 0, size);
    free_count_--;
  }
  return ret;
}

void MemPoolManager::free(void *buffer) {
  std::unique_lock<std::mutex> lck(mutex_);
  inno_log_panic_if_not(buffer >= pool_, "%s invalid pointer %p < %p", name_, buffer, pool_);
  uint64_t offset = reinterpret_cast<char *>(buffer) - reinterpret_cast<char *>(pool_);
  unsigned int idx = offset / unit_size_;
  unsigned int mod = offset % unit_size_;
  inno_log_panic_if_not(((mod == 0) && (idx < unit_count_)),
                        "%s invalid pointer, buffer=%p pool=%p "
                        "size=%u count=%u",
                        name_, buffer, pool_, unit_size_, unit_count_);
  inno_log_panic_if_not(in_use_[idx], "%s double free pointer %p, idx=%u", name_, buffer, idx);
  in_use_[idx] = false;
  inno_log_panic_if_not(free_count_ < unit_count_, "%s invalid free point %p. %u/%u", name_, buffer, free_count_,
                        unit_count_);
  int l = last_free_;
  if (l == -1) {
    inno_log_panic_if_not(free_count_ == 0, "%s bad free_count %u", name_, free_count_);
    inno_log_panic_if_not(first_free_ == -1, "%s bad first_free %d", name_, first_free_);
    last_free_ = idx;
    first_free_ = idx;
  } else {
    inno_log_panic_if_not(free_count_ > 0, "%s bad free_count %u", name_, free_count_);
    inno_log_panic_if_not(first_free_ != -1, "%s bad first_free %d", name_, first_free_);
    inno_log_panic_if_not(l >= 0 && l < static_cast<int>(unit_count_), "%s invalid last free idx %d", name_, l);
    next_free_[l] = idx;
    last_free_ = idx;
  }
  free_count_++;

  return;
}

MemPool::MemPool(const char *name, unsigned int unit_sz, unsigned int unit_nm, uint64_t alignment, bool is_sys_malloc)
    : is_sys_malloc_(is_sys_malloc) {
  inno_log_verify(unit_sz > 0, "%s unit_size = %u", name, unit_sz);
  inno_log_verify(unit_nm > 0, "%s unit_number = %u", name, unit_nm);
  inno_log_verify(alignment > 0, "%s alignment = %" PRI_SIZEU, name, alignment);
  inno_log_verify((alignment & (alignment - 1)) == 0, "%s alignment = %" PRI_SIZEU, name, alignment);

  unit_size_ = ((unit_sz - 1) / alignment + 1) * alignment;
  unit_count_ = unit_nm;
  alignment_ = alignment;

  if (is_sys_malloc_ == true) {
    // allocate one extra for alignment adjustment
    alloc_delegate_ = NULL;
    pool_ = calloc(unit_count_ + 1, unit_size_);
  } else {
    // allocate one extra for alignment adjustment
    alloc_delegate_ = MemAllocDelegate::get_instance();
    inno_log_verify(alloc_delegate_, "allocator_ is null");
    pool_ = alloc_delegate_->calloc(unit_count_ + 1, unit_size_);
  }
  inno_log_verify(pool_, "%s cannot alloc memory %u %u", name, unit_sz, unit_nm);

  name_ = strdup(name);
  inno_log_verify(name_, "%s cannot alloc name", name);
  if (reinterpret_cast<uintptr_t>(pool_) % alignment != 0) {
    aligned_pool_ = reinterpret_cast<void *>((uintptr_t(pool_) + alignment - 1) & ~(alignment - 1));
  } else {
    aligned_pool_ = pool_;
  }
  manager_ = new MemPoolManager(name_, aligned_pool_, unit_size_, unit_count_);
  inno_log_verify(manager_, "%s cannot alloc manager", name);
}

MemPool::~MemPool() {
  delete manager_;
  manager_ = NULL;
  // free
  if (is_sys_malloc_ == true) {
    ::free(pool_);
  } else {
    alloc_delegate_->free(pool_);
  }
  pool_ = NULL;
  ::free(name_);
  name_ = NULL;
}

void *MemPool::alloc() {
  return manager_->alloc(unit_size_);
}

void MemPool::free(void *buffer) {
  manager_->free(buffer);
}

}  // namespace innovusion
