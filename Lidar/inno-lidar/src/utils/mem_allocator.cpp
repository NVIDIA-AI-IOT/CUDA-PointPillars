/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include <fcntl.h>
#if !(defined(__MINGW64__) || defined(_WIN32))
#include <sys/mman.h>
#endif

#include "utils/mem_allocator.h"
#include "utils/inno_lidar_log.h"

namespace innovusion {
#if !(defined(__MINGW64__) || defined(_WIN32))
//======================================
// MemMapMemAllocator
//======================================
std::mutex MemMapMemAllocator::mutex_;

/**
 * When delegate setup direct memory mode,
 * this constructor will be called.
 * Do not panic in this constructor, just set
 * this instance to invalid if some error
 * occurs.
 */
#define MEM_FD_INIT (-1)
MemMapMemAllocator::MemMapMemAllocator(size_t base_addr, size_t high_addr) {
  base_addr_ = base_addr;
  high_addr_ = high_addr;
  map_base_ = MAP_FAILED;
  mem_fd_ = MEM_FD_INIT;
  if (base_addr_ >= high_addr_) {
    inno_log_warning("init with base_addr:%" PRI_SIZELX
                  ", higi_addr:%" PRI_SIZELX,
                  base_addr_,
                  high_addr_);
    return;
  }
  mem_fd_ = open("/dev/mem", O_RDWR | O_NDELAY);
  if (mem_fd_ >= 0) {
    map_base_ = mmap(NULL,
                     high_addr_ - base_addr_ + 1,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED,
                     mem_fd_,
                     base_addr_);
    if (map_base_ != MAP_FAILED) {
      // init range list
      range_list_ = new MemRange(map_base_, high_addr_ - base_addr_ + 1);
      range_list_->next = nullptr;
      range_list_->pre = nullptr;
      range_list_->is_dirty = false;
    } else {
      inno_log_warning("map failed! errno:%d", errno);
    }
  } else {
    inno_log_warning("open /dev/mem failed! errno:%d", errno);
  }
  inno_log_info("init MemMapMemAllocator success");
}

MemMapMemAllocator::~MemMapMemAllocator() {
  // verify all memory is free
  while (range_list_) {
    MemRange *range = range_list_;
    range_list_ = range->next;
    inno_log_verify(range->is_dirty == false, "range %p is dirty", range->offset);
    delete range;
  }
  range_list_ = nullptr;
  if (map_base_ != MAP_FAILED) {
    // invalid cache
#ifdef ARCH_ARM64
    MemMapMemAllocator::invalid_cache_(map_base_, high_addr_ - base_addr_ + 1);
#endif
    munmap(map_base_, high_addr_ - base_addr_ + 1);
    inno_log_info("unmap success, map_base:%p", map_base_);
  }
  if (mem_fd_ >= 0) {
    close(mem_fd_);
    mem_fd_ = MEM_FD_INIT;
  }
}


void *MemMapMemAllocator::calloc(size_t nmemb, size_t size) {
  std::unique_lock<std::mutex> lk(mutex_);
  size_t size_need = nmemb * size;
  // align by page
  size_need += (page_size_ - 1);
  size_need &= ~(page_size_ - 1);
  if (size_need < page_size_) {
    size_need = page_size_;
  }
  // traversal range list to find the first applicable range
  MemRange *range = search_first_applicable_range_(size_need);
  if (!range) {
    inno_log_info("no range is applicable");
    return nullptr;
  }
  inno_log_trace("[MemMapAllocator] will alloc %lu bytes at %p",
                 size_need, range->offset);
  void *ret = alloc_in_free_range_(range, size_need);
  inno_log_trace("[MemMapAllocator] current range list:");
  range = range_list_;
  while (range) {
    inno_log_trace("[MemMapAllocator] <start:%p, end:%p, size:%zx, %s>",
                  range->offset, range->end, range->size,
                  range->is_dirty ? "dirty":"free");
    range = range->next;
  }
  return ret;
}

void MemMapMemAllocator::free(void *buffer) {
  std::unique_lock<std::mutex> lk(mutex_);
  // set a range's is_dirty flag to true.
  MemRange *range = find_target_range_(buffer);
  inno_log_verify(range, "try to free a invalid point %p", buffer);
  range->is_dirty = false;
  // merge ranges if the range before and/or after this range is free.
  MemRange *next_range = range->next;
  if (next_range && !next_range->is_dirty) {
    merge_contiguous_range_(range, next_range);
  }
  MemRange *pre_range = range->pre;
  if (pre_range && !pre_range->is_dirty) {
    merge_contiguous_range_(pre_range, range);
  }
}

bool MemMapMemAllocator::addr_is_valid(void *addr) const {
  return addr >= map_base_ &&
         addr < reinterpret_cast<char *>(map_base_)
                + (high_addr_ - base_addr_ + 1);
}

bool MemMapMemAllocator::is_valid() const {
  return mem_fd_ >= 0 && map_base_ != MAP_FAILED;
}

MemRange *MemMapMemAllocator::search_first_applicable_range_(size_t size_need) {
  MemRange *range = range_list_;
  while (range) {
    if (!range->is_dirty && range->size >= size_need) {
      return range;
    }
    range = range->next;
  }
  return nullptr;
}

void *MemMapMemAllocator::alloc_in_free_range_(MemRange *range,
                                              size_t size_need) {
  inno_log_verify(range &&
                  !range->is_dirty &&
                  size_need <= range->size,
                  "Try to alloc memory in a inapplicable range!");

  if (range->size - size_need < page_size_) {
    range->is_dirty = true;
    memset(range->offset, 0x0, range->size);
    return range->offset;
  }
  auto *alloc_range = new MemRange(range->offset, size_need);
  alloc_range->is_dirty = true;
  alloc_range->next = range;
  alloc_range->pre = range->pre;
  if (alloc_range->pre) {
    alloc_range->pre->next = alloc_range;
  }
  // origin range is the first range, move head list forward
  if (range_list_ == range) {
    range_list_ = alloc_range;
  }
  range->offset = reinterpret_cast<char *>(alloc_range->end) + 1;
  range->size = range->size - size_need;
  range->pre = alloc_range;
  memset(alloc_range->offset, 0x0, alloc_range->size);
  return alloc_range->offset;
}

/**
 * Find the range which start at specific address
 * @param buffer
 * @return
 */
MemRange *MemMapMemAllocator::find_target_range_(void *buffer) {
  MemRange *range = range_list_;
  while (range) {
    if (range->offset == buffer) {
      return range;
    }
    range = range->next;
  }
  return nullptr;
}

/**
 * merge the second range to the first
 * @param first
 * @param second
 */
void MemMapMemAllocator::merge_contiguous_range_(MemRange *first,
                                                 MemRange *second) {
  inno_log_verify(first && second &&
                  !first->is_dirty &&
                  !second->is_dirty &&
                  reinterpret_cast<char *>(first->end) + 1 == second->offset,
                  "Merge range %p with %p is invalid", first, second);
  first->size = first->size + second->size;
  first->end = second->end;
  first->next = second->next;
  if (first->next) {
    first->next->pre = first;
  }
  delete second;
  second = nullptr;
}

#ifdef ARCH_ARM64
void MemMapMemAllocator::invalid_cache_(void *addr, size_t size) {
  // get cache line size from system
  const std::size_t cache_line_size = 64;
  inno_log_verify(((reinterpret_cast<uint64_t>(addr) & cache_line_size) == 0),
                  "size %" PRI_SIZEU
                  " is not aligned to cache line size %" PRI_SIZEU,
                  size, cache_line_size);
  char *ptr = reinterpret_cast<char *>(addr);
  // make sure then end address will not exceed the size
  char *end = ptr + size;
  while (ptr < end) {
    __asm__ volatile("dc civac, %0" ::"r"(ptr) : "memory");
    ptr += cache_line_size;
  }
  __asm__ volatile("dsb sy" ::: "memory");
}
#endif  // ARCH_ARM64

#endif  // !__MINGW64__

//======================================
// DefaultMemAllocator
//======================================
std::mutex DefaultMemAllocator::mutex_;
DefaultMemAllocator::DefaultMemAllocator() {}
void *DefaultMemAllocator::calloc(size_t nmemb, size_t size) {
  return ::calloc(nmemb, size);
}

void DefaultMemAllocator::free(void *buffer) {
  inno_log_verify(buffer, "try to free null ptr!");
  ::free(buffer);
}

//======================================
// MemAllocDelegate
//======================================
MemAllocDelegate *MemAllocDelegate::instance_ = nullptr;
std::mutex MemAllocDelegate::mutex_;
/**
 * [CST Bugfinder Defect ID 53738] Reviewed
 *
 * PolySpace report a defect here:
 * Unnecessary code, if-condition is always true.
 *
 * Avoid multi-threaded competition consumption ,
 *
 * So ignore this defect
 */
MemAllocDelegate *MemAllocDelegate::get_instance() {
  if (instance_ == nullptr) {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      if (instance_ == nullptr) {
        instance_ = new MemAllocDelegate();
        inno_log_verify(instance_, "Create MemAllocDelegate failed!");
        instance_->is_live_direct_memory_mode = false;
      }
    }
  }
  return instance_;
}

MemAllocDelegate::MemAllocDelegate() {
  // TODO(hooya.hu@cn.innovusion.com): refine the choose logic
  // init allocator list
  MemAllocator *default_allocator = nullptr;
  allocator_list_ = nullptr;
#if MEM_ALLOC_DEFAULT
  default_allocator = new DefaultMemAllocator();
#endif
  //  note that sort of the list indicates priorities of the allocators
  if (default_allocator && default_allocator->is_valid()) {
    default_allocator->next = allocator_list_;
    allocator_list_ = default_allocator;
  }
  allocation_cnt_ = 0;
}

void MemAllocDelegate::setup_live_direct_memory_mode(size_t base_addr,
                                                     size_t high_addr) {
#if MEM_ALLOC_MAP
  // add memmap_allocator into allocator list
  std::unique_lock<std::mutex> lk(instance_mutex_);
  if (is_live_direct_memory_mode) {
    inno_log_info("Already in live direct memory mode.");
    return;
  }
  // Calling of this function means lidar is in direct memory mode.
  // Verify there is allocation before delegate setup direct memory mode.
  inno_log_verify(allocation_cnt_ == 0,
                 "Did memory allocation with MemAllocDelegate before the direct"
                 "memory mode setting up in direct memory mode!"
                 "Please checking the usage of MemAllocDelegate");
#if !(defined(__MINGW64__) || defined(_WIN32))
  MemAllocator *memmap_allocator = new MemMapMemAllocator(base_addr, high_addr);
  if (memmap_allocator && memmap_allocator->is_valid()) {
    memmap_allocator->next = allocator_list_;
    allocator_list_ = memmap_allocator;
    is_live_direct_memory_mode = true;
  }
#endif
#endif
}

void MemAllocDelegate::close_live_direct_memory_mode() {
  std::unique_lock<std::mutex> lk(instance_mutex_);
  if (!is_live_direct_memory_mode) {
    inno_log_info("Not in live direct memory mode.");
    return;
  }
  delete_allocator_with_lock_("MemMapMemAllocator");
  is_live_direct_memory_mode = false;
  allocation_cnt_ = 0;
}

void *MemAllocDelegate::calloc(size_t nmemb, size_t size) {
  MemAllocator *allocator;
  {
    std::unique_lock<std::mutex> lk(instance_mutex_);
    allocator = allocator_list_;
    allocation_cnt_++;
  }
  while (allocator) {
    inno_log_info("%s calloc start", allocator->get_name());
    void *alloc_rslt = allocator->calloc(nmemb, size);
    if (alloc_rslt) {
      return alloc_rslt;
    }
    inno_log_info("%s calloc failed", allocator->get_name());
    allocator = allocator->next;
  }
  return nullptr;
}

void MemAllocDelegate::free(void *buffer) {
  MemAllocator *allocator;
  {
    std::unique_lock<std::mutex> lk(instance_mutex_);
    allocator = allocator_list_;
  }
  while (allocator) {
    if (allocator->addr_is_valid(buffer)) {
      allocator->free(buffer);
      return;
    }
    allocator = allocator->next;
  }
  inno_log_assert(false, "free %p failed", buffer);
}

const char *MemAllocDelegate::get_allocator_name(void *buffer) {
  MemAllocator *allocator;
  {
    std::unique_lock<std::mutex> lk(instance_mutex_);
    allocator = allocator_list_;
  }
  while (allocator) {
    if (allocator->addr_is_valid(buffer)) {
      return allocator->get_name();
    }
    allocator = allocator->next;
  }
  return "unknown";
}

void MemAllocDelegate::delete_allocator_with_lock_(const char *name) {
  MemAllocator *allocator;
  allocator = allocator_list_;

  MemAllocator *pre_allocator = nullptr;
  while (allocator) {
    if (strcmp(allocator->get_name(), name) == 0) {
      if (pre_allocator) {
        pre_allocator->next = allocator->next;
      } else {
        allocator_list_ = allocator->next;
      }
      delete allocator;
      return;
    }
    pre_allocator = allocator;
    allocator = allocator->next;
  }
}
}  // namespace innovusion
