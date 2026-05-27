/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 *
 * Manager usage of vm mapped to /dev/mem 0x1000000~0x1dfffff
 * When create a MemPool, it's pool_ should be alloc use this class.
 * It will first alloc firstly try to alloc from memory map to /dev/mem.
 * If there is not enough memory in /dev/mem 0x1000000~0x1dfffff, it
 * will try to call calloc() func of stdlib to alloc memory.
 */

#ifndef UTILS_MEM_ALLOCATOR_H_
#define UTILS_MEM_ALLOCATOR_H_

#ifndef _WIN32
#include <unistd.h>
#endif

#include <mutex>

#include "utils/log.h"

#define MEM_ALLOC_DEFAULT 1
#define MEM_ALLOC_MAP 1
#ifndef ARCH_ARM64
#undef MEM_ALLOC_MAP
#define MEM_ALLOC_MAP 0
#endif

namespace innovusion {
class MemAllocDelegate;
class MemRange;

//=====================================================================
// MemAllocator
// Define interfaces.
//=====================================================================

/**
 * @brief MemAllocator
 */
class MemAllocator {
  friend MemAllocDelegate;

 public:
  virtual ~MemAllocator() = default;

 protected:
  /**
   * @brief Calloc memory
   * @param nmemb Number of elements
   * @param size  Size of each element
   * @return Return the buffer allocated, or NULL for error
   */
  virtual void *calloc(size_t nmemb, size_t size) = 0;
  /**
   * @brief Free memory to the allocator
   * @param buffer Buffer to be freed
   */
  virtual void free(void *buffer) = 0;
  /**
   * @brief Check the address is calloced by this allocator
   * @param addr Address to be checked
   * @return Return true if the address is calloced by this allocator
   *         Return false if the address is not calloced by this allocator
   */
  virtual bool addr_is_valid(void *addr) const = 0;
  /**
   * @brief Check the allocator is valid
   * @return Return true if the allocator is valid
   *         Return false if the allocator is not valid
   */
  virtual bool is_valid() const = 0;
  /**
   * @brief Get the name of the allocator
   * @return Return the name of the allocator
   */
  virtual const char *get_name() const = 0;

 public:
  MemAllocator *next{};
};

//=====================================================================
// MemMapMemAllocator
// Use MemMap to map /dev/mem 0x1000000-0x1dffefff into virtual memory.
// Use a linked table to store ranges in this mapping space. A range
// may be free or in-use.
//=====================================================================

/**
 * @brief MemMapMemAllocator
 */
class MemMapMemAllocator final: MemAllocator {
  friend MemAllocDelegate;

 protected:
  /**
   * @brief Calloc memory, which is aligned to page size
   * @param nmemb Number of elements
   * @param size  Size of each element
   * @return Return the buffer allocated, or NULL for error
   */
  void *calloc(size_t nmemb, size_t size) override;
  /**
   * @brief Free memory to the allocator
   * @param buffer Buffer to be freed
   */
  void free(void *buffer) override;
  /**
   * @brief Check the address is calloced by this allocator
   * @param addr Address to be checked
   * @return Return true if the address is calloced by this allocator
   *         Return false if the address is not calloced by this allocator
   */
  bool addr_is_valid(void *addr) const override;
  /**
   * @brief Check the allocator is valid
   * @return Return true if the allocator is valid, or false for error
   */
  bool is_valid() const override;
  /**
   * @brief Get the name of the allocator
   * @return Return the name of the allocator
   */
  const char *get_name() const override {
    return "MemMapMemAllocator";
  }
  MemMapMemAllocator();
  /**
   * @brief MemMapMemAllocator constructor
   *        It will map /dev/mem into virtual memory
   * @param base_ddr  Base address of the memory map
   * @param high_addr High address of the memory map
   */
  MemMapMemAllocator(size_t base_ddr, size_t high_addr);
  ~MemMapMemAllocator();

 private:
#ifdef ARCH_ARM64
  static void invalid_cache_(void *addr, size_t size);
#endif
  /**
   * @brief Get the first valid range, which size is larger than size_need
   * @param size_need Size of the range needed
   * @return Return the first valid range, or NULL for error
   */
  MemRange *search_first_applicable_range_(size_t size_need);
  /**
   * @brief Alloc size_need memory in the given range or a new range.
   *        If the given range size is one page larger than size_need,
   *        it will split the range into two ranges, and insert the new
   *        range into the range list. The first range is size_need,
   *        and the second range is the rest.
   * @param range     Range to alloc memory
   * @param size_need Size of the memory needed
   * @return Return the buffer allocated, or NULL for error
   */
  void *alloc_in_free_range_(MemRange *range, size_t size_need);
  /**
   * @brief Get the range which the buffer belongs to
   * @param buffer Buffer to be checked
   * @return Return the range which the buffer belongs to, or NULL for error
   */
  MemRange *find_target_range_(void *buffer);
  /**
   * @brief Merge the second range into the first range, and free the second range
   * @param first  First range
   * @param second Second range
   */
  void merge_contiguous_range_(MemRange *first, MemRange *second);

 private:
  MemRange *range_list_;
  static std::mutex mutex_;
  int mem_fd_;
  void *map_base_;
#if !(defined(__MINGW64__) || defined(_WIN32))
  size_t page_size_ = sysconf(_SC_PAGESIZE);
#else
  size_t page_size_ = 4 * 1024;
#endif
  size_t base_addr_;
  size_t high_addr_;
};

/**
 * @brief MemRange
 */
class MemRange {
  friend MemMapMemAllocator;
 protected:
  /**
   * @brief MemRange constructor
   * @param offset Offset of the range
   * @param size   Size of the range
   */
  MemRange(void *offset, size_t size):offset(offset), size(size) {
    end = reinterpret_cast<void *>(reinterpret_cast<char *>(offset) + size - 1);
    is_dirty = false;
    next = nullptr;
    pre = nullptr;
  }
  ~MemRange() = default;

 protected:
  void *offset;
  void *end;
  size_t size;
  bool is_dirty;
  MemRange *next;
  MemRange *pre;
};


//=====================================================================
// DefaultMemAllocator
// use std::alloc() and std::free()
//=====================================================================
/**
 * @brief DefaultMemAllocator
 */
class DefaultMemAllocator final: MemAllocator {
  friend MemAllocDelegate;

 protected:
  DefaultMemAllocator();
  /**
   * @brief Alloc memory by std::alloc()
   * @param nmemb  Number of elements
   * @param size   Size of each element
   * @return Return the buffer allocated, or NULL for error
   */
  void *calloc(size_t nmemb, size_t size) override;
  /**
   * @brief Free memory by std::free()
   * @param buffer Buffer to be freed
   */
  void free(void *buffer) override;
  /**
   * @brief Return true always
   * @param addr Address to be checked
   * @return Return true always
   */
  bool addr_is_valid(void *addr) const override {
    return true;
  };
  /**
   * @brief Return true always
   * @return Return true always
   */
  bool is_valid() const override {
    return true;
  }
  /**
   * @brief Get the name of the allocator
   * @return Return the name of the allocator
   */
  const char *get_name() const override {
    return "DefaultMemAllocator";
  }

 private:
  static std::mutex mutex_;
};

/**
 * @brief MemAllocDelegate
 */
class MemAllocDelegate {
 public:
  /**
   * @brief Alloc memory by the allocator list when the allocator is valid
   * @param nmemb Number of elements
   * @param size  Size of each element
   * @return Return the buffer allocated, or NULL for error
   */
  void *calloc(size_t nmemb, size_t size);
  /**
   * @brief Free buffer by the allocator list
   * @param buffer Buffer to be freed
   */
  void free(void *buffer);
  /**
   * @brief Delegate instance of MemAllocDelegate
   * @return Return the instance of MemAllocDelegate
   */
  static MemAllocDelegate *get_instance();
  /**
   * @brief Get which allocator be used to allocated the buffer
   * @return Return allocator's name
   */
  const char *get_allocator_name(void *buffer);
  /**
   * @brief New a MemMapMemAllocator and add it into the allocator list
   * @param base_addr Base address of the memory map
   * @param high_addr High address of the memory map
   */
  void setup_live_direct_memory_mode(size_t base_addr, size_t high_addr);
  void close_live_direct_memory_mode();

 private:
  /**
   * @brief Setup the allocator list
   * If MEM_ALLOC_DEFAULT is set, it will add DefaultMemAllocator into the list
   */
  MemAllocDelegate();
  ~MemAllocDelegate() {
    while (allocator_list_) {
      MemAllocator *head = allocator_list_;
      allocator_list_ = head->next;
      delete head;
    }
    allocator_list_ = nullptr;
  }
  void delete_allocator_with_lock_(const char *name);

 private:
  static MemAllocDelegate *instance_;
  static std::mutex mutex_;
  MemAllocator *allocator_list_;
  std::mutex instance_mutex_;
  bool is_live_direct_memory_mode;
  uint32_t allocation_cnt_;
};
}  // namespace innovusion
#endif  // UTILS_MEM_ALLOCATOR_H_
