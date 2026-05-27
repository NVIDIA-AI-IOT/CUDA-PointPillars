/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_MEM_POOL_MANAGER_H_
#define UTILS_MEM_POOL_MANAGER_H_

#include <limits.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <stdint.h>
#include <sys/types.h>

#include <cstddef>
#include <map>
#include <string>

#include "utils/log.h"
#include "utils/mem_allocator.h"

namespace innovusion {
/**
 * @brief MemPoolManager
 */
class MemPoolManager {
 public:
  /**
   * @brief MemPoolManager constructor
   * @param name        Name of the pool
   * @param buffer      A large buffer to be used as the pool
   * @param unit_size   Size of each unit in the pool
   * @param unit_number Number of units in the pool
   */
  MemPoolManager(const char *name, void *buffer, unsigned int unit_size, unsigned int unit_number);
  ~MemPoolManager();
  /**
   * @brief Allocate a buffer from the pool.
   *        The buffer size should be less than or equal to the unit size
   * @param size Size of the buffer to be allocated
   * @return Return the buffer allocated from the pool, or NULL for error
   */
  void *alloc(unsigned int size);
  /**
   * @brief Free buffer to the pool
   * @param buffer Buffer to be freed
   */
  void free(void *buffer);
  /**
   * @brief Get the entire pool buffer
   * @return Return the pool buffer
   */
  const void *get_pool() const {
    return reinterpret_cast<const void *>(pool_);
  }
  /**
   * @brief Check a buffer is allocated from the pool
   * @param b Buffer to be checked
   * @return Return true if the buffer is allocated from the pool, otherwise return false
   */
  bool is_manager_of(void *b) const {
    return b >= pool_ && b < reinterpret_cast<char *>(pool_) + unit_size_ * unit_count_;
  }

 private:
  char *name_;
  void *pool_;
  unsigned int unit_size_;
  unsigned int unit_count_;
  unsigned int free_count_;
  uint64_t alloc_call_count_;
  uint64_t return_null_count_;
  uint64_t request_too_big_;
  std::mutex mutex_;
  bool *in_use_;
  int *next_free_;
  int first_free_;
  int last_free_;
};

/**
 * @brief MemPool
 */
class MemPool {
 public:
  /**
   * @brief MemPool constructor
   * @param name          Name of the pool
   * @param unit_sz       Size of each unit in the pool
   * @param unit_nm       Number of units in the pool
   * @param alignment     Alignment of each unit in the pool
   * @param is_sys_malloc Is the pool created by sys malloc or a dedicated buffer
   */
  MemPool(const char *name, unsigned int unit_sz, unsigned int unit_nm, uint64_t alignment, bool is_sys_malloc = false);

  ~MemPool();
  /**
   * @brief Allocate a buffer with unit size from the pool.
   * @return Return the buffer allocated from the pool, or NULL for error
   */
  void *alloc();
  /**
   * @brief Free buffer to the pool
   * @param buffer Buffer to be freed
   */
  void free(void *buffer);

  /**
   * @brief Check if the address is in the pool
   * @param  address_p  :   Address Pointer
   * @return true  : Is pool pointer
   * @return false; : Not pool pointer
   */
  bool is_manager_of(void *address_p) const {
    return manager_->is_manager_of(address_p);
  }

 private:
  char *name_;
  void *pool_;
  void *aligned_pool_;
  unsigned int unit_size_;
  unsigned int unit_count_;
  uint64_t alignment_;
  MemPoolManager *manager_;
  MemAllocDelegate *alloc_delegate_;
  // support the sys malloc
  bool is_sys_malloc_ = false;
};

template <class T>
class ObjectPool {
 public:
  /**
   * @brief ObjectPool constructor
   *        Each unit in the pool is sizeof(T) bytes
   * @param name        Name of the pool
   * @param unit_number Number of units in the pool
   * @param alignment   Alignment of each unit in the pool
   */
  ObjectPool(const char *name, unsigned int unit_number, uint64_t alignment)
      : pool_(name, sizeof(T), unit_number, alignment) {
  }

  ~ObjectPool() {
  }

  /**
   * @brief Allocate an object from the pool, and call the constructor of the object
   * @return Return the object allocated from the pool, or NULL for error
   */
  virtual T *alloc() {
    return new (pool_.alloc()) T;
  }

  /**
   * @brief Free an object buffer to the pool, and call the destructor of the object
   * @param o Object buffer to be freed
   */
  void free(T *o) {
    return pool_.free(o);
  }

 private:
  MemPool pool_;
};

}  // namespace innovusion
#endif  // UTILS_MEM_POOL_MANAGER_H_
