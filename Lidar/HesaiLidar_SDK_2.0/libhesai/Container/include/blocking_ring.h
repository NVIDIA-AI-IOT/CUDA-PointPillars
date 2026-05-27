#ifndef ___HESAI__CONTAINER__BLOCKING_RING_HH___
#define ___HESAI__CONTAINER__BLOCKING_RING_HH___

#include "ring.h"
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <functional>
namespace hesai
{
namespace lidar
{
template <typename T, size_t N>
class BlockingRing : public Ring_SDK<T, N>{
public:
    using Super = Ring_SDK<T, N>;
    using Mutex = std::mutex;
    using Condv = std::condition_variable;
    using LockS = std::lock_guard<Mutex>;           // lock in scope
    using LockC = std::unique_lock<Mutex>;          // lock by condition
private:
    Mutex _mutex;
    Condv _condv;
public:
    template <typename... Args>
    void emplace_back(Args&&... args);
    void push_back(T&& item);
    T pop_back();
    void push_front(T&& item);
    T pop_front();
    bool try_pop_front(T&);
    bool empty();
    bool not_empty();
    bool full();
    bool not_full();
    void clear();
    void eff_clear();
    void eff_pop_front();
};
}  // namespace lidar
}  // namespace hesai

#include "blocking_ring.cc"

#endif // !___HESAI__CONTAINER__BLOCKING_RING_HH___
