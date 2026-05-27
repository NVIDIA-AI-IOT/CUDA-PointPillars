#include "blocking_ring.h"
#include <functional>
using namespace hesai::lidar;
template <typename T, size_t N>
void BlockingRing<T, N>::push_front(T&& value) {
    LockC lock(_mutex);
    _condv.wait(lock, std::bind(&Super::not_full, this));
    Super::push_front(std::move(value));
    _condv.notify_one();
}

template <typename T, size_t N>
template <typename... Args>
void BlockingRing<T, N>::emplace_back(Args&&... args) {
    LockC lock(_mutex);
    _condv.wait(lock, std::bind(&Super::not_full, this));
    Super::emplace_back(args...);
    _condv.notify_one();
}

template <typename T, size_t N>
void BlockingRing<T, N>::push_back(T&& value) {
    LockC lock(_mutex);
    _condv.wait(lock, std::bind(&Super::not_full, this));
    Super::push_back(std::move(value));
    _condv.notify_one();
}

template <typename T, size_t N>
T BlockingRing<T, N>::pop_front() {
    LockC lock(_mutex);
    _condv.wait(lock, std::bind(&Super::not_empty, this));
    T value = Super::pop_front();
    _condv.notify_one();
    return value;
}

template <typename T, size_t N>
bool BlockingRing<T, N>::try_pop_front(T& value) {
    LockC lock(_mutex);
    using namespace std::literals::chrono_literals;
    bool ret = _condv.wait_for(lock, 100ms, std::bind(&Super::not_empty, this));
    if (ret) {
        value = Super::peek_front();
        Super::eff_pop_front();
    }
    else{
        ;
    }
    _condv.notify_one();
    return ret;
}

template <typename T, size_t N>
T BlockingRing<T, N>::pop_back() {
    LockC lock(_mutex);
    _condv.wait(lock, std::bind(&Super::not_empty, this));
    T value = Super::pop_back();
    _condv.notify_one();
    return value;
}

template <typename T, size_t N>
bool BlockingRing<T, N>::empty()
{
    LockS lock(_mutex);
    return Super::empty();
}

template <typename T, size_t N>
bool BlockingRing<T, N>::not_empty()
{
    LockS lock(_mutex);
    return Super::not_empty();
}

template <typename T, size_t N>
bool  BlockingRing<T, N>::full()
{
    LockS lock(_mutex);
    return Super::full();
}

template <typename T, size_t N>
bool BlockingRing<T, N>::not_full()
{
    LockS lock(_mutex);
    return Super::not_full();
}

template <typename T, size_t N>
void BlockingRing<T, N>::clear()
{
    LockS lock(_mutex);
    Super::clear();
}

template <typename T, size_t N>
void BlockingRing<T, N>::eff_clear()
{
    LockS lock(_mutex);
    Super::eff_clear();
}

template <typename T, size_t N>
void BlockingRing<T, N>::eff_pop_front()
{
    LockS lock(_mutex);
    Super::eff_pop_front();
}