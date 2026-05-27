#include "ring.h"
using namespace hesai::lidar;
template <typename T, size_t N>
template <typename Item>
class Ring_SDK<T, N>::ItemIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using reference = T&;
private:
    pointer _p0;
    difference_type _steps;
public:
    ItemIterator() : _p0(nullptr), _steps(0) {}
    ItemIterator(const ItemIterator&) = default;
    ItemIterator& operator=(const ItemIterator&) = default;
    ItemIterator(pointer p0, difference_type steps) : _p0(p0), _steps(steps) {}

    inline ItemIterator& operator++() { ++_steps; return *this; }
    inline ItemIterator& operator--() { assert(_steps >= 1); --_steps; return *this; }
    inline ItemIterator operator++(int) { ItemIterator old(*this); ++(*this); return old; }
    inline ItemIterator operator--(int) { assert(_steps >= 1); ItemIterator old(*this); --(*this); return old; }
    inline ItemIterator operator+(difference_type n) const { return ItemIterator(_p0, _steps + n); }
    inline ItemIterator operator-(difference_type n) const { assert(_steps >= n); return ItemIterator(_p0, _steps - n); }
    inline ItemIterator& operator+=(difference_type n) { _steps += n; return *this; }
    inline ItemIterator& operator-=(difference_type n) { assert(_steps >= n); _steps -= n; return *this; }
    inline reference operator*() const { return _p0[_steps % N]; }
    inline operator pointer () const { return _p0 + _steps % N; }
    // inline pointer operator->() const { return this->operator pointer (); }
    inline pointer p0() const { return _p0; }
    inline difference_type steps() const { return _steps; }
    inline bool operator==(const ItemIterator& other) const { return this->p0() == other.p0() && this->steps() == other.steps(); }
    inline bool operator!=(const ItemIterator& other) const { return !(*this == other); }
    inline difference_type operator-(const ItemIterator& other) const { assert(this->p0() == other.p0()); return this->steps() - other.steps(); }
};

template <typename T, size_t N>
Ring_SDK<T, N>::Ring_SDK() : _ring(new std::array<T, N>), _begin(0), _size(0) {};

template <typename T, size_t N>
inline constexpr size_t Ring_SDK<T, N>::size() const { return _size; }

template <typename T, size_t N>
inline bool Ring_SDK<T, N>::empty() const { return _size == 0; }

template <typename T, size_t N>
inline bool Ring_SDK<T, N>::not_empty() const { return !empty(); }

template <typename T, size_t N>
inline bool Ring_SDK<T, N>::full() const { return _size >= N; }

template <typename T, size_t N>
inline bool Ring_SDK<T, N>::not_full() const { return !full(); }

template <typename T, size_t N>
inline void Ring_SDK<T, N>::clear() { std::generate(_ring->begin(), _ring->end(), []() { return T(); }); _size = 0; _begin = 0; }

template <typename T, size_t N>
inline void Ring_SDK<T, N>::eff_clear() { _size = 0; _begin = 0; }

template <typename T, size_t N>
template <typename... Args>
inline void Ring_SDK<T, N>::emplace_back(Args&&... args) { assert(!full()); (*_ring)[(_begin + _size) % N] = T(args...); ++_size;}

template <typename T, size_t N>
inline void Ring_SDK<T, N>::push_back(T&& item) { assert(!full()); (*_ring)[(_begin + _size) % N] = item; ++_size; }

template <typename T, size_t N>
inline const T& Ring_SDK<T, N>::peek_back() const { assert(!empty()); return (*_ring)[(_begin + _size - 1) % N]; }

template <typename T, size_t N>
inline T Ring_SDK<T, N>::pop_back() { assert(!empty()); --_size; T item; std::swap(item, (*_ring)[(_begin + _size) % N]); return item; }

template <typename T, size_t N>
inline void Ring_SDK<T, N>::eff_pop_back() { --_size; }

template <typename T, size_t N>
template <typename... Args>
inline void Ring_SDK<T, N>::emplace_front(Args&&... args) { assert(!full()); _begin = (_begin + N - 1) % N; (*_ring)[_begin] = T(args...); ++_size; }

template <typename T, size_t N>
inline void Ring_SDK<T, N>::push_front(T&& item) { assert(!full()); _begin = (_begin + N - 1) % N; (*_ring)[_begin] = item; ++_size; }

template <typename T, size_t N>
inline const T& Ring_SDK<T, N>::peek_front() const { assert(!empty()); return (*_ring)[_begin]; }

template <typename T, size_t N>
inline T Ring_SDK<T, N>::pop_front() { assert(!empty()); --_size; T item; std::swap(item, (*_ring)[_begin]); _begin = (_begin + 1) % N; return item; }

template <typename T, size_t N>
inline void Ring_SDK<T, N>::eff_pop_front() { --_size; _begin = (_begin + 1) % N; }

template <typename T, size_t N>
inline typename Ring_SDK<T, N>::iterator Ring_SDK<T, N>::begin() { return iterator((T*)_ring.get(), _begin); }

template <typename T, size_t N>
inline typename Ring_SDK<T, N>::iterator Ring_SDK<T, N>::end() { return iterator((T*)_ring.get(), _begin + _size); }

template <typename T, size_t N>
inline typename Ring_SDK<T, N>::const_iterator Ring_SDK<T, N>::cbegin() const { return const_iterator((const T*)_ring.get(), _begin); }

template <typename T, size_t N>
inline typename Ring_SDK<T, N>::const_iterator Ring_SDK<T, N>::cend() const { return const_iterator((const T*)_ring.get(), _begin + _size); }

template <typename T, size_t N>
inline T& Ring_SDK<T, N>::operator[](size_t index) { assert(index < size()); return (*_ring)[(_begin + index) % N]; }

template <typename T, size_t N>
inline const T& Ring_SDK<T, N>::operator[](size_t index) const { assert(index < size()); return (*_ring)[(_begin + index) % N]; }

template <typename T, size_t N>
inline T* Ring_SDK<T, N>::data() { return _ring->data(); }

template <typename T, size_t N>
inline const T* Ring_SDK<T, N>::data() const { return _ring->data(); };

