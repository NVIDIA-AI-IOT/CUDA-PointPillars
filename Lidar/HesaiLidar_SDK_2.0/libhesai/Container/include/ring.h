#ifndef ___HESAI__CONTAINER__RING_SDK_HH___
#define ___HESAI__CONTAINER__RING_SDK_HH___

#include <memory>
#include <array>
#include <cassert>
#include <iterator>
#include <algorithm>
namespace hesai
{
namespace lidar
{
template <typename T, size_t N>
class Ring_SDK {
public:
    static constexpr size_t Size = N;
    template <typename Item> class ItemIterator;
    using iterator = ItemIterator<T>;
    using const_iterator = ItemIterator<const T>;
private:
    std::unique_ptr<std::array<T, N>> _ring;
    size_t _begin, _size;
public:
    Ring_SDK();
    inline constexpr size_t size() const;
    inline bool empty() const;
    inline bool not_empty() const;
    inline bool full() const;
    inline bool not_full() const;
    inline void clear();
    inline void eff_clear();
    template <typename... Args>
    inline void emplace_back(Args&&... args);
    inline void push_back(T&& item);
    inline const T& peek_back() const;
    inline const T& back() const { return peek_back(); }
    inline T pop_back();
    inline void eff_pop_back();
    template <typename... Args>
    inline void emplace_front(Args&&... args);
    inline void push_front(T&& item);
    inline const T& peek_front() const;
    inline const T& front() const { return peek_front(); }
    inline T pop_front();
    inline void eff_pop_front();
    inline iterator begin();
    inline iterator end();
    inline const_iterator cbegin() const;
    inline const_iterator cend() const;
    inline T& operator[](size_t index);
    inline const T& operator[](size_t index) const;
    inline T* data();
    inline const T* data() const;
};
}  // namespace lidar
}  // namespace hesai

#include "ring.cc"

#endif // !___HESAI__CONTAINER__RING_HH___
