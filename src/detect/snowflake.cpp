#include "snowflake.hpp"
#include <chrono>
#include <thread>
#include <stdexcept>
#include <string>

namespace snowflake {

Snowflake::Snowflake(int datacenterId, int workerId, int64_t epoch)
    : datacenterId_(datacenterId)
    , workerId_(workerId)
    , sequence_(0)
    , lastTimestamp_(-1)
    , epoch_(epoch)
{
    // 验证数据中心ID范围
    if (datacenterId_ < 0 || datacenterId_ > maxDatacenterId) {
        throw std::invalid_argument("datacenterId must be between 0 and " + 
                                   std::to_string(maxDatacenterId));
    }
    
    // 验证机器ID范围
    if (workerId_ < 0 || workerId_ > maxWorkerId) {
        throw std::invalid_argument("workerId must be between 0 and " + 
                                   std::to_string(maxWorkerId));
    }
}

int64_t Snowflake::nextId() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int64_t timestamp = currentTimestamp();
    
    // 检测时钟回拨
    if (timestamp < lastTimestamp_) {
        throw std::runtime_error(std::string("Clock moved backwards. Refusing to generate id. ") +
                                "Last timestamp: " + std::to_string(lastTimestamp_) +
                                ", Current timestamp: " + std::to_string(timestamp));
    }
    
    // 同一毫秒内
    if (timestamp == lastTimestamp_) {
        sequence_ = (sequence_ + 1) & sequenceMask;
        
        // 序列号溢出，等待下一毫秒
        if (sequence_ == 0) {
            timestamp = waitNextMillis(lastTimestamp_);
        }
    } else {
        // 新的毫秒，序列号重置为0
        sequence_ = 0;
    }
    
    lastTimestamp_ = timestamp;
    
    // 组装ID
    return ((timestamp - epoch_) << timestampLeftShift) |
           (static_cast<int64_t>(datacenterId_) << datacenterIdShift) |
           (static_cast<int64_t>(workerId_) << workerIdShift) |
           sequence_;
}

int64_t Snowflake::currentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

int64_t Snowflake::waitNextMillis(int64_t lastTimestamp) const {
    int64_t timestamp = currentTimestamp();
    while (timestamp <= lastTimestamp) {
        timestamp = currentTimestamp();
        // 如果时间戳仍然相同，短暂休眠避免CPU占用过高
        if (timestamp == lastTimestamp) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return timestamp;
}

} // namespace snowflake
