#ifndef SNOWFLAKE_HPP
#define SNOWFLAKE_HPP

#include <cstdint>
#include <mutex>
#include <chrono>
#include <stdexcept>

namespace snowflake {

/**
 * 雪花算法ID生成器
 * 生成64位唯一ID，格式：
 * - 符号位（1位）：始终为0
 * - 时间戳（41位）：毫秒级时间戳
 * - 数据中心ID（5位）：0-31
 * - 机器ID（5位）：0-31
 * - 序列号（12位）：每毫秒最多4096个ID
 */
class Snowflake {
public:
    /**
     * 构造函数
     * @param datacenterId 数据中心ID，范围0-31
     * @param workerId 机器ID，范围0-31
     * @param epoch 时间戳起始点（毫秒），默认为2020-01-01 00:00:00 UTC
     */
    Snowflake(int datacenterId, int workerId, int64_t epoch = 1577836800000LL);
    
    /**
     * 生成下一个唯一ID
     * @return 64位唯一ID
     * @throws std::runtime_error 如果检测到时钟回拨
     */
    int64_t nextId();
    
    /**
     * 获取当前时间戳（毫秒）
     * @return 当前时间戳
     */
    int64_t currentTimestamp() const;

private:
    // 位分配常量
    static constexpr int datacenterIdBits = 5;
    static constexpr int workerIdBits = 5;
    static constexpr int sequenceBits = 12;
    static constexpr int datacenterIdShift = sequenceBits;
    static constexpr int workerIdShift = sequenceBits + datacenterIdBits;
    static constexpr int timestampLeftShift = sequenceBits + datacenterIdBits + workerIdBits;
    static constexpr int64_t sequenceMask = (1LL << sequenceBits) - 1;
    static constexpr int64_t maxDatacenterId = (1LL << datacenterIdBits) - 1;
    static constexpr int64_t maxWorkerId = (1LL << workerIdBits) - 1;

    int datacenterId_;      // 数据中心ID
    int workerId_;          // 机器ID
    int64_t sequence_;      // 序列号
    int64_t lastTimestamp_; // 上次生成ID的时间戳
    int64_t epoch_;         // 时间戳起始点
    mutable std::mutex mutex_; // 互斥锁，保证线程安全

    /**
     * 等待下一毫秒
     * @param lastTimestamp 上次时间戳
     * @return 新的时间戳
     */
    int64_t waitNextMillis(int64_t lastTimestamp) const;
};

} // namespace snowflake

#endif // SNOWFLAKE_HPP
