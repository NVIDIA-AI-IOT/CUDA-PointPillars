#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <string.h>
#include "hs_com.h"

namespace hesai
{
namespace lidar
{
//half of the max value in degrees, 1 LSB represents 0.01 degree, float type
static constexpr float kHalfCircleFloat = 18000.0f;
//half of the max value in degrees, 1 LSB represents 0.01 degree, int type
static constexpr int kHalfCircleInt = 18000;
//max value in degrees
static constexpr float kCircle = 360.f;
//laser azimuth resolution, 100 units represents 1 degree, int type
static constexpr int kResolutionInt = 100;
//laser azimuth resolution, 100 units represents 1 degree, float type
static constexpr float kResolutionFloat = 100.0f;
//conversion factor between second and micorsecond
static constexpr float kMicrosecondToSecond = 1000000.0f;
static constexpr float kNanosecondToSecond = 1000000000.0f;
static constexpr int kMicrosecondToSecondInt = 1000000;
static constexpr int kNanosecondToSecondInt = 1000000000;
static constexpr int kMicrosecondToNanosecondInt = 1000;
//laser fine azimuth resolution, 1 LSB represents 0.01 / 256 degree, float type
static constexpr float kFineResolutionFloat = 256.0f;
static constexpr float kAllFineResolutionFloat = kResolutionFloat * kFineResolutionFloat;
//laser fine azimuth resolution, 1 LSB represents 0.01 / 256 degree, int type
static constexpr int kFineResolutionInt = 256;
static constexpr int kAllFineResolutionInt = kResolutionInt * kFineResolutionInt;
//synchronize host time with sensor time per kPcapPlaySynchronizationCount packets
static constexpr int kPcapPlaySynchronizationCount = 50;
//length of fault message packet
static constexpr int kFaultMessageLength = 99;
//default udp data max lenth
static const uint16_t kBufSize = 3000;
static const uint16_t kPacketBufferSize = 36000;
//min points of one frame for displaying frame message
static constexpr int kMinPointsOfOneFrame = 1000;
//max time interval between two frame
static constexpr int kMaxTimeInterval = 250000;

#define HS_MIN(x, y) ((x) > (y) ? (y) : (x))
#define HS_MAX(x, y) ((x) > (y) ? (x) : (y))

#pragma pack(push, 1)
union {
  uint32_t i;
  uint8_t c[4]; 
} static const u = {0x01020304};
#pragma pack(pop)

inline bool IsLittleEndian() { 
  if (u.c[0] == 0x04)
    return true; 
  else return false;
}  

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>  
inline T reverseBytes(T value) {  
    if (sizeof(T) == 1) return value;
    T reversed = 0;  
    for (size_t i = 0; i < sizeof(T); ++i) {  
        reversed = (reversed << 8) | (static_cast<unsigned char>(value) & 0xFF);  
        value >>= 8;  
    }  
    return reversed;  
}  

template <typename T>
T little_to_native(T data) {
  if (IsLittleEndian()) {
    return data;
  } else {
    return reverseBytes(data);
  }
}

template <typename T>
T big_to_native(T data) {
  if (!IsLittleEndian()) {
    return data;
  } else {
    return reverseBytes(data);
  }
}

template <typename T>
T native_to_little(T data) {
  if (IsLittleEndian()) {
    return data;
  } else {
    return reverseBytes(data);
  }
}

template <typename T>
T native_to_big(T data) {
  if (!IsLittleEndian()) {
    return data;
  } else {
    return reverseBytes(data);
  }
}

struct LidarPointXYZDAE
{
    float x; 
    float y;             
    float z;             
    float distance;            
    float azimuth;
    float elevation;
    uint8_t reserved[8];           
    float azimuthCalib;             
    float elevationCalib;  
};

struct UdpPacket {
  uint8_t buffer[kBufSize];
  uint16_t packet_len;
  uint64_t recv_timestamp;
  uint32_t ip;
  uint16_t port;
  bool is_timeout = false;
  UdpPacket(const uint8_t* data = nullptr, uint16_t sz = 0, uint64_t tm = 0, 
            uint32_t i_ip = 0, uint16_t i_port = 0, bool is_timeout = false)
  : packet_len(sz), recv_timestamp(tm), ip(i_ip), port(i_port), is_timeout(is_timeout)
  {
    memset(buffer, 0, kBufSize);
    if(data != nullptr)
      memcpy(buffer, data, packet_len);
  }
};

typedef std::vector<uint8_t> u8Array_t;
typedef std::vector<uint16_t> u16Array_t;
typedef std::vector<uint32_t> u32Array_t;
typedef std::vector<double> doubleArray_t;
typedef std::vector<u8Array_t> u8ArrayArray_t;
typedef std::vector<doubleArray_t> doubleArrayArray_t;
typedef std::vector<u16Array_t> u16ArrayArray_t;
typedef std::vector<std::string> stringArray_t;
typedef std::vector<UdpPacket> UdpFrame_t;
typedef std::vector<UdpFrame_t> UdpFrameArray_t;

struct LidarOpticalCenter {
  float x;
  float y;
  float z;
  LidarOpticalCenter() {
    x = 0;
    y = 0;
    z = 0;
  }
  LidarOpticalCenter(float _x, float _y, float _z) {
    x = _x;
    y = _y;
    z = _z;
  }
  void setNoFlag(LidarOpticalCenter other) {
    x = other.x;
    y = other.y;
    z = other.z;
  }
  LidarOpticalCenter& operator=(LidarOpticalCenter&) = delete;  
};

#pragma pack(push, 1)


struct FunctionSafety {
  uint8_t fs_version;
  uint8_t status;
  uint8_t fault_info;
  uint16_t fault_code;
  bool is_valid;
  uint8_t getVersion() const { return fs_version; }
  uint8_t getLidarState() const { return (status >> 5) & 0x07; }
  uint8_t getFaultType() const { return (status >> 3) & 0x03; }
  bool isHistoryFault() const { return (status & 0x10) == 0x10; }
  bool isCurrentFault() const { return (status & 0x08) == 0x08; }
  uint8_t getRollingCnt() const { return status & 0x07; }
  uint8_t getFaultNum() const { return (fault_info >> 4) & 0x0F; }
  uint8_t getFaultID() const { return fault_info & 0x0F; }
  uint16_t getFaultCode() const { return fault_code; }
};

struct PointDecodeData {
  float azimuth;
  float elevation;
  uint16_t distance;
  uint16_t channel_index;
  union {
    struct {
      int32_t ns_offset;
    } dnsOff;
    struct {
      uint8_t confidence;
      uint8_t weightFactor;
      uint8_t envLight;
      uint8_t angleState;
      int32_t ns_offset;
    } d1_4;
    struct {
      uint8_t dirtyDegree;
    } d1_8;
    struct {
      uint8_t confidence;
      uint8_t loopIndex;
      int32_t ns_offset;
    } dQT;
    struct {
      uint8_t confidence;
      uint8_t field;
      float azimuth_offset;
      uint8_t glare_in_d;
    } dAT;
    struct
    {
      uint8_t confidence;
    } dcfd;
    struct 
    {
      uint8_t confidence;
      int32_t ns_offset;
    } dXT;
    struct
    {
      uint8_t confidence;
      uint16_t point_id;
    } dET;
  } data;
  uint8_t reflectivity;
};

struct PacketDecodeData {
  union {
    uint64_t sensor_timestamp;
    struct {
      uint32_t s;
      uint32_t ns;
    } s_ns;
  } t;
  
  uint16_t spin_speed; 
  union {
    struct {
      uint8_t optMode;
    } d1_4;
    struct {
      uint8_t mirror_index;
    } dET;
    struct {
      uint8_t frame_id;
    } dATX;
    struct {
      uint16_t ns; // [0,1000)
    } dNs;
    struct {
      uint8_t column_id;
    } dFT;
  } data;
  uint8_t echo_count;
  uint8_t echo_num;
};
#pragma pack(pop)

struct LidarDecodeConfig {
    int fov_start;
    int fov_end;

    LidarDecodeConfig() {
      fov_start = -1;
      fov_end = -1;
    }
};

class SHA256_USE {  
public:  
    SHA256_USE() {  
        // Initialize the hash values  
        state = {{  
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,  
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19  
        }};  
        bufferIndex = 0;  
        bitcount = 0;  
    }  

    void update(const char* data, size_t length) {  
        for (size_t i = 0; i < length; i++) {  
            buffer[bufferIndex++] = data[i];  
            if (bufferIndex == 64) {  
                transform(buffer.data());  
                bufferIndex = 0;  
            }  
            bitcount += 8;  
        }  
    }  

    void hexdigest(uint8_t *hash) {  
        pad();  
        for (auto data : state) {
          *hash = (data >> 24) & 0xFF;
          *(hash + 1) = (data >> 16) & 0xFF;
          *(hash + 2) = (data >> 8) & 0xFF;
          *(hash + 3) = (data >> 0) & 0xFF;
          hash += sizeof(uint32_t);
        }
    }  

private:  
    void transform(const uint8_t* data) {  
        // Prepare the message schedule  
        std::array<uint32_t, 64> w = {};  
        for (size_t i = 0; i < 16; i++) {  
            w[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |  
                   ((uint32_t)data[i * 4 + 2] << 8) | (uint32_t)data[i * 4 + 3];  
        }  

        for (size_t i = 16; i < 64; i++) {  
            uint32_t s0 = (w[i - 15] >> 7) | (w[i - 15] << (32 - 7));  
            s0 ^= (w[i - 15] >> 18) | (w[i - 15] << (32 - 18));  
            s0 ^= (w[i - 15] >> 3);  
            uint32_t s1 = (w[i - 2] >> 17) | (w[i - 2] << (32 - 17));  
            s1 ^= (w[i - 2] >> 19) | (w[i - 2] << (32 - 19));  
            s1 ^= (w[i - 2] >> 10);  
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;  
        }  

        // Initialize working variables to current hash value  
        uint32_t a = state[0];  
        uint32_t b = state[1];  
        uint32_t c = state[2];  
        uint32_t d = state[3];  
        uint32_t e = state[4];  
        uint32_t f = state[5];  
        uint32_t g = state[6];  
        uint32_t h = state[7];  

        // Compression function main loop  
        for (size_t i = 0; i < 64; i++) {  
            uint32_t S1 = (e >> 6) | (e << (32 - 6));  
            S1 ^= (e >> 11) | (e << (32 - 11));  
            S1 ^= (e >> 25) | (e << (32 - 25));  
            uint32_t ch = (e & f) ^ (~e & g);  
            uint32_t temp1 = h + S1 + ch + k[i] + w[i];  
            uint32_t S0 = (a >> 2) | (a << (32 - 2));  
            S0 ^= (a >> 13) | (a << (32 - 13));  
            S0 ^= (a >> 22) | (a << (32 - 22));  
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);  
            uint32_t temp2 = S0 + maj;  

            h = g;  
            g = f;  
            f = e;  
            e = d + temp1;  
            d = c;  
            c = b;  
            b = a;  
            a = temp1 + temp2;  
        }  

        // Add the compressed chunk to the current hash value  
        state[0] += a;  
        state[1] += b;  
        state[2] += c;  
        state[3] += d;  
        state[4] += e;  
        state[5] += f;  
        state[6] += g;  
        state[7] += h;  
    }  

    void pad() {  
        buffer[bufferIndex++] = 0x80;  
        if (bufferIndex > 56) {  
            while (bufferIndex < 64) {  
                buffer[bufferIndex++] = 0x00;  
            }  
            transform(buffer.data());  
            bufferIndex = 0;  
        }  
        while (bufferIndex < 56) {  
            buffer[bufferIndex++] = 0x00;  
        }  
        // Append the bits count  
        for (int i = 0; i < 8; i++) {  
            buffer[56 + i] = (bitcount >> (56 - i * 8)) & 0xFF;  
        }  
        transform(buffer.data());  
    }  

    std::array<uint32_t, 8> state;  
    std::array<uint8_t, 64> buffer;  
    uint64_t bitcount;  
    size_t bufferIndex;  

    const std::array<uint32_t, 64> k = {  
      0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL,
      0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL,
      0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL,
      0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
      0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL, 0x983e5152UL,
      0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
      0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL,
      0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
      0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL,
      0xd6990624UL, 0xf40e3585UL, 0x106aa070UL, 0x19a4c116UL, 0x1e376c08UL,
      0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL,
      0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
      0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL 
    };   
};  

} // namespace lidar
} // namespace hesai


