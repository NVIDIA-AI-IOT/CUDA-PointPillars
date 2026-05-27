/************************************************************************************************
Copyright (C) 2023 Hesai Technology Co., Ltd.
Copyright (C) 2023 Original Authors
All rights reserved.

All code in this repository is released under the terms of the following Modified BSD License. 
Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and 
  the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
  the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************************************************************/
#ifndef PCPASAVER_H_
#define PCPASAVER_H_

#include <functional>
#include <array>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <string>
#include <fstream>
#include <memory>
#include <queue>
#include <cstring>
#include "pcap_source.h"
#include "blocking_ring.h"
#include <chrono>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include "pcap_source.h"
#include "lidar_types.h"
namespace hesai
{
namespace lidar
{
using Clock = std::chrono::system_clock;
using Time = Clock::time_point;
using Period = Clock::period;
using Duration = Clock::duration;

struct PandarPacket {
  uint32_t size;
  uint16_t port;
  uint8_t buffer[kBufSize];
  PandarPacket(const uint8_t *data = nullptr, uint32_t sz = 0, uint16_t prt = 0)
      : size(sz), port(prt) {
    if (data != nullptr) {
      memcpy(buffer, data, size);
    }
  }
};

class PcapSaver {
public:
    bool tcp_dumped_;
private:
    std::ofstream ofs_;
    std::string pcap_path_;

    using Container = BlockingRing<PandarPacket, 32*1024>;
    std::shared_ptr<Container> packets_cache_;
    bool dumping_;
    bool dumping_blocked_;
    // std::mutex _mutex;
    std::thread dumping_thread_;
    uint8_t* write_buffer_;
    size_t buffer_pos_;
    void flush_buffer();
    void write_to_buffer(const void* data, size_t size);
public:
    PcapSaver();
    PcapSaver(const PcapSaver&) = delete;
    PcapSaver& operator=(const PcapSaver&) = delete;
    ~PcapSaver();

    std::string pcap_path() const;
    void SetPcapPath(std::string pcap_path);
    int Save();
    int Save(const std::string &recordPath, const UdpFrame_t &packets,
            int port = 2368);
    int Save(const std::string &recordPath, const UdpFrameArray_t &packets,
            int port = 2368);
    void Dump(const uint8_t*, uint32_t, uint16_t port = 2368);
    void TcpDump(const uint8_t*, uint32_t, uint32_t max_pkt_len = kBufSize - sizeof(PcapTCPHeader), uint16_t port = 2368);
    void close();
};
}  // namespace lidar
}  // namespace hesai


#endif // PCPASTREAMER_H_
