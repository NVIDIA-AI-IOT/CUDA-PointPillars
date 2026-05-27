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
#ifndef PCPASTREAMER_H_
#define PCPASTREAMER_H_

#define NOMINMAX
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
#include "source.h"
namespace hesai
{
namespace lidar
{
#pragma pack(push, 1)
struct PcapHeader {
    uint32_t magic_number;
    uint16_t version_major;
    uint16_t version_minor;
    int32_t thiszone;
    uint32_t sigfigs;
    uint32_t snaplen;
    uint32_t network;
};
static_assert(sizeof(PcapHeader) == 24);

struct PcapRecord {
    uint32_t ts_sec;
    uint32_t ts_usec;
    uint32_t incl_len;
    uint32_t orig_len;
};
static_assert(sizeof(PcapRecord) == 16);
struct Ethernet {
    uint8_t destination_MAC_address[6];
    uint8_t source_MAC_address[6];
    uint16_t ether_type;
};
static_assert(sizeof(Ethernet) == 14);
struct PcapIPHeader {
    Ethernet ether;
    struct IP {
        uint8_t     version : 4;    /* version */
        uint8_t     length : 4;     /* header length */
        uint8_t     type_of_service;/* type of service */
        uint16_t    total_length;   /* total length */
        uint16_t    identification;	/* identification */
        uint16_t    fragment_offset;/* fragment offset field */
#define	IP_DF 0x4000			    /* dont fragment flag */
#define	IP_MF 0x2000			    /* more fragments flag */
        uint8_t	    time_to_live;	/* time to live */
        uint8_t	    protocol;		/* protocol */
        uint16_t    checksum;		/* checksum */
        uint32_t    srce_addr;      /* source address*/
        uint32_t    dist_addr;	    /* destination address */
    } ip;
    PcapIPHeader(uint8_t protocol, uint16_t pkt_len);
};
static_assert(sizeof(PcapIPHeader::IP) == 20);
struct PcapIPv6Header {
    Ethernet ether;
    struct IPv6 {
        uint32_t    version : 4;    /* version */
        uint32_t    flow_level : 8;	
        uint32_t    flow_label : 20;
        uint16_t    payload_length;
        uint8_t     next_header;    /* protocol */
        uint8_t     hop_limit;
        uint8_t     srce_addr[16];  /* source address*/
        uint8_t     dist_addr[16];	/* destination address */
    } ipv6;
    PcapIPv6Header(uint8_t protocol, uint16_t pkt_len);
};
static_assert(sizeof(PcapIPv6Header::IPv6) == 40);
struct UDP {
    uint16_t source_port;
    uint16_t distination_port;
    uint16_t length;
    uint16_t check_sum;
};
static_assert(sizeof(UDP) == 8);
struct PcapUDPHeader : public PcapIPHeader {
    UDP udp;
    PcapUDPHeader(uint16_t pkt_len, uint16_t port = 2368);
};
static_assert(sizeof(PcapUDPHeader) == 42);
struct PcapUDPv6Header : public PcapIPv6Header {
    UDP udp;
    PcapUDPv6Header(uint16_t pkt_len, uint16_t port = 2368);
};
static_assert(sizeof(PcapUDPv6Header) == 62);
struct TCP {
    uint16_t source_port;
    uint16_t distination_port;
    uint32_t seq;
    uint32_t ack;
    uint8_t len_flags;
    uint8_t flags;
    uint16_t win;
    uint16_t check_sum;
    uint16_t urp;
    uint32_t getLenres() { return ((len_flags & 0xF0) >> 4) * 4; }
    uint16_t getFlags() { return (len_flags & 0x0F) << 8 | flags; }
};
static_assert(sizeof(TCP) == 20);
struct PcapTCPHeader : public PcapIPHeader {
    TCP tcp;
    PcapTCPHeader(uint16_t pkt_len, uint16_t port = 2368);
};
static_assert(sizeof(PcapTCPHeader) == 54);
struct PcapTCPv6Header : public PcapIPv6Header {
    TCP tcp;
    PcapTCPv6Header(uint16_t pkt_len, uint16_t port = 2368);
};
static_assert(sizeof(PcapTCPv6Header) == 74);
#pragma pack(pop)

class PcapSource : public Source {
public:
    using Callback = std::function<int(const uint8_t*, uint32_t)>;
public:
    class Private;
private:
    Private* _p;
    std::string pcap_path_;
    Callback udp_callback_;
    Callback tcp_callback_;
    PcapHeader pcap_header_;
    PcapRecord pcap_record_;
    UDP pcap_udp_header_;
    TCP pcap_tcp_header_;
    std::array<uint8_t, kBufSize> payload_;
    std::array<uint8_t, kBufSize * 4> payload_tcp_;
    int packet_interval_;
    bool is_loop = false;
    int begin_pos = 0;
public:
    PcapSource(std::string path, int packet_interval);
    PcapSource(const PcapSource&) = delete;
    PcapSource& operator=(const PcapSource&) = delete;
    virtual ~PcapSource();
    void setPcapPath(std::string path);

    Callback callback() const;
    void callback(Callback);
    Callback tcp_callback() const;
    void tcp_callback(Callback);
    size_t fpos() const;
    void fpos(size_t);
    std::string pcap_path() const;
    int next();
    virtual bool Open();
    virtual void Close();

    virtual bool IsOpened();
    virtual int Send(uint8_t* u8Buf, uint16_t u16Len, int flags = 0);
    virtual int Receive(UdpPacket& udpPacket, uint16_t u16Len, int flags = 0,
                      int timeout = 20000);
    int distinationPort();
    void setPacketInterval(int microsecond);
    virtual void SetSocketBufferSize(uint32_t u32BufSize) { (void)u32BufSize; };
    virtual void SetPcapLoop(bool loop) { is_loop = loop; };
};
}  // namespace lidar
}  // namespace hesai
#endif // PCPASTREAMER_H_
