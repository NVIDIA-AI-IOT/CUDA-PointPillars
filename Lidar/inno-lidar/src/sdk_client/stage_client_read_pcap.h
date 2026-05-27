/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef SDK_CLIENT_STAGE_CLIENT_READ_PCAP_H_
#define SDK_CLIENT_STAGE_CLIENT_READ_PCAP_H_
#include <string>
#include "sdk_client/stage_client_read.h"

#ifndef IP_DF
#define IP_DF 0x4000  // dont fragment flag
#endif
#ifndef IP_MF
#define IP_MF 0x2000  // more fragments flag
#endif
#ifndef IP_OFFMASK
#define IP_OFFMASK 0x1fff  // mask for fragmenting bits
#endif

#ifndef ETHERTYPE_IP
#define ETHERTYPE_IP 0x0800 /* ip protocol */
#endif
#ifndef DLT_LINUX_SLL
#define DLT_LINUX_SLL 113
#endif
#ifndef DLT_EN10MB
#define DLT_EN10MB 1 /* Ethernet (10Mb) */
#endif
#ifndef ETHERTYPE_VLAN
#define ETHERTYPE_VLAN 0x8100 /* IEEE 802.1Q VLAN tagging */
#endif
#ifndef ARPHRD_ETHER
#define ARPHRD_ETHER 1        /* Ethernet 10/100Mbps.  */
#endif


namespace innovusion {

#pragma pack(push, 1)
struct PcapHeader {
  uint32_t magic;
  uint16_t version_major;
  uint16_t version_minor;
  int32_t thiszone;
  uint32_t sigfigs;
  uint32_t snaplen;
  uint32_t linktype;
};

struct PcapPktHeader {
  uint64_t ts;
  uint32_t caplen;
  uint32_t len;
};

struct Etherheader {
  uint8_t dest[6];
  uint8_t src[6];
  uint16_t ether_type;
};

struct VlanTag {
  uint16_t vlan_tpid; /* ETH_P_8021Q */
  uint16_t vlan_tci;  /* VLAN TCI */
};

struct SllHeader {
  uint16_t sll_pkttype;  /* packet type */
  uint16_t sll_hatype;   /* link-layer address type */
  uint16_t sll_halen;    /* link-layer address length */
  uint8_t sll_addr[8];   /* link-layer address */
  uint16_t sll_protocol; /* protocol */
};

struct IpHeader {
  uint8_t ihl : 4, version : 4;
  uint8_t tos;
  uint16_t tot_len;
  uint16_t id;
  uint16_t frag_off;
  uint8_t ttl;
  uint8_t protocol;
  uint16_t checksum;
  uint32_t saddr;
  uint32_t dst_addr;
};
struct UdpHeader {
  uint16_t src_port;
  uint16_t dst_port;
  uint16_t length;
  uint16_t checksum;
};
#pragma pack(pop)
class InnoLidarClient;

class PcapInput : public FileInput {
 public:
  explicit PcapInput(InnoLidarClient *lidar, const void *param) : FileInput(lidar, param) {
    lidar_ip_ = input_param_.pcap_param.lidar_ip;
    filename_ = input_param_.pcap_param.filename;
    inno_log_info("filename: %s, play_round: %d", filename_, play_round_);
  }

  virtual ~PcapInput() {
  }
  int read_data() override;

  static const int UDP_PACKET_MAX_SIZE = 65536;
  static const int PCAP_CACHE_LENGTH = 64;

 protected:
  int read_fd_(char *buf, size_t len) override;

 private:
  PcapInput() = delete;
  PcapInput(const PcapInput&) = delete;
  PcapInput operator=(const PcapInput&) = delete;
  bool read_next_packet_();
  bool pcap_next(PcapPktHeader &packet_header);
  int open_file_();

 private:
  struct PcapBuffer {
    uint16_t id = 0;
    uint32_t offset = 0;
    uint32_t len = 0;
    char buf[UDP_PACKET_MAX_SIZE] = {0};
  };
  std::string lidar_ip_;
  FILE* pcap_handle_ = NULL;
  char pcap_pkt_data_[UDP_PACKET_MAX_SIZE];
  PcapHeader pcap_header_;
  PcapBuffer *curr_pkt_ = NULL;
  int32_t curr_pkt_idx_ = -1;
  PcapBuffer cache_[PCAP_CACHE_LENGTH];
  PcapBuffer single_cache_;
};
}  // namespace innovusion

#endif  // SDK_CLIENT_STAGE_CLIENT_READ_PCAP_H_
