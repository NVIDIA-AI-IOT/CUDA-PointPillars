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
#include "pcap_source.h"

#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdint>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

using namespace hesai::lidar;

static constexpr uint32_t pcap_magic_number = 0xa1b2c3d4;

PcapIPHeader::PcapIPHeader(uint8_t protocol, uint16_t pkt_len)
    : ether{ {0xff,0xff,0xff,0xff,0xff,0xff}, {0x00, 0x0a, 0x35, 0x00, 0x1e, 0x53}, 0x0008 }
    , ip{ 0x5, 0x4, 0x00, uint16_t((((pkt_len + 20) & 0xff) << 8) | ((pkt_len + 20) >> 8)), 0xf17a, 0x0040, 0x40, protocol, 0x80f8, 0xc901a8c0, 0xffffffff }
{}

PcapIPv6Header::PcapIPv6Header(uint8_t protocol, uint16_t pkt_len)
    : ether{ {0xff,0xff,0xff,0xff,0xff,0xff}, {0x00, 0x0a, 0x35, 0x00, 0x1e, 0x53}, 0xdd86 }
    , ipv6 { 0x6, 0, 0, 0, uint8_t((((pkt_len + 40) & 0xff) << 8) | ((pkt_len + 40) >> 8)), protocol, }
{}

PcapUDPHeader::PcapUDPHeader(uint16_t pkt_len, uint16_t port)
    : PcapIPHeader::PcapIPHeader(0x11, pkt_len + 8)
    , udp{ 0x1027, uint16_t(((port & 0xff) << 8) | (port >> 8)), uint16_t((((pkt_len + 8) & 0xff) << 8) | ((pkt_len + 8) >> 8)), 0x0000 }
{}

PcapUDPv6Header::PcapUDPv6Header(uint16_t pkt_len, uint16_t port)
    : PcapIPv6Header::PcapIPv6Header(0x11, pkt_len + 8)
    , udp{ 0x1027, uint16_t(((port & 0xff) << 8) | (port >> 8)), uint16_t((((pkt_len + 8) & 0xff) << 8) | ((pkt_len + 8) >> 8)), 0x0000 }
{}

PcapTCPHeader::PcapTCPHeader(uint16_t pkt_len, uint16_t port)
    : PcapIPHeader::PcapIPHeader(0x06, pkt_len + 20)
    , tcp{ 0x1027, uint16_t(((port & 0xff) << 8) | (port >> 8)), 0x000000000, 0x00000000, 0x50, 0x00, 0x0000, 0x0000, 0x0000 }
{}

PcapTCPv6Header::PcapTCPv6Header(uint16_t pkt_len, uint16_t port)
    : PcapIPv6Header::PcapIPv6Header(0x06, pkt_len + 20)
    , tcp{ 0x1027, uint16_t(((port & 0xff) << 8) | (port >> 8)), 0x000000000, 0x00000000, 0x50, 0x00, 0x0000, 0x0000, 0x0000 }
{}

class PcapSource::Private {
private:
#ifdef _MSC_VER
    // Windows平台使用HANDLE
    HANDLE _fd;
    HANDLE _file_mapping;
#else
    // Linux平台使用文件描述符
    int _fd;
#endif
    // 当前映射的内存地址
    void* _mapped_memory;
    // 当前映射的大小
    size_t _mapped_size;
    // 当前映射的起始偏移
    size_t _mapped_offset;
    // 文件总大小
    size_t _file_size;
    // 当前文件偏移
    size_t _fpos;
    // 块大小
    static const size_t CHUNK_SIZE = 64 * 1024 * 1024; // 64MB
    // 是否已经映射
    bool _is_mapped;

public:
    Private() : 
#ifdef _MSC_VER
        _fd(INVALID_HANDLE_VALUE),
        _file_mapping(NULL),
#else
        _fd(-1),
#endif
        _mapped_memory(nullptr), 
        _mapped_size(0),
        _mapped_offset(0), 
        _file_size(0), 
        _fpos(0), 
        _is_mapped(false) {}

    ~Private() { 
        close(); 
    }

    void open(const std::string& pcap_path) {
        close();
        
#ifdef _MSC_VER
        // Windows平台打开文件
        _fd = CreateFileA(pcap_path.c_str(),
                         GENERIC_READ,
                         FILE_SHARE_READ,
                         NULL,
                         OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL,
                         NULL);
        if (_fd == INVALID_HANDLE_VALUE) {
            LogError("Failed to open file: %s", pcap_path.c_str());
            return;
        }

        // 获取文件大小
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(_fd, &file_size)) {
            CloseHandle(_fd);
            LogError("Failed to get file size");
            _fd = INVALID_HANDLE_VALUE;
            return;
        }
        _file_size = file_size.QuadPart;
#else
        // Linux平台打开文件
        _fd = ::open(pcap_path.c_str(), O_RDONLY);
        if (_fd == -1) {
            LogError("Failed to open file: %s", pcap_path.c_str());
            return;
        }

        // 获取文件大小
        struct stat st;
        if (fstat(_fd, &st) == -1) {
            ::close(_fd);
            LogError("Failed to get file size");
            _fd = -1;
            return;
        }
        _file_size = st.st_size;
#endif

        // 映射第一个块
        map_chunk(0);
    }

    bool is_open() const { 
#ifdef _MSC_VER
        return _fd != INVALID_HANDLE_VALUE;
#else
        return _fd != -1;
#endif
    }

    void close() {
        if (_is_mapped) {
#ifdef _MSC_VER
            UnmapViewOfFile(_mapped_memory);
            if (_file_mapping) {
                CloseHandle(_file_mapping);
                _file_mapping = NULL;
            }
#else
            munmap(_mapped_memory, _mapped_size);
#endif
            _mapped_memory = nullptr;
            _is_mapped = false;
        }
#ifdef _MSC_VER
        if (_fd != INVALID_HANDLE_VALUE) {
            CloseHandle(_fd);
            _fd = INVALID_HANDLE_VALUE;
        }
#else
        if (_fd != -1) {
            ::close(_fd);
            _fd = -1;
        }
#endif
        _mapped_size = 0;
        _mapped_offset = 0;
        _fpos = 0;
    }

    inline bool eof() const { 
        return _fpos >= _file_size; 
    }

    inline size_t fpos() const { 
        return _fpos; 
    }

    inline void fpos(size_t new_fpos) { 
        if (new_fpos > _file_size) {
            LogError("Position out of file range");
            return;
        }
        _fpos = new_fpos;
        ensure_mapped();
    }

    template<typename T>
    bool read(T& value) {
        if (_fpos + sizeof(T) > _file_size) return false;
        
        check_and_map_chunk(_fpos);
        // 检查是否需要跨块读取
        if (_fpos + sizeof(T) > _mapped_offset + _mapped_size) {
            char temp_buffer[sizeof(T)];
            read_across_one_chunks(temp_buffer, sizeof(T), _fpos);
            
            // 组合数据
            value = *(T*)temp_buffer;
            _fpos += sizeof(T);
            return true;
        }

        value = *(T*)(static_cast<char*>(_mapped_memory) + (_fpos - _mapped_offset));
        _fpos += sizeof(T);
        return true;
    }

    bool read(void* dst, size_t length) {
        if (length == 0) return true;
        if (_fpos + length > _file_size) return false;

        size_t remaining = length;
        size_t current_pos = _fpos;
        char* current_dst = static_cast<char*>(dst);

        while (remaining > 0) {
            check_and_map_chunk(current_pos);
            // 计算当前块中可读取的数据量
            size_t available = _mapped_size - (current_pos - _mapped_offset);
            size_t to_read = std::min(remaining, available);

            // 复制数据
            std::memcpy(current_dst,
                       static_cast<char*>(_mapped_memory) + (current_pos - _mapped_offset),
                       to_read);

            current_dst += to_read;
            current_pos += to_read;
            remaining -= to_read;
            _fpos = current_pos;   
        }

        return true;
    }

    bool move(size_t length) {
        if (_fpos + length > _file_size) return false;
        
        size_t new_pos = _fpos + length;
        check_and_map_chunk(new_pos);
        
        _fpos = new_pos;
        return true;
    }

private:
    void map_chunk(size_t offset) {
        if (offset >= _file_size) {
            return;
        }
        if (_is_mapped) {
#ifdef _MSC_VER
            UnmapViewOfFile(_mapped_memory);
            if (_file_mapping) {
                CloseHandle(_file_mapping);
                _file_mapping = NULL;
            }
#else
            munmap(_mapped_memory, _mapped_size);
#endif
            _mapped_memory = nullptr;
            _is_mapped = false;
        }

        // 确保偏移量是块大小的整数倍
        offset = (offset / CHUNK_SIZE) * CHUNK_SIZE;
        
        // 计算要映射的大小
        _mapped_size = std::min(CHUNK_SIZE, _file_size - offset);
        _mapped_offset = offset;

#ifdef _MSC_VER
        // Windows平台内存映射
        _file_mapping = CreateFileMapping(_fd,
                                        NULL,
                                        PAGE_READONLY,
                                        0,
                                        0,
                                        NULL);
        if (!_file_mapping) {
            LogFatal("Failed to create file mapping");
            _is_mapped = false;
            return;
        }

        _mapped_memory = MapViewOfFile(_file_mapping,
                                     FILE_MAP_READ,
                                     (DWORD)(offset >> 32),
                                     (DWORD)(offset & 0xFFFFFFFF),
                                     _mapped_size);
        if (!_mapped_memory) {
            CloseHandle(_file_mapping);
            _file_mapping = NULL;
            LogFatal("Failed to map view of file");
            _is_mapped = false;
            return;
        }
#else
        // Linux平台内存映射
        _mapped_memory = mmap(nullptr, _mapped_size, PROT_READ, 
                             MAP_PRIVATE, _fd, offset);
        
        if (_mapped_memory == MAP_FAILED) {
            LogFatal("Failed to map file chunk");
            _is_mapped = false;
            return;
        }
#endif

        _is_mapped = true;
    }

    void map_next_chunk() {
        size_t new_offset = _mapped_offset + CHUNK_SIZE;
        if (new_offset >= _file_size) {
            return;
        }
        map_chunk(new_offset);
    }

    void check_and_map_chunk(size_t pos) {
        if (_is_mapped && pos >= _mapped_offset && pos < _mapped_offset + _mapped_size) {
            return;
        }
        map_chunk(pos);
    }

    void ensure_mapped() {
        // 检查当前偏移是否在已映射的范围内
        if (!_is_mapped || _fpos < _mapped_offset || 
            _fpos >= _mapped_offset + _mapped_size) {
            // 计算新的映射起始位置
            size_t new_offset = (_fpos / CHUNK_SIZE) * CHUNK_SIZE;
            map_chunk(new_offset);
        }
    }

    // 添加一个辅助函数来处理跨块读取
    bool read_across_one_chunks(void* dst, size_t length, size_t current_pos) {
        if (length > CHUNK_SIZE) {
            return false;
        }
        check_and_map_chunk(current_pos);
        if (!_is_mapped || current_pos < _mapped_offset || 
            current_pos >= _mapped_offset + _mapped_size) {
            return false;
        }
        // 计算当前块剩余可读数据
        size_t remaining_in_current = _mapped_offset + _mapped_size - current_pos;

        if (length <= remaining_in_current) {
            std::memcpy(dst,
                        static_cast<char*>(_mapped_memory) + (current_pos - _mapped_offset),
                        length);
            return true;
        }
        // 先读取当前块剩余的数据
        std::memcpy(dst,
                    static_cast<char*>(_mapped_memory) + (current_pos - _mapped_offset),
                    remaining_in_current);
        
        // 映射下一个块
        size_t next_chunk = (current_pos / CHUNK_SIZE) + 1;
        if (next_chunk * CHUNK_SIZE >= _file_size) {
            return false;
        }
        map_chunk(next_chunk * CHUNK_SIZE);
        if (!_is_mapped || next_chunk * CHUNK_SIZE < _mapped_offset || 
            next_chunk * CHUNK_SIZE >= _mapped_offset + _mapped_size || length - remaining_in_current > _mapped_size) {
            return false;
        }

        // 读取下一块的数据
        std::memcpy(static_cast<char*>(dst) + remaining_in_current,
                    _mapped_memory,
                    length - remaining_in_current);
        
        return true;
    }
};

PcapSource::PcapSource(std::string path, int packet_interval)
    : _p(new Private)
    , pcap_udp_header_{0, 0, 0, 0}
{
    pcap_path_ = path;
    packet_interval_ = packet_interval;
}

PcapSource::~PcapSource() {
    PcapSource::Close();
    delete _p;
}

PcapSource::Callback PcapSource::callback() const {
    return udp_callback_;
}

void PcapSource::callback(Callback callback) {
    udp_callback_ = callback;
}
PcapSource::Callback PcapSource::tcp_callback() const {
    return tcp_callback_;
}

void PcapSource::tcp_callback(Callback callback) {
    tcp_callback_ = callback;
}

size_t PcapSource::fpos() const
{
    return _p->fpos();
}

void PcapSource::fpos(size_t new_fpos)
{
    _p->fpos(new_fpos);
}

bool PcapSource::Open() {

    try
    {
        PcapSource::Close();
        _p->open(pcap_path_);
        if (!_p->is_open()) {
            throw std::runtime_error(std::string("Fail to open .pcap file: \"") + (pcap_path_)+"\"");
        }
        _p->read(pcap_header_);
        if (pcap_header_.magic_number != pcap_magic_number) {
            throw std::runtime_error(std::string("Not a valid .pcap file: \"") + (pcap_path_)+"\"");
        }
        begin_pos = _p->fpos();
        return true;
    }
    catch (const std::exception& e)
    {
        LogError("%s", e.what());
        return false;
    }

}

std::string PcapSource::pcap_path() const {
    return pcap_path_;
}

int PcapSource::next() {                 
    if(_p->eof()) {
        if (is_loop) {
            _p->fpos(begin_pos);
        }
        else {
            is_pcap_end = true;
            return -1;
        }
    }
    bool ret = _p->read(pcap_record_);
    if (!ret) return -1;
    if (pcap_record_.incl_len <= kBufSize && pcap_record_.incl_len > sizeof(Ethernet)) {
        ret = _p->read(payload_.data(), pcap_record_.incl_len);
        if (!ret) return -1;
        switch (((Ethernet*)payload_.data())->ether_type)
        {
        case 0x0008: // ipv4
            switch (((PcapIPHeader*)payload_.data())->ip.protocol)
            {
            case 17:
                if (pcap_record_.incl_len < sizeof(PcapUDPHeader)) return 1;
                pcap_udp_header_ = *(const UDP*)(payload_.data() + sizeof(PcapIPHeader));
                if (udp_callback_)
                    return udp_callback_(payload_.data() + sizeof(PcapUDPHeader), pcap_record_.incl_len - sizeof(PcapUDPHeader));
                else 
                    return 2;
                break;
            case 6:
                if (pcap_record_.incl_len < sizeof(PcapTCPHeader)) return 1;
                pcap_tcp_header_ = *(const TCP*)(payload_.data() + sizeof(PcapIPHeader));
                if (tcp_callback_) {
                    if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                        return 2;
                    }
                    int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                    return tcp_callback_(payload_.data() + sizeof(PcapTCPHeader) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPHeader) - header_options_len);
                }
                else
                    return 2;
                break;
            default:
                return 2;
                break;
            }
            break;
        case 0xdd86: // ipv6 
            switch (((PcapIPv6Header*)payload_.data())->ipv6.next_header)
            {
            case 17:
                if (pcap_record_.incl_len < sizeof(PcapUDPv6Header)) return 1;
                pcap_udp_header_ = *(const UDP*)(payload_.data() + sizeof(PcapIPv6Header));
                if (udp_callback_)
                    return udp_callback_(payload_.data() + sizeof(PcapUDPv6Header), pcap_record_.incl_len - sizeof(PcapUDPv6Header));
                else 
                    return 2;
                break;
            case 6:
                if (pcap_record_.incl_len < sizeof(PcapTCPv6Header)) return 1;
                pcap_tcp_header_ = *(const TCP*)(payload_.data() + sizeof(PcapIPv6Header));
                if (tcp_callback_) {
                    if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                        return 2;
                    }
                    int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                    return tcp_callback_(payload_.data() + sizeof(PcapTCPv6Header) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPv6Header) - header_options_len);
                }
                else
                    return 2;
                break;
            default:
                return 2;
                break;
            }
            break;
        case 0x0081:  // vlan tag
        switch (*((uint16_t*)(payload_.data() + sizeof(Ethernet) + 2)))
            {
            case 0x0008:
                switch (*((uint8_t*)(payload_.data() + sizeof(Ethernet) + 13)))
                {
                case 17:
                    if (pcap_record_.incl_len < sizeof(PcapUDPHeader) + 4) return 1;
                    pcap_udp_header_ = *(const UDP*)(payload_.data() + sizeof(PcapIPHeader) + 4);
                    if (udp_callback_)
                        return udp_callback_(payload_.data() + sizeof(PcapUDPHeader) + 4, pcap_record_.incl_len - (sizeof(PcapUDPHeader) + 4));
                    else
                        return 2;
                    break;
                case 6:
                    if (pcap_record_.incl_len < sizeof(PcapTCPHeader) + 4) return 1;
                    pcap_tcp_header_ = *(const TCP*)(payload_.data() + sizeof(PcapIPHeader) + 4);
                    if (tcp_callback_) {
                        if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                            return 2;
                        }
                        int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                        return tcp_callback_(payload_.data() + sizeof(PcapTCPHeader) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPHeader) - header_options_len);
                    }
                    else
                        return 2;
                    break;
                default:
                    return 2;
                    break;
                }
                break;
            case 0xdd86:
                switch (*((uint8_t*)(payload_.data() + sizeof(Ethernet) + 13)))
                {
                case 17:
                    if (pcap_record_.incl_len < sizeof(PcapUDPv6Header) + 4) return 1;
                    pcap_udp_header_ = *(const UDP*)(payload_.data() + sizeof(PcapIPv6Header) + 4);
                    if (udp_callback_)
                        return udp_callback_(payload_.data() + sizeof(PcapUDPv6Header) + 4, pcap_record_.incl_len - (sizeof(PcapUDPv6Header) + 4));
                    else 
                        return 2;
                    break;
                case 6:
                    if (pcap_record_.incl_len < sizeof(PcapTCPv6Header) + 4) return 1;
                    pcap_tcp_header_ = *(const TCP*)(payload_.data() + sizeof(PcapIPv6Header) + 4);
                    if (tcp_callback_) {
                        if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                            return 2;
                        }
                        int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                        return tcp_callback_(payload_.data() + sizeof(PcapTCPv6Header) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPv6Header) - header_options_len);
                    }
                    else
                        return 2;
                    break;
                default:
                    return 2;
                    break;
                }
            default:
                return 2;
                break;
            }
            break;
        default:
            LogWarning("can not parser Ethernet data, type = %x ", ((Ethernet*)payload_.data())->ether_type);
            break;
        }
    }
    else if (tcp_callback_ && pcap_record_.incl_len > kBufSize && pcap_record_.incl_len <= kBufSize * 4) {
        ret = _p->read(payload_tcp_.data(), pcap_record_.incl_len);
        if (!ret) return -1;
        switch (((Ethernet*)payload_tcp_.data())->ether_type)
        {
        case 0x0008: // ipv4
            switch (((PcapIPHeader*)payload_tcp_.data())->ip.protocol)
            {
            case 6:
                if (pcap_record_.incl_len < sizeof(PcapTCPHeader)) return 1;
                pcap_tcp_header_ = *(const TCP*)(payload_tcp_.data() + sizeof(PcapIPHeader));
                if (tcp_callback_) {
                    if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                        return 2;
                    }
                    int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                    return tcp_callback_(payload_tcp_.data() + sizeof(PcapTCPHeader) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPHeader) - header_options_len);
                }
                else
                    return 2;
                break;
            default:
                return 2;
                break;
            }
            break;
        case 0xdd86: // ipv6 
            switch (((PcapIPv6Header*)payload_tcp_.data())->ipv6.next_header)
            {
            case 6:
                if (pcap_record_.incl_len < sizeof(PcapTCPv6Header)) return 1;
                pcap_tcp_header_ = *(const TCP*)(payload_tcp_.data() + sizeof(PcapIPv6Header));
                if (tcp_callback_) {
                    if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                        return 2;
                    }
                    int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                    return tcp_callback_(payload_tcp_.data() + sizeof(PcapTCPv6Header) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPv6Header) - header_options_len);
                }
                else
                    return 2;
                break;
            default:
                return 2;
                break;
            }
            break;
        case 0x0081:  // vlan tag
        switch (*((uint16_t*)(payload_tcp_.data() + sizeof(Ethernet) + 2)))
            {
            case 0x0008:
                switch (*((uint8_t*)(payload_tcp_.data() + sizeof(Ethernet) + 13)))
                {
                case 6:
                    if (pcap_record_.incl_len < sizeof(PcapTCPHeader) + 4) return 1;
                    pcap_tcp_header_ = *(const TCP*)(payload_tcp_.data() + sizeof(PcapIPHeader) + 4);
                    if (tcp_callback_) {
                        if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                            return 2;
                        }
                        int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                        return tcp_callback_(payload_tcp_.data() + sizeof(PcapTCPHeader) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPHeader) - header_options_len);
                    }
                    else
                        return 2;
                    break;
                default:
                    return 2;
                    break;
                }
                break;
            case 0xdd86:
                switch (*((uint8_t*)(payload_tcp_.data() + sizeof(Ethernet) + 13)))
                {
                case 6:
                    if (pcap_record_.incl_len < sizeof(PcapTCPv6Header) + 4) return 1;
                    pcap_tcp_header_ = *(const TCP*)(payload_tcp_.data() + sizeof(PcapIPv6Header) + 4);
                    if (tcp_callback_) {
                        if(pcap_tcp_header_.getLenres() < sizeof(TCP)) {
                            return 2;
                        }
                        int header_options_len = pcap_tcp_header_.getLenres() - sizeof(TCP);
                        return tcp_callback_(payload_tcp_.data() + sizeof(PcapTCPv6Header) + header_options_len, pcap_record_.incl_len - sizeof(PcapTCPv6Header) - header_options_len);
                    }
                    else
                        return 2;
                    break;
                default:
                    return 2;
                    break;
                }
            default:
                return 2;
                break;
            }
            break;
        default:
            break;
        }
    }
    else {
        ret = _p->move(pcap_record_.incl_len);
        if (!ret) return -1;
        return 1;
    }
    return 1;
}
int PcapSource::distinationPort() {
    uint16_t port = pcap_udp_header_.distination_port;
    return ((port & 0xff) << 8) | ((port >> 8) & 0xff);
}

void PcapSource::Close() {
    _p->close();
}


bool PcapSource::IsOpened() {
    return _p->is_open();
}

int PcapSource::Receive(UdpPacket& udpPacket, uint16_t u16Len, int flags,
                       int iTimeout) {
    
    callback([&](
        const uint8_t* data,
        uint32_t len
        ) ->int {
            memcpy(udpPacket.buffer, data, len);
            return len;
        }
    );
    return next();
}

int PcapSource::Send(uint8_t* u8Buf, uint16_t u16Len, int flags) {
    (void)u8Buf;
    (void)flags;
    return u16Len;
}

void PcapSource::setPcapPath(std::string path) {
    pcap_path_ = path;
}

void PcapSource::setPacketInterval(int microsecond) {
    packet_interval_ = microsecond;
}