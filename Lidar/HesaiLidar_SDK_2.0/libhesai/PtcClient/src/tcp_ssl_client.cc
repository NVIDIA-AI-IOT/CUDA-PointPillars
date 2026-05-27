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

/*
 * File:   tcp_ssl_client_boost.cc
 * Author: zhang xu<int_zhangxu@hesaitech.com>
 */
#include "tcp_ssl_client.h"

#include <plat_utils.h>

#ifdef _MSC_VER
#include <winsock2.h>
#include <ws2tcpip.h> 
#pragma comment(lib, "ws2_32.lib")  // Winsock Library
#include <BaseTsd.h>

typedef int socklen_t;
#define MSG_DONTWAIT (0x40)
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h> 
#endif
using namespace hesai::lidar;
using std::placeholders::_1;
using std::placeholders::_2;
static int tcp_try_open(const char* ipaddr, int port, uint32_t timeout) {
  #ifdef _MSC_VER
  WSADATA wsaData;
  WORD version = MAKEWORD(2, 2);
  int res = WSAStartup(version, &wsaData);  // win sock start up
  if (res) {
    LogError("Initilize winsock error !");
    return -1;
  }
#endif  
  struct sockaddr_in serverAddr;
  int sockfd;
  sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  if ((int)sockfd == -1) { 
#ifdef _MSC_VER
    WSACleanup();
#endif
    return -1;
  }

  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(static_cast<u_short>(port));
  if (inet_pton(AF_INET, ipaddr, &serverAddr.sin_addr) <= 0) {
#ifdef _MSC_VER
    closesocket(sockfd);
    WSACleanup();
#else
    close(sockfd);
#endif
    LogError("tcp_try_open inet_pton error:%s", ipaddr);
    return -1;
  }

  // 设置非阻塞模式  
#ifdef _MSC_VER  
  u_long mode = 1; // 1为非阻塞模式  
  ioctlsocket(sockfd, FIONBIO, &mode);  
#else  
  int flags = fcntl(sockfd, F_GETFL, 0); 
  fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);   
#endif 

  int result = connect(sockfd, (sockaddr*)&serverAddr, sizeof(serverAddr));  
  if (result < 0) {  
#ifdef _MSC_VER  
    if (WSAGetLastError() != WSAEWOULDBLOCK) 
#else  
    if (errno != EINPROGRESS)  
#endif  
    {
      LogError("socket Connection error.");  
#ifdef _MSC_VER
      closesocket(sockfd);
      WSACleanup();
#else
      close(sockfd);
#endif
      return -1;  
    }  
  }

  fd_set writefds;  
  FD_ZERO(&writefds);  
  FD_SET(sockfd, &writefds);  
  struct timeval tv;  
  tv.tv_sec = timeout; // 超时1秒  
  tv.tv_usec = 0;   
  result = select(sockfd + 1, nullptr, &writefds, nullptr, &tv);  
  if (result <= 0) {  
#ifdef _MSC_VER
    closesocket(sockfd);
    WSACleanup();
#else
    close(sockfd);
#endif
    return -1;  
  } else {
    int slen = sizeof(int);
    int error = -1;
    getsockopt(sockfd, SOL_SOCKET, SO_ERROR, (char *)&error, (socklen_t *)&slen);
    if (error != 0) {
#ifdef _MSC_VER
    closesocket(sockfd);
    WSACleanup();
#else
    close(sockfd);
#endif
      return false;
    }
  } 

#ifdef _MSC_VER  
  mode = 0; // 0为阻塞模式  
  ioctlsocket(sockfd, FIONBIO, &mode);  
#else  
  flags = fcntl(sockfd, F_GETFL, 0); 
  fcntl(sockfd, F_SETFL, flags & ~O_NONBLOCK); 
#endif  
  return sockfd;
}

static int tcp_open(const char* ipaddr, int port) {
#ifdef _MSC_VER
  WSADATA wsaData;
  WORD version = MAKEWORD(2, 2);
  int res = WSAStartup(version, &wsaData);  // win sock start up
  if (res) {
      LogError("Initilize winsock error !");
      return false;
  }
#endif  
  int sockfd;
  struct sockaddr_in servaddr;
  LogInfo("ip:%s port:%d", ipaddr, port);

  if ((sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1){
    LogError("socket errno:%d, %s", errno, strerror(errno));
    return -1;
  }
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(static_cast<u_short>(port));
  if (inet_pton(AF_INET, ipaddr, &servaddr.sin_addr) <= 0) {
#ifdef _MSC_VER
          closesocket(sockfd);
          WSACleanup();
#else
          close(sockfd);
#endif
    sockfd = -1;
    LogError("tcp_open inet_pton error:%s", ipaddr);
    return sockfd;
  }

  if (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
    LogError("connect errno:%d, %s",errno,strerror(errno));
#ifdef _MSC_VER
          closesocket(sockfd);
          WSACleanup();
#else
          close(sockfd);
#endif
    sockfd = -1;
    return sockfd;
  }

  return sockfd;
}

static int sys_readn_by_ssl(SSL *ssl, void *vptr, int n)
{
    int nleft;
    char *ptr;

    ptr = (char *)(vptr);
    nleft = n;
    while (nleft > 0) {
      int nread = SSL_read(ssl, ptr, nleft);
      if (nread < 0) {
          if (errno == EINTR)
              nread = 0;
          else
              return -1;
      }
      else if (nread == 0)
          break;

      nleft -= nread;
      ptr += nread;
    }
    return n - nleft;
}

/*
 *
 * 调用顺序：TcpSslClient() -> InitSslClient() -> Open()
 * 
 */
TcpSslClient::TcpSslClient() {
  m_sServerIP.clear();
  ptc_port_ = 0;
  tcpsock_ = (SOCKET)(-1);
  m_bLidarConnected = false;
  m_u32ReceiveBufferSize = 4096;
  ctx_ = nullptr;
  ssl_ = nullptr;
  private_key_ = nullptr;
  ca_ = nullptr;
  cert_ = nullptr;
}

TcpSslClient::~TcpSslClient() {
  Close();
}

void TcpSslClient::Close() {

  // m_sServerIP.clear();
  // ptc_port_ = 0;
  m_bLidarConnected = false;
  if((int)tcpsock_ != -1) {
#ifdef _MSC_VER
    closesocket(tcpsock_);
    WSACleanup();
#else
    close(tcpsock_);
#endif
    tcpsock_ = (SOCKET)(-1);
  }
  if (ctx_ != nullptr) {
    SSL_CTX_free(ctx_);
    ctx_ = nullptr;
  }
  if (ssl_ != nullptr) {
    SSL_shutdown(ssl_);
    ssl_ = nullptr;
  }
}

bool TcpSslClient::IsOpened() {
  return m_bLidarConnected;
}

bool TcpSslClient::IsOpened(bool bExpectation) {
  return m_bLidarConnected == bExpectation;
}

bool TcpSslClient::TryOpen(std::string IPAddr, 
                        uint16_t u16Port,
                        bool bAutoReceive, 
                        const char* cert, 
                        const char* private_key, 
                        const char* ca,
                        uint32_t timeout) {
  (void)bAutoReceive;
  if (IsOpened(true) && m_sServerIP == IPAddr && u16Port == ptc_port_) {
    return true;
  }
  if (IsOpened()) Close();
  m_sServerIP = IPAddr;
  ptc_port_ = u16Port;
  cert_ = cert;
  private_key_ = private_key;
  ca_ = ca;
  ctx_ = InitSslClient(cert, private_key, ca);
  if(ctx_ == NULL) {
		return false;
	}
  tcpsock_ = tcp_try_open(m_sServerIP.c_str(), ptc_port_, timeout);
  if((int)tcpsock_ != -1) {
		LogError("Connect to Server Failed!~!~");
    Close();
    return false;
	}
  ssl_ = SSL_new(ctx_);
  if (ssl_ == NULL) {
		LogError("create ssl failed");
		Close();
    return false;
	}

  SSL_set_fd(ssl_, (int)tcpsock_);
	if(SSL_connect(ssl_) == 0) {
		LogError("connect ssl failed");
		Close();
    return false;
	}

  if(SSL_get_verify_result(ssl_) != X509_V_OK) {
		LogError("verify ssl failed");
		Close();
    return false;
	}

  LogInfo("TcpSslClient::Open() success!");
  m_bLidarConnected = true;
  return true;
}

bool TcpSslClient::Open(std::string IPAddr, 
                        uint16_t u16Port,
                        bool bAutoReceive, 
                        const char* cert, 
                        const char* private_key, 
                        const char* ca) {
  (void)bAutoReceive;
  if (IsOpened(true) && m_sServerIP == IPAddr && u16Port == ptc_port_) {
    return true;
  }
#ifdef _MSC_VER
  LogInfo("Open() IP %s port %u", IPAddr.c_str(), u16Port);
#else
  LogInfo("Open() IP %s port %u", IPAddr.c_str(), u16Port);
#endif
  Close();
  m_sServerIP = IPAddr;
  ptc_port_ = u16Port;
  cert_ = cert;
  private_key_ = private_key;
  ca_ = ca;
  return Open();
}

bool TcpSslClient::Open() {
  ctx_ = InitSslClient(cert_, private_key_, ca_);
  if(ctx_ == NULL) {
    LogError("create SSL_CTX failed");
		return false;
	}
  tcpsock_ = tcp_open(m_sServerIP.c_str(), ptc_port_);
  if((int)tcpsock_ != -1) {
		LogError("Connect to Server Failed!~!~");
    Close();
    return false;
	}
  ssl_ = SSL_new(ctx_);
  if (ssl_ == NULL) {
		LogError("create ssl failed");
		Close();
    return false;
	}

  SSL_set_fd(ssl_, (int)tcpsock_);
	if(SSL_connect(ssl_) == 0) {
		LogError("connect ssl failed");
		Close();
    return false;
	}

  if(SSL_get_verify_result(ssl_) != X509_V_OK) {
		LogError("verify ssl failed");
		Close();
    return false;
	}

  LogInfo("TcpSslClient::Open() success!");
  m_bLidarConnected = true;
  return true;
}

SSL_CTX* TcpSslClient::InitSslClient(const char* cert, const char* private_key, const char* ca) {
  SSL_library_init();
	SSL_load_error_strings();
	OpenSSL_add_all_algorithms();
	SSL_CTX *ctx = SSL_CTX_new(SSLv23_client_method());

	if(ctx == NULL) {
		LogError("create SSL_CTX failed");
		return NULL;
	}

  if (ca) {
    LogInfo("ca path: %s",ca);
		if(	SSL_CTX_load_verify_locations(ctx, ca, NULL) == 0) {
			// ERR_print_errors_fp(stderr);
			LogError("load ca failed,please check ca file path");
			return NULL;
		}
	}

	if (cert && private_key){
    LogInfo("cert path: %s,\nprivate_key path: %s",cert, private_key);
		SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
		if(SSL_CTX_use_certificate_file(ctx, cert, SSL_FILETYPE_PEM) == 0) {
      LogError("load cert file failed,please check cert file path");
			return NULL;
    }
    if(SSL_CTX_use_PrivateKey_file(ctx, private_key, SSL_FILETYPE_PEM) == 0) {
      LogError("load private key file failed,please check private key file path");
			return NULL;
    }
    if(SSL_CTX_check_private_key(ctx) == 0) {
      LogError("check private key failed");
			return NULL;
    }
	}
	return ctx;
}

int TcpSslClient::Send(uint8_t *u8Buf, uint16_t u16Len, int flags) {
  (void)flags;
  int len = -1;
  bool ret = true;

  if(!IsOpened()) ret = Open();
  if(ret) {
    len = SSL_write(ssl_, u8Buf, u16Len);
    if (len != u16Len && errno != EAGAIN && errno != EWOULDBLOCK &&
      errno != EINTR) {
      LogError("Send errno %d",  errno);
      ERR_print_errors_fp(stderr);
      Close();
      return -1;
    }
  }

  return len;
}

int TcpSslClient::Receive(uint8_t *u8Buf, uint32_t u32Len, int flags) {
  if (flags == 0xFF) return -1;
  int len = -1;
  bool ret = true;
  
  if(!IsOpened()) ret = Open();

  int tick = GetMicroTickCount();
  if(ret) {
    len = sys_readn_by_ssl(ssl_, u8Buf , (int32_t)u32Len);
    if(len != (int32_t)u32Len) {
      LogError("ssl receive failed");
		  Close();
      return -1;
    }
  }
  
  int delta = GetMicroTickCount() - tick;

  if (delta >= 1000000) {
    LogDebug("Receive execu: %dus", delta);
  }

  return len;
}


bool TcpSslClient::SetReceiveTimeout(uint32_t u32Timeout) {
  if((int)tcpsock_ != -1) {
    LogWarning("TcpClient not open");
    return false;
  }
#ifdef _MSC_VER
  int timeout_ms = u32Timeout;
  int retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout_ms,
      sizeof(timeout_ms));
#else
  uint32_t sec = u32Timeout / 1000;
  uint32_t msec = u32Timeout % 1000;

  struct timeval timeout;
  timeout.tv_sec = sec;
  timeout.tv_usec = msec * 1000;
  int retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_RCVTIMEO,
                          (const void *)&timeout, sizeof(timeval));
#endif
  return retVal == 0;
}


int TcpSslClient::SetTimeout(uint32_t u32RecMillisecond,
                          uint32_t u32SendMillisecond) {
  if((int)tcpsock_ != -1) {
    LogWarning("TcpClient not open");
    return -1;
  }
  m_u32RecTimeout = u32RecMillisecond;
  m_u32SendTimeout = u32SendMillisecond;
#ifdef _MSC_VER
  int timeout_ms = u32RecMillisecond;
  int retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_RCVTIMEO,
                        (char*)&timeout_ms, sizeof(timeout_ms));
#else
  uint32_t sec = u32RecMillisecond / 1000;
  uint32_t msec = u32RecMillisecond % 1000;

  struct timeval timeout;
  timeout.tv_sec = sec;
  timeout.tv_usec = msec * 1000;
  int retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_RCVTIMEO,
                          (const void *)&timeout, sizeof(timeval));
#endif
  if (retVal == 0) {
#ifdef _MSC_VER
    int send_timeout_ms = u32SendMillisecond;
    retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_SNDTIMEO,
                        (char*)&send_timeout_ms, sizeof(send_timeout_ms));  
#else
    uint32_t send_sec = u32SendMillisecond / 1000;
    uint32_t send_msec = u32SendMillisecond % 1000;

    struct timeval send_timeout;
    send_timeout.tv_sec = send_sec;
    send_timeout.tv_usec = send_msec * 1000;
    retVal = setsockopt(tcpsock_, SOL_SOCKET, SO_SNDTIMEO,
                        (const void *)&send_timeout, sizeof(timeval));
#endif
  }
  return retVal;
}

void TcpSslClient::SetReceiveBufferSize(const uint32_t &size) {
  if((int)tcpsock_ != -1) {
    LogWarning("TcpClient not open");
    return;
  }

  m_u32ReceiveBufferSize = size;
  uint32_t recbuffSize = 0;
  socklen_t optlen = sizeof(recbuffSize);
  int ret = getsockopt(tcpsock_, SOL_SOCKET, SO_RCVBUF, (char*)&recbuffSize, &optlen);
  if (ret == 0 && recbuffSize < size) {
    setsockopt(tcpsock_, SOL_SOCKET, SO_RCVBUF, (char*)&size, sizeof(size));
  }
}