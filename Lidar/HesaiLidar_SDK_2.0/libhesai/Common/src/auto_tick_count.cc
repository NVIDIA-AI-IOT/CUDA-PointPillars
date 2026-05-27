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

#include "auto_tick_count.h"

#include "plat_utils.h"
using namespace hesai::lidar;
TickCount::TickCount() {
  m_u64StartTime = m_u64EndTime = GetMicroTickCountU64();
}

TickCount::~TickCount() {
  m_u64EndTime = GetMicroTickCountU64();
  std::cout << "cost" << GetTimeCost().c_str() << std::endl;
  m_timeSlice.clear();
}

int TickCount::Start() {
  m_u64StartTime = m_u64EndTime = GetMicroTickCountU64();
  return 0;
}

int TickCount::Pause() {
  m_u64EndTime = GetMicroTickCountU64();
  return 0;
}

std::string TickCount::GetTimeCost(uint64_t &u64TimeCost) {
  u64TimeCost = m_u64EndTime - m_u64StartTime;
  uint32_t u32Sec = static_cast<uint32_t>(u64TimeCost / kMicroToSec);
  uint32_t u32Min = u32Sec / kClockUnit;
  uint32_t u32Hour = u32Min / kClockUnit;

  u32Sec = u32Sec % kClockUnit;
  u32Min = u32Min % kClockUnit;

  return std::to_string(u32Hour) + std::string("h ") + std::to_string(u32Min) +
         std::string("m ") + std::to_string(u32Sec) + std::string("s ") +
         std::to_string(u64TimeCost % kMicroToSec) + std::string("us");
}

std::string TickCount::GetTimeCost() {
  uint64_t u64Time;
  return GetTimeCost(u64Time);
}

int TickCount::AppentTimeSlice(std::string sKey, uint64_t u64Time) {
  if (m_timeSlice.find(sKey) == m_timeSlice.end()) {
    std::vector<uint64_t> vTimes;
    m_timeSlice[sKey] = vTimes;
  }

  m_timeSlice[sKey].push_back(u64Time);

  return 0;
}

int TickCount::ShowTimeSlice(std::string sLogFile, bool bSaveSysTime) {
  std::map<std::string, std::vector<uint64_t>>::iterator iter =
      m_timeSlice.begin();
  FILE *pFileSel = fopen(sLogFile.c_str(), "a");
  FILE *pFile = fopen("timeConsumption.txt", "a");

  // 获取系统时间写入文件
  time_t time_t_now;
  struct tm *tm_now;
  time(&time_t_now);
  tm_now = localtime(&time_t_now);

  if (bSaveSysTime) {
    char time_str[100];
    strftime(time_str, sizeof(time_str), "%a %b %d %H:%M:%S %Y", tm_now);
    if (pFile != NULL)
      fprintf(pFile, "recording time: %s-------------\n", time_str);
    if (pFileSel != NULL) {
      fprintf(pFileSel, "recording time: %s-------------\n", time_str);
    }
  }

  if (pFile != NULL) fprintf(pFile, "object %s:\n", m_sName.c_str());

  if (pFileSel != NULL) fprintf(pFileSel, "object %s:\n", m_sName.c_str());

  while (iter != m_timeSlice.end()) {
    std::cout << iter->first.c_str() << std::endl;

    if (pFile != NULL) fprintf(pFile, "%s:\n", iter->first.c_str());

    if (pFileSel != NULL) fprintf(pFileSel, "%s:\n", iter->first.c_str());

    for (size_t i = 0; i < iter->second.size(); i++) {
      std::cout << "  " << static_cast<double>(iter->second.at(i)) / kMicroToSec
               << "s\n";

      if (pFile != NULL)
        fprintf(pFile, " %lf\n",
                static_cast<double>(iter->second.at(i)) / kMicroToSec);

      if (pFileSel != NULL)
        fprintf(pFileSel, " %lf\n",
                static_cast<double>(iter->second.at(i)) / kMicroToSec);
    }

    ++iter;
  }
  m_u64EndTime = GetMicroTickCountU64();

  std::cout << "total cost:" << GetTimeCost().c_str() << std::endl;

  if (pFile != NULL) fprintf(pFile, "total cost:%s\n", GetTimeCost().c_str());

  if (pFileSel != NULL)
    fprintf(pFileSel, "total cost:%s\n", GetTimeCost().c_str());

  if(pFile != NULL){
    fclose(pFile);
    pFile = NULL;
  }
  if(pFileSel != NULL){
    fclose(pFileSel);
    pFileSel = NULL;
  }

  return 0;
}

uint64_t TickCount::GetTimeSlice(std::string sKey, int nTime) {
  if (m_timeSlice.find(sKey) == m_timeSlice.end()) {
    return 0;
  }

  if (m_timeSlice[sKey].empty()) {
    return 0;
  }

  int nCount = m_timeSlice[sKey].size();

  if (nTime < 0) {
    nTime = nCount + nTime;
  }

  if (nTime <= 0) {
    return m_timeSlice[sKey].at(0);
  }

  if (nTime >= nCount) {
    return m_timeSlice[sKey].at(nCount - 1);
  }

  return m_timeSlice[sKey].at(nTime);
}

int TickCount::Begin(std::string sKey) {
  if (m_startTime.find(sKey) == m_startTime.end()) {
    m_startTime[sKey] = GetMicroTickCountU64();
    return 0;
  }
  return -1;
}

int TickCount::End(std::string sKey, bool bShow) {
  if (m_startTime.find(sKey) == m_startTime.end()) {
    return -1;
  }

  uint64_t u64Time = GetMicroTickCountU64();

  uint64_t u64TimeCost = u64Time - m_startTime[sKey];
  for (std::map<std::string, uint64_t>::iterator it = m_startTime.begin();
       it != m_startTime.end(); ++it) {
    if (it->first == sKey) {
      m_startTime.erase(it);
      break;
    }
  }

  AppentTimeSlice(sKey, u64TimeCost);

  if (bShow) {
    std::cout << sKey.c_str() << "cost"
             << static_cast<double>(u64TimeCost) / kMicroToSec << "s\n";
  }

  return 0;
}

AutoTickCount::AutoTickCount(TickCount &t, std::string sKey, bool bShow) {
  m_pT = &t;
  m_u64StartTime = m_u64EndTime = GetMicroTickCountU64();
  m_sKey = sKey;
  m_bShow = bShow;
}

AutoTickCount::~AutoTickCount() {
  m_u64EndTime = GetMicroTickCountU64();

  if (m_sKey.length() > 0) {
    m_pT->AppentTimeSlice(m_sKey, m_u64EndTime - m_u64StartTime);
  }

  if (m_bShow) {
    std::cout << m_sKey.c_str() << "cost"
             << static_cast<double>(m_u64EndTime - m_u64StartTime) /
                    TickCount::kMicroToSec
             << "s\n";
  }
}
