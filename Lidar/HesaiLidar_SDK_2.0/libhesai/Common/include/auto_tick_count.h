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

#ifndef AUTOTIMER_H_
#define AUTOTIMER_H_

#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
namespace hesai
{
namespace lidar
{
class AutoTickCount;

class TickCount {
 public:
  TickCount();
  ~TickCount();

  int Start();
  int Pause();
  std::string GetTimeCost(uint64_t &u64TimeCost);
  std::string GetTimeCost();
  int ShowTimeSlice(std::string sLogFile = "", bool bSaveSysTime = false);
  std::map<std::string, std::vector<uint64_t>> GetTimeSlice() {
    return m_timeSlice;
  }
  uint64_t GetTimeSlice(std::string sKey, int nTime = -1);
  int Begin(std::string sKey);
  int End(std::string sKey, bool bShow = true);
  int SetName(std::string sName) {
    m_sName = sName;

    return 0;
  }

 private:
  static const uint32_t kMicroToSec = 1000000;
  static const uint8_t kClockUnit = 60;
  uint64_t m_u64StartTime;
  uint64_t m_u64EndTime;
  std::map<std::string, std::vector<uint64_t>> m_timeSlice;
  std::map<std::string, uint64_t> m_startTime;
  std::string m_sName;

  friend class AutoTickCount;

  int AppentTimeSlice(std::string sKey, uint64_t u64Time);
};

class AutoTickCount {
 public:
  AutoTickCount(TickCount &t, std::string sKey = "", bool bShow = true);
  ~AutoTickCount();

 private:
  TickCount *m_pT;
  uint64_t m_u64StartTime;
  uint64_t m_u64EndTime;
  bool m_bShow;
  std::string m_sKey;
};
}  // namespace lidar
}  // namespace hesai
#endif  // AUTOTIMER_H_
