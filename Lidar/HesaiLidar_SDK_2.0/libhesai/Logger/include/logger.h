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
#ifndef __LOGGER_H__  
#define __LOGGER_H__  
  
#include <string>  
#include <memory>  
#include <thread>  
#include <mutex>  
#include <condition_variable>  
#include <list>  
#include <functional>
#include <algorithm>
#include <iostream>

#ifdef _MSC_VER
  #ifdef LOG_LIB_API_EXPORTS
    #define LOG_LIB_API __declspec(dllexport)
  #else
    #define LOG_LIB_API __declspec(dllimport)
  #endif
#else 
  #define LOG_LIB_API
#endif

enum LOGLEVEL
{
	HESAI_LOG_DEBUG   =  0x01,      /*debug*/
  HESAI_LOG_INFO    =  0x02,      /*info*/
  HESAI_LOG_WARNING =  0x04,      /*warning*/
  HESAI_LOG_ERROR   =  0x08,      /*error*/
  HESAI_LOG_FATAL   =  0x10,      /*assert*/
};

enum LOGTARGET
{
	HESAI_LOG_TARGET_NONE      = 0x00,
	HESAI_LOG_TARGET_CONSOLE   = 0x01,
	HESAI_LOG_TARGET_FILE      = 0x10
};

#ifdef _MSC_VER
#define __FUNCTION_NAME__ __FUNCSIG__
#else
#define __FUNCTION_NAME__ __PRETTY_FUNCTION__
#endif
#define LogDebug(...)        Logger::GetInstance().AddToQueue(HESAI_LOG_DEBUG, __FILE__, __LINE__, __FUNCTION_NAME__, __VA_ARGS__)  
#define LogInfo(...)        Logger::GetInstance().AddToQueue(HESAI_LOG_INFO, __FILE__, __LINE__, __FUNCTION_NAME__, __VA_ARGS__)  
#define LogWarning(...)     Logger::GetInstance().AddToQueue(HESAI_LOG_WARNING, __FILE__, __LINE__, __FUNCTION_NAME__, __VA_ARGS__)  
#define LogError(...)       Logger::GetInstance().AddToQueue(HESAI_LOG_ERROR, __FILE__, __LINE__, __FUNCTION_NAME__, __VA_ARGS__)
#define LogFatal(...)       Logger::GetInstance().AddToQueue(HESAI_LOG_FATAL, __FILE__, __LINE__, __FUNCTION_NAME__, __VA_ARGS__)    

class LOG_LIB_API Logger  
{  
public:  
  static Logger& GetInstance();  

  void SetFileName(const char* filename);  

  bool Start();  
  void Stop();  
  
  void AddToQueue(LOGLEVEL loglevel, const char* pszFile, int lineNo, const char* pszFuncSig, const char* pszFmt, ...); 
  void setLogLevelRule(uint8_t rule);
  void setLogTargetRule(uint8_t rule);
  void bindLogCallback(std::function<void(LOGLEVEL loglevel, const char* pszFile, int lineNo, const char* pszFuncSig, char* pszFmt)> log_callback);
private:  
    Logger() {}
    Logger(const Logger& rhs) = delete;  
    Logger& operator =(Logger& rhs) = delete;  
  
    void threadfunc();  
  
  
private:  
    std::string                     filename_;  
    FILE*                           fp_{NULL};  
    std::shared_ptr<std::thread>    spthread_;  
    std::mutex                      mutex_;  
    std::mutex                      mutex_running;  
    std::condition_variable         cv_; 
    std::list<std::string>          queue_;  
    std::function<void(LOGLEVEL loglevel, const char* pszFile, int lineNo, const char* pszFuncSig, char* pszFmt)> log_callback_;
    bool                            running_{false};
    bool                            exit_{false}; 
    uint8_t                         log_level_rule_;
    uint8_t                         log_target_rule_;
};  

#endif