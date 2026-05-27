#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "hs_com.h"
#include "logger.h"
#include "ptc_client.h"
using namespace hesai::lidar;

// #define SET_NET
// #define SET_DES_IP_AND_PORT
// #define SET_RETURN_MODE
// #define SET_SYNC_ANGLE
// #define SET_STANDBY_MODE
// #define SET_SPIN_SPEED

#define DEFINE_YOURSELF

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <device_ip_address> <ptc_port>" << std::endl;
        return -1;
    }
    // Initialize logger
    Logger::GetInstance().setLogTargetRule(LOGTARGET::HESAI_LOG_TARGET_CONSOLE);
    Logger::GetInstance().setLogLevelRule(LOGLEVEL::HESAI_LOG_INFO | LOGLEVEL::HESAI_LOG_WARNING | LOGLEVEL::HESAI_LOG_ERROR | LOGLEVEL::HESAI_LOG_FATAL);

    std::string device_ip_address = argv[1];
    int ptc_port = atoi(argv[2]);

    PtcClient *ptc_client_ = new PtcClient(device_ip_address, ptc_port);

    // must add sleep to wait for the client to be ready                                          
    LogInfo("Waiting for the client to be ready");                                          
    while(ptc_client_->IsOpen() == false) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    LogInfo("Client is ready");   

#ifdef SET_NET
    std::string net_IP = "192.168.1.201";
    std::string net_mask = "255.255.255.0";
    std::string net_getway = "192.168.1.1";
    uint8_t vlan_flag = 0;
    uint16_t vlan_ID = 0;
    if(ptc_client_->SetNet(net_IP, net_mask, net_getway, vlan_flag, vlan_ID) == false) {
        LogError("SetNet failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetNet succeeded");
    }
#endif

#ifdef SET_DES_IP_AND_PORT
    std::string destination_ip = "255.255.255.255";
    uint16_t udp_port = 2368;
    uint16_t gps_udp_port = 10110;
    if(ptc_client_->SetDesIpandPort(destination_ip, udp_port, gps_udp_port) == false) {
        LogError("SetDesIpandPort failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetDesIpandPort succeeded");
    }
#endif

#ifdef SET_RETURN_MODE
    uint8_t return_mode = 1;
    if(ptc_client_->SetReturnMode(return_mode) == false) {
        LogError("SetReturnMode failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetReturnMode succeeded");
    }
#endif

#ifdef SET_SYNC_ANGLE
    uint8_t enable_sync_angle = 1;
    uint16_t sync_angle = 20;
    if(ptc_client_->SetSyncAngle(enable_sync_angle, sync_angle) == false) {
        LogError("SetSyncAngle failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetSyncAngle succeeded");
    }
#endif

#ifdef SET_STANDBY_MODE
    uint32_t standby_mode = 0;
    if(ptc_client_->SetStandbyMode(standby_mode) == false) {
        LogError("SetStandbyMode failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetStandbyMode succeeded");
    }
#endif

#ifdef SET_SPIN_SPEED
    uint32_t speed = 1200;
    if(ptc_client_->SetSpinSpeed(speed) == false) {
        LogError("SetSpinSpeed failed! Please make sure your input is valid, return_code: %d", ptc_client_->ret_code_);
    }
    else {
        LogInfo("SetSpinSpeed succeeded");
    }
#endif

#ifdef DEFINE_YOURSELF
    u8Array_t dataIn;       // load data, filled in as command, for example, is empty when you want to get the correction file
    u8Array_t dataOut;      // response load data, for example, is correction file data when you want to get the correction file
    uint8_t ptc_cmd = 0x05; // ptc command, for example, 0x05 is GetLidarCalibration

    int ret = -1;
    ret = ptc_client_->QueryCommand(dataIn, dataOut, ptc_cmd);
    if (ret == 0) {
        LogInfo("Define yourself succeeded");
    } else {
        LogWarning("Define yourself failed! return_code: %d", ptc_client_->ret_code_); // if ret_code_ is a positive number, it represents an error code returned by PTC, or if it is a negative number, it indicates some unexpected errors. Please refer to the code for details.
    }
#endif

    delete ptc_client_;
    return 0;
}