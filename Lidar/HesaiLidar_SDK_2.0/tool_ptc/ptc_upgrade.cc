#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "hs_com.h"
#include "logger.h"
#include "ptc_client.h"
#include <fstream>
#include <vector>
using namespace hesai::lidar;

/********************************************************************************
    if you want to use the upgrade process function, please uncomment the following code
    and implement the upgradeProcessFunction in your code. 
    And default is to use the default upgrade process function.
********************************************************************************/
// void* upgradeProcessFunction(void* threadArgs) {
//     UpgradeProgress* progressInfo  = (UpgradeProgress*)threadArgs;
//     float progress = 0.0f;

//     while(progressInfo->status == 0 && progressInfo->current_packet < progressInfo->total_packets)
//     {
//         progress = progressInfo->current_packet / (float)progressInfo->total_packets;
//         std::cout << "total_packets: " << progressInfo->total_packets << std::endl;
//         std::cout << "current_packet: " << progressInfo->current_packet << std::endl;
//         std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // sleep for 1s
//     }
//     std::cout << "total_packets: " << progressInfo->total_packets << std::endl;
//     std::cout << "current_packet: " << progressInfo->current_packet << std::endl;
//     std::cout << "status: " << progressInfo->status << std::endl;
//     return NULL;
// }


int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <device_ip_address> <ptc_port> <upgrade patch file path>" << std::endl;
        return -1;
    }
    // Initialize logger
    Logger::GetInstance().setLogTargetRule(LOGTARGET::HESAI_LOG_TARGET_CONSOLE);
    Logger::GetInstance().setLogLevelRule(LOGLEVEL::HESAI_LOG_INFO | LOGLEVEL::HESAI_LOG_WARNING | LOGLEVEL::HESAI_LOG_ERROR | LOGLEVEL::HESAI_LOG_FATAL);

    std::string device_ip_address = argv[1];
    int ptc_port = atoi(argv[2]);
    const std::string file_path = argv[3]; 

    PtcClient *ptc_client_ = new PtcClient(device_ip_address, ptc_port);

    /********************************************************************************
    if you want to use the upgrade process function, please uncomment the following code
    ********************************************************************************/
    // ptc_client_->RegisterUpgradeProcessFunc(upgradeProcessFunction);

    // must add sleep to wait for the client to be ready
    LogInfo("Waiting for the client to be ready");                                          
    while(ptc_client_->IsOpen() == false) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    LogInfo("Client is ready");

    int ret = ptc_client_->UpgradeLidarPatch(file_path, kPTCUpgradeLidar, 0);
    if (ret != 0) {
        return -1;
    }
    else {
        if (ptc_client_->RebootLidar() == true) {
            LogInfo("RebootLidar successful\n");
        }
        else {
            LogError("RebootLidar failed\n");
            return -1;
        }
    }
    // Process the content as needed
    delete ptc_client_;
    return 0;
}