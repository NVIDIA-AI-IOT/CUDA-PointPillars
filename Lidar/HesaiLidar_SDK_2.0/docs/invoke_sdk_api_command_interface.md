# Sending PTC Commands 
This document describes how to send PTC commands to the LiDAR through the SDK in order to retrieve LiDAR information or configure its parameters.

## Preparation
Open the file [ptc_tool.cc](../tool_ptc/ptc_tool.cc).

Select the desired communication command:

```cpp
// #define SET_NET
// #define SET_DES_IP_AND_PORT
// #define SET_RETURN_MODE
// #define SET_SYNC_ANGLE
// #define SET_STANDBY_MODE
// #define SET_SPIN_SPEED

#define DEFINE_YOURSELF
```
1. **`SET_NET`** : Set LiDAR network configuration
2. **`SET_DES_IP_AND_PORT`** : Set destination IP and port for point cloud data
3. **`SET_RETURN_MODE`** : Set return mode of the LiDAR
4. **`SET_SYNC_ANGLE`** : Set sync angle 
5. **`SET_STANDBY_MODE`** : Set operating mode of the LiDAR
6. **`SET_SPIN_SPEED`** : Set spin speed of the LiDAR
7. **`DEFINE_YOURSELF`** : Custom command

## Steps
### 1 Compilation
In the `HesaiLidar_SDK_2.0` folder, open a terminal and execute the following commands:
```bash
cd HesaiLidar_SDK_2.0/tool_ptc
mkdir build
cd build
cmake ..
make
```

### 2 Run
After successful compilation, run the generated executable file `ptc_tool` inside the `build` directory.

When executing, you need to provide the **LiDAR's IP address** and the **PTC communication port**:

```bash
./ptc_tool 192.168.1.201 9347
```

## Reference

#### 1 Setting Destination IP
Set the destination IP address and related ports:
```cpp
std::string destination_ip = "255.255.255.255";  // Set to unicast, multicast, or broadcast
uint16_t udp_port = 2368;  // Set UDP port
uint16_t gps_udp_port = 10110;  // Set GPS UDP port
```
The following log indicates that the parameters were successfully set:
```log
SetDesIpandPort succeeded!
```

#### 2 Setting Network Configuration
Configure IP address, subnet mask, gateway, and VLAN:
```cpp
std::string net_IP = "192.168.1.201";  // Set LiDAR IP
std::string net_mask = "255.255.255.0";  // Set subnet mask
std::string net_getway = "192.168.1.1";  // Set gateway
uint8_t vlan_flag = 0;  // VLAN flag, 0 means not set, 1 means set
uint16_t vlan_ID = 0;  // Set VLAN ID
```
The following log indicates that the parameters were successfully set:
```log
SetNet succeeded!
```
> Note: Running this command may terminate the program. You will need to reconfigure your network with the new LiDAR IP before proceeding with further operations.

#### 3 Getting Correction File
In [ptc_tool.cc](../tool_ptc/ptc_tool.cc), use the `DEFINE_YOURSELF` macro and add saving logic. This will generate a calibration file named `correction.dat` in the `build` directory:
```cpp
#ifdef DEFINE_YOURSELF
    u8Array_t dataIn;       
    u8Array_t dataOut;      
    uint8_t ptc_cmd = 0x05;

    int ret = -1;
    ret = ptc_client_->QueryCommand(dataIn, dataOut, ptc_cmd);
    if (ret == 0) {
        LogInfo("GetCorrectionInfo succeeded!");
        // Save to a file, e.g., correction.dat
        FILE* fp = fopen("correction.dat", "wb");
        if (fp != nullptr) {
            fwrite(dataOut.data(), 1, dataOut.size(), fp);
            fclose(fp);
            LogInfo("Saved correction data to correction");
        } else {
            LogInfo("Failed to open file for writing correction data!");
        }
    } else {
        LogWarning("GetCorrectionInfo failed!");
    }
#endif
```

#### 4 Adding Custom Commands Based on PTC Protocol
In [ptc_tool.cc](../tool_ptc/ptc_tool.cc), define and use `DEFINE_YOURSELF`

```cpp
#ifdef DEFINE_YOURSELF
    u8Array_t dataIn;       // Payload data, as defined by the PTC protocol. For extended commands, it also includes the extended command itself.
    u8Array_t dataOut;      // Response data, excluding header information, only contains payload data.
    uint8_t ptc_cmd = 0x05; // PTC command. For extended commands, it should be set to 0xFF uniformly.

    int ret = -1;
    ret = ptc_client_->QueryCommand(dataIn, dataOut, ptc_cmd);
    if (ret == 0) {
        LogInfo("Define yourself succeeded");
    } else {
        LogWarning("Define yourself failed! return_code: %d", ptc_client_->ret_code_); // If ret_code_ is positive, it represents an error code returned by PTC. If negative, it indicates some unexpected error. Refer to the code for details.
    }
#endif

```