# 发送PTC指令 
本文档介绍了如何通过SDK向雷达发送PTC指令，获取雷达的信息或者设置雷达参数

## 准备
进入 [ptc_tool.cc](../tool_ptc/ptc_tool.cc) 

选择需要的通讯指令

```cpp
// #define SET_NET
// #define SET_DES_IP_AND_PORT
// #define SET_RETURN_MODE
// #define SET_SYNC_ANGLE
// #define SET_STANDBY_MODE
// #define SET_SPIN_SPEED

#define DEFINE_YOURSELF
```
1. **`SET_NET`** ： 设置雷达配置信息
2. **`SET_DES_IP_AND_PORT`** ： 设置雷达点云目的信息
3. **`SET_RETURN_MODE`** ： 设置雷达回波模式
4. **`SET_SYNC_ANGLE`** ： 设置雷达同步角
5. **`SET_STANDBY_MODE`** ： 设置雷达工作模式
6. **`SET_SPIN_SPEED`** ： 设置雷达转速信息
7. **`DEFINE_YOURSELF`** ： 自定义命令

## 操作
### 1 编译
在HesaiLidar_SDK_2.0文件夹下，启动Terminal终端，执行以下指令。
```bash
cd HesaiLidar_SDK_2.0/tool_ptc
mkdir build
cd build
cmake ..
make
```

### 2 运行
成功编译后，在`build`文件夹下运行生成的`ptc_tool`可执行文件。

执行时需要添加 **雷达的IP地址** 和 **PTC通讯端口**

```bash
./ptc_tool 192.168.1.201 9347
```

## 更多参考
以下介绍几个常见的示例程序。
#### 1 设置雷达目的IP
设置目的IP和相关端口号
```cpp
std::string destination_ip = "255.255.255.255";  //可以按需设置为单播、组播、广播
uint16_t udp_port = 2368;  //设置udp端口
uint16_t gps_udp_port = 10110;  //设置gps端口
```
运行后终端有如下相关打印代表设置参数成功
```log
SetDesIpandPort succeeded!
```

#### 2 设置雷达IP
设置IP、子网掩码、网关和VLAN
```cpp
std::string net_IP = "192.168.1.201";  //设置雷达IP
std::string net_mask = "255.255.255.0";  //设置子网掩码
std::string net_getway = "192.168.1.1";  //设置网关
uint8_t vlan_flag = 0;  //设置VLAN标志，0表示不设置，1表示设置
uint16_t vlan_ID = 0;  //设置VLAN ID
```
运行后终端有如下相关打印代表设置参数成功
```log
SetNet succeeded!
```
**注意：设置雷达IP后运行会终止，需要重新设置网络配置为雷达对应的IP后，再继续进行其它操作**

#### 3 获取雷达角度校准文件
在[ptc_tool.cc](../tool_ptc/ptc_tool.cc)中使用 `DEFINE_YOURSELF`, 添加保存逻辑，会在build目录下生成correction文件
```cpp
#ifdef DEFINE_YOURSELF
    u8Array_t dataIn;       
    u8Array_t dataOut;      
    uint8_t ptc_cmd = 0x05;

    int ret = -1;
    ret = ptc_client_->QueryCommand(dataIn, dataOut, ptc_cmd);
    if (ret == 0) {
        LogInfo("GetCorrectionInfo succeeded!");
        // 如果你想保存到文件，比如保存为correction.dat
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

#### 4 根据PTC协议添加需要的指令
在[ptc_tool.cc](../tool_ptc/ptc_tool.cc)中使用 `DEFINE_YOURSELF`

```cpp
#ifdef DEFINE_YOURSELF
    u8Array_t dataIn;       // 负载数据，为PTC协议定义的data数据部分，如果为扩展命令也包含扩展命令本身
    u8Array_t dataOut;      // 响应结果，不包含头部信息，只包含负载数据
    uint8_t ptc_cmd = 0x05; // PTC命令，如果为扩展命令，则统一为0xFF

    int ret = -1;
    ret = ptc_client_->QueryCommand(dataIn, dataOut, ptc_cmd);
    if (ret == 0) {
        LogInfo("Define yourself succeeded");
    } else {
        LogWarning("Define yourself failed! return_code: %d", ptc_client_->ret_code_); // ret_code_如果为正数，则为PTC返回的错误码，如果为负数，则是一些意外的错误，详见代码
    }
#endif

```