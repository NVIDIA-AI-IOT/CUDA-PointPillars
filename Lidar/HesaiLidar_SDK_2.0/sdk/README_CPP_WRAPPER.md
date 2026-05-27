# Hesai SDK C++ Wrapper

这是一个纯 C++ 接口封装，去除了 Python 依赖，用于在 C++ 项目中直接使用 Hesai 激光雷达 SDK。

## 文件说明

- **hsai_sdk_wrapper.h**: 头文件，定义接口
- **hsai_sdk_wrapper.cc**: 实现文件
- **example_usage.cpp**: 使用示例

## 编译方法

### 方法 1: 手动编译

```bash
cd HesaiLidar_SDK_2.0/build

# 编译库文件
g++ -c -o hsai_sdk_wrapper.o ../sdk/hsai_sdk_wrapper.cc \
    -std=c++14 -O3 -Wall \
    -I../driver -I../libhesai \
    -I../

# 链接成库
ar rcs libhesai_sdk_wrapper.a hsai_sdk_wrapper.o

# 编译示例程序
g++ -o example_usage ../sdk/example_usage.cpp hsai_sdk_wrapper.o \
    -std=c++14 -O3 -Wall \
    -I../driver -I../libhesai -I../ \
    -L. -lhesai_sdk_lib -pthread
```

### 方法 2: 使用 CMake（推荐）

在主 CMakeLists.txt 中添加：

```cmake
# 编译 wrapper 库
add_library(hesai_sdk_wrapper
    sdk/hsai_sdk_wrapper.cc
)
target_include_directories(hesai_sdk_wrapper PUBLIC 
    driver
    libhesai
    ./
)
target_link_libraries(hesai_sdk_wrapper PRIVATE hesai_sdk_lib)

# 编译示例程序
add_executable(example_usage sdk/example_usage.cpp)
target_link_libraries(example_usage PRIVATE hesai_sdk_wrapper)
```

然后在 build 目录下：
```bash
cd HesaiLidar_SDK_2.0/build
cmake ..
make
```

## 使用方法

### 基本用法

```cpp
#include "hsai_sdk_wrapper.h"
#include <iostream>

using namespace HesaiSDK;

int main() {
    // 1. 初始化 SDK
    std::string lidar_ip = "192.168.1.201";
    InitHesaiSDK(lidar_ip, 0.0f, 0.0f, 0.0f);
    
    // 2. 循环获取点云数据
    for (int i = 0; i < 100; ++i) {
        PointCloudResult result;
        
        if (GetPointCloudData(result)) {
            std::cout << "Got " << result.num_points << " points" << std::endl;
            
            // 访问点坐标
            // result.points 是扁平数组：[x0, y0, z0, x1, y1, z1, ...]
            for (size_t j = 0; j < result.num_points; ++j) {
                float x = result.points[j * 3];
                float y = result.points[j * 3 + 1];
                float z = result.points[j * 3 + 2];
                uint8_t intensity = result.reflectivity[j];
                
                // 处理点...
            }
        }
        
        usleep(100000); // 100ms
    }
    
    // 3. 清理
    UninitHesaiSDK();
    return 0;
}
```

### 数据结构

#### PointCloudResult

```cpp
struct PointCloudResult {
    std::vector<float> points;        // xyz 坐标 (size = num_points * 3)
    std::vector<uint8_t> reflectivity; // 反射强度 (size = num_points)
    size_t num_points;                 // 点的数量
};
```

**访问点的坐标**：
```cpp
// 第 i 个点的坐标
size_t idx = i * 3;
float x = result.points[idx];
float y = result.points[idx + 1];
float z = result.points[idx + 2];
uint8_t intensity = result.reflectivity[i];
```

### API 说明

#### InitHesaiSDK
```cpp
int InitHesaiSDK(const std::string& lidar_ip, float roll, float pitch, float yaw);
```
- **功能**: 初始化 SDK，在后台线程中启动
- **参数**:
  - `lidar_ip`: 激光雷达 IP 地址
  - `roll`: 翻滚角（弧度）
  - `pitch`: 俯仰角（弧度）
  - `yaw`: 偏航角（弧度）
- **返回**: 0 成功，-1 失败

#### UninitHesaiSDK
```cpp
void UninitHesaiSDK();
```
- **功能**: 停止 SDK 并清理资源

#### GetPointCloudData
```cpp
bool GetPointCloudData(PointCloudResult& result);
```
- **功能**: 获取点云数据
- **参数**: `result` - 输出参数，存储点云数据
- **返回**: `true` 表示有数据，`false` 表示无数据

#### IsSDKRunning
```cpp
bool IsSDKRunning();
```
- **功能**: 检查 SDK 是否正在运行

## 与 Python 版本的对比

| 特性 | Python 版本 | C++ 版本 |
|------|------------|----------|
| 依赖 | pybind11 | 无 |
| 性能 | 较慢（需要类型转换） | 快 |
| 易用性 | 更方便（支持 numpy） | 需要手动管理数据 |
| 适用场景 | 原型开发、Python 项目 | 生产环境、C++ 项目 |

## 注意事项

1. **线程安全**: `GetPointCloudData` 是线程安全的
2. **数据格式**: 点坐标是扁平数组 `[x0, y0, z0, x1, y1, z1, ...]`
3. **生命周期**: 调用 `UninitHesaiSDK` 后，`result` 中的数据仍然有效
4. **频率**: 建议每 100ms 调用一次 `GetPointCloudData`

## 与 PointPillars 集成

可以将获取的点云数据直接传给 PointPillars 推理：

```cpp
// 获取点云
PointCloudResult pc_result;
GetPointCloudData(pc_result);

// 转换为 PointPillars 需要的格式
// 假设需要将数据拷贝到 GPU
cudaMemcpy(gpu_points, pc_result.points.data(), 
           pc_result.num_points * 3 * sizeof(float), 
           cudaMemcpyHostToDevice);

// 调用 PointPillars 推理
auto bboxes = core->forward((float *)gpu_points, pc_result.num_points, stream);
```

## 配置说明

修改 `hsai_sdk_wrapper.cc` 中的以下路径以适应你的环境：

```cpp
param.input_param.correction_file_path = "/path/to/your/correction/file.dat";
param.input_param.firetimes_path = "/path/to/your/firetime/file.csv";
```

## 故障排查

1. **连接失败**: 检查 IP 地址和网络
2. **无数据返回**: 确保激光雷达正在发送数据
3. **编译错误**: 确保安装了 Hesai SDK 的所有依赖

## 许可证

与 HesaiLidar SDK 保持一致

