# 编译宏的启用与禁用控制
以下是控制编译宏启用状态的配置方法（以 `WITH_PTCS_USE` 宏为例）。

## Ubuntu系统配置
### 方式1：命令行配置
1. **进入build目录**：  
    ```bash
    cd build
    ```
2. **执行CMake命令**：  
    ```bash
    cmake .. -DWITH_PTCS_USE=true  # 启用宏
    ```
    或者
    ```bash
    cmake .. -DWITH_PTCS_USE=false  # 禁用宏
    ```
3. **错误处理**：
    若配置失败，请清空build目录后重试：
    ```bash
    rm -rf ./*  # 警告：此操作将永久删除build目录下所有文件
    ```
    然后重新执行CMake命令。

### 方式2：CMake GUI工具
1. **安装工具**： 
    ```bash
    sudo apt install cmake-gui
    ```
2. **运行工具**：
    ```bash
    cmake-gui
    ```
3. **设置工程路径**：
    - 源码路径：选择 `HesaiLidar_SDK_2.0` 目录
    - 构建路径：选择 `HesaiLidar_SDK_2.0/build` 目录
4. **配置宏状态**：
    在界面中找到 `WITH_PTCS_USE`选择，将其Value值勾选为生效，去掉勾选为失效（通常位于`WITH` 栏中）:
    - ✅ 勾选：启用宏
    - ⬜ 取消勾选：禁用宏
5. **手动添加宏（可选）**：
    若未找到该选项，点击 `Add Entry` 添加：
    - Key：`WITH_PTCS_USE`
    - Type：`BOOL`
6. **生成工程**：
    依次点击 `Configure` → `Generate` 按钮，生成配置工程。
 
## Windows系统配置
参考 **[如何在Windows下编译SDK](../docs/compile_on_windows_CN.md)** 中的`2 CMake配置`，使用 `CMake GUI` 工具配置。
1. **在参数列表中找到 WITH_PTCS_USE 选项**：
    - ✅ 勾选：启用宏
    - ⬜ 取消勾选：禁用宏
2. **手动添加宏（可选）**：
    若未找到该选项，可以在 `Add Entry` 中添加：
    - Key：`WITH_PTCS_USE`
    - Type：`BOOL`