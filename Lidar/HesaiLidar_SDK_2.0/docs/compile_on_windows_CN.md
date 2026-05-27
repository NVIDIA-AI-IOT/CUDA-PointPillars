# 如何在Windows下编译SDK

## 1 依赖项安装

**基本组件**:
1. **[Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)**  
   
   在安装时需要选择以下组件:

   - Desktop development with C++
   - Windows 10/11 SDK (latest version)
   - C++ CMake tools

   > 对于 Visual Studio 2019，安装 "Windows 10 SDK" (v10.0.18362.0 或者更高版本).

2. **[CMake 3.25+](https://cmake.org/download/)**

   ```powershell
   # 验证安装
   cmake --version
   ```

3. **[Git](https://git-scm.com/)**
  
    ```powershell
   # 推荐配置（在安装期间对行尾换行转换的配置）
   git config --global core.autocrlf true
   ```
   ```powershell
   # 验证安装
   git --version
   ```

4. 为了使能PTCS (PTC over TLS/mTLS) 和雷达通讯，需要安装 **[OpenSSL 1.1.1](https://slproweb.com/products/Win32OpenSSL.html)**。 (选配)

   > 如果不使用PTCS，请参考 **[编译宏的启用与禁用控制](../docs/compile_macro_control_description_CN.md)** 中的操作，将宏 `WITH_PTCS_USE` 配置为失效即可。

5. 为了可视化点云数据，需要安装 **[PCL 1.12.1](https://github.com/PointCloudLibrary/pcl/releases/tag/pcl-1.12.1)**。 (选配)

   > 下载AllInOne可执行程序。


## 2 CMake配置

1. 运行CMake GUI。

2. 指定源目录和构建目录。

   - **Where is the source code**: `path_to_your_project/HesaiLidar_SDK_2.0`            // 填入对应的源代码路径

      (e.g., `D:/HesaiLidar_SDK_2.0`)
   
   - **Where to build the binaries**: `path_to_your_project/HesaiLidar_SDK_2.0/build`   // 填入构建二进制文件路径
   
      (e.g., `D:/HesaiLidar_SDK_2.0/build`)

3. 点击 **Configure** > 选择 **Visual Studio 17 2022** 作为指定生成器 > **Finish**.

4. 为了使能PTCS (PTC over TLS) 和雷达通讯，需要配置 `OPENSSL_ROOT_DIR` (选配)

   1. 在主CMake图形用户界面窗口的右上角勾选 **Advanced** 复选框。
   2. 向下滚动高级变量列表以找到 `OPENSSL_ROOT_DIR`
   3. 双击其 **Value** 并键入：

      ```cmake
      C:\Program Files\OpenSSL-Win64
      ```

   4. 再次点击 **Configure** 以便让CMake使用新的路径来检测OpenSSL。

5. 一旦配置完成且没有错误，单击 **Generate**。

6. 如果发生错误，检查OpenSSL路径或者其他缺失的依赖项。否则，请在Visual Studio打开生成的解决方案。

## 3 Visual Studio 编译

1. 选择编译配置:
   
   - **Solution Configuration:** Debug
   - **Solution Platform:** x64

2. 构建解决方案:

   - **完整解决方案:** 右键点击 **solution** → **Build Solution**.
   - **单个项目:** 右键点击 **sample** → **Build**.

3. 验证可执行性：

   ```powershell
   cd build/Debug
   .\sample.exe --version
   ```

   **构建成功指示**:

   ```log
   - Output Window: "========== Build: 1 succeeded, 0 failed =========="
   - Executable Location: 
   Debug: build/Debug/sample.exe
   Release: build/Release/sample.exe
   ```

## 4 故障排查

常见报错以及故障排查步骤:

- **LNK2005 Duplicate Symbols**:
   
   1. 在 **Solution Explorer**, 右键点击 project > **Properties**.  
   2. 导航到 **Linker** > **Command Line** > **Additional Options**.  
   3. 添加标志:  

      ```cmake
      /FORCE:MULTIPLE
      ```

- **C4996 Deprecation Warnings**:
 
   1. 在 **Solution Explorer**, 右键点击 project > **Properties**.  
   2. 导航到 **C/C++** > **Preprocessor** > **Preprocessor Definitions** > **Edit**.  
   3. 添加: 

      ```cpp
      _SILENCE_ALL_DEPRECATION_WARNINGS
      ```

- **Windows Path Length Limit**:

   1. 键盘按住 **Win + R**, 键入 `regedit`, 然后左击 **Enter** 来打开注册表编辑器。

   2. 前往此路径:

      ```reg
      HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
      ```

   3. 在右侧窗格中，查找名为 `LongPathsEnabled` 的值。

      > 如果这个值不存在:
      > 1. 在该窗口内右击鼠标。
      > 2. 选择 **New** > **DWORD (32-bit) Value**.
      > 3. 命名为 `LongPathsEnabled`.

   4. 左键双击 `LongPathsEnabled` 并把它的值设置为 `1`。
   5. 点击 **OK** 并且关闭注册表编辑器。
   6. 请重启你的电脑，以便更改生效。