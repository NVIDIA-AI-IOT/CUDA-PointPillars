# Compile on Windows

## 1 Dependency Installation

**Essential Components**:
1. **[Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)**  
    
    During installation, select:

   - Desktop development with C++
   - Windows 10/11 SDK (latest version)
   - C++ CMake tools

   > For Visual Studio 2019, install "Windows 10 SDK" (v10.0.18362.0 or later).

2. **[CMake 3.25+](https://cmake.org/download/)**
   
   ```powershell
   # Verify installation
   cmake --version
   ```

3. **[Git](https://git-scm.com/)**  

   ```powershell
   # Recommended configuration(Configuring the line ending conversions during installation)
   git config --global core.autocrlf true
   ```
   ```powershell
   # Verify installation
   git --version
   ```

4. To enable PTCS (PTC over TLS/mTLS) communication with the lidar, install **[OpenSSL 1.1.1](https://slproweb.com/products/Win32OpenSSL.html)**. (Optional)

   > If you do not use PTCS, please refer to **[Compile Macro Control](../docs/compile_macro_control_description.md)** for the operation to disable the macro `WITH_PTCS_USE`.

5. To visualize point cloud data, install **[PCL 1.12.1](https://github.com/PointCloudLibrary/pcl/releases/tag/pcl-1.12.1)**. (Optional)
   
   > Download the AllInOne executable.


## 2 CMake Configuration

1. Launch the CMake GUI.
2. Specify source and build directories.

   - **Where is the source code**: `path_to_your_project/HesaiLidar_SDK_2.0`

      (e.g., `D:/HesaiLidar_SDK_2.0`)
   
   - **Where to build the binaries**: `path_to_your_project/HesaiLidar_SDK_2.0/build`
   
      (e.g., `D:/HesaiLidar_SDK_2.0/build`)

3. Click **Configure** > Select **Visual Studio 17 2022** as the generator > **Finish**.

4. To enable PTCS (PTC over TLS) communication with the lidar, configure `OPENSSL_ROOT_DIR`. (Optional)

   1. Check the **Advanced** checkbox on the upper right of the main CMake GUI window.
   2. Scroll down the list of advanced variables to find `OPENSSL_ROOT_DIR`.
   3. Double-click its **Value** field and enter:

      ```cmake
      C:\Program Files\OpenSSL-Win64
      ```

   4. Click **Configure** again to let CMake detect OpenSSL using the new path.

5. Once configuration finishes without errors, click **Generate**.

6. If errors appear, verify the OpenSSL path or other missing dependencies. Otherwise, open the generated solution in Visual Studio.


## 3 Visual Studio Compilation

1. Select Configuration & Platform.

   - **Solution Configuration:** Debug
   - **Solution Platform:** x64

2. Build

   - **Full solution:** Right-click the **solution** → **Build Solution**.
   - **Single project:** Right-click **sample** → **Build**.

3. Verify Executable

   ```powershell
   cd build/Debug
   .\sample.exe --version
   ```

   **Output of a successful build**:

   ```log
   - Output Window: "========== Build: 1 succeeded, 0 failed =========="
   - Executable Location: 
   Debug: build/Debug/sample.exe
   Release: build/Release/sample.exe
   ```


## 4 Troubleshooting

Common errors and troubleshooting steps:

- **LNK2005: Duplicate Symbols**:

   1. In the **Solution Explorer**, right-click your project > **Properties**.  
   2. Navigate to **Linker** > **Command Line** > **Additional Options**.  
   3. Add the flag:  

      ```cmake
      /FORCE:MULTIPLE
      ```

- **C4996 Deprecation Warnings**:
  
  1. In the **Solution Explorer**, right-click your project > **Properties**.  
  2. Go to **C/C++** > **Preprocessor** > **Preprocessor Definitions** > **Edit**.  
  3. Add:  

      ```cpp
      _SILENCE_ALL_DEPRECATION_WARNINGS
      ```

- **Windows Path Length Limit**:
  
  1. Press **Win + R**, type `regedit`, and press **Enter** to open the Registry Editor.

  2. Navigate to this path:

     ```reg
     HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
     ```

  3. In the pane on the right, look for a value named `LongPathsEnabled`.

     > If this value does not exist:
     > 1. Right-click in the pane.
     > 2. Select **New** > **DWORD (32-bit) Value**.
     > 3. Name it `LongPathsEnabled`.

  4. Double-click on `LongPathsEnabled` and set its value to `1`.
  5. Click **OK** and close the Registry Editor.
  6. Restart your computer for the changes to take effect.