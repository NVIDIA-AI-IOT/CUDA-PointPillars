# Enabling and Disabling Compilation Macros
The following is the configuration method for controlling the activation status of compilation macros (using the `WITH_PTCS_USE` macro as an example).

## Ubuntu System Configuration
### Method 1：Command Line Configuration
1. **Enter the build directory**：  
    ```bash
    cd build
    ```
2. **Run the CMake command**：  
    ```bash
    cmake .. -DWITH_PTCS_USE=true  # Enable macro
    ```
    or
    ```bash
    cmake .. -DWITH_PTCS_USE=false  # Disable macro
    ```
3. **Error Handling**：
    If the configuration fails, please clear the build directory and try again:
    ```bash
    rm -rf ./*  # Warning: This operation will permanently delete all files in the build directory
    ```
    Then re-run the CMake command.

### Method 2：CMake GUI Tool
1. **Install the tool**： 
    ```bash
    sudo apt install cmake-gui
    ```
2. **Run the tool**：
    ```bash
    cmake-gui
    ```
3. **Set project paths**：
    - Source path: Select the `HesaiLidar_SDK_2.0` directory
    - Build path: Select the `HesaiLidar_SDK_2.0/build` directory
4. **Configure macro status**：
    Find the `WITH_PTCS_USE`option in the interface and set its value to enable or disable (usually located under the `WITH` section):
    - ✅ Checked: Enable macro
    - ⬜ Unchecked: Disable macro
5. **Manually add the macro (optional)**：
    If this option is not found, click `Add Entry` to add it:
    - Key：`WITH_PTCS_USE`
    - Type：`BOOL`
6. **Generate project**：
    Click the `Configure` → `Generate` buttons in sequence to generate the configuration project.
 
## Windows System Configuration
Refer to **[How to Compile the SDK on Windows](../docs/compile_on_windows.md)** specifically the section`2 CMake Configuration`, for configuring using the `CMake GUI` tool.
1. **In the parameter list, find the `WITH_PTCS_USE` option**：
    - ✅ Checked: Enable macro
    - ⬜ Unchecked: Disable macro
2. **Manually add the macro (optional)**：
    If this option is not found, you can add it under `Add Entry`:
    - Key：`WITH_PTCS_USE`
    - Type：`BOOL`