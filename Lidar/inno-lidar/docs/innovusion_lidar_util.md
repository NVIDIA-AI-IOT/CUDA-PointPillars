# get_pcd
## A. What is innovusion_lidar_util
```shell
innovusion_lidar_util mainly provides the function of modifying and obtaining lidar related parameters.
```

## B. How to use innovusion_lidar_util
```shell
# Get detailed version information of SW/FW for radar
# ./innovusion_lidar_util <ip of LIDAR> get_version
./innovusion_lidar_util 172.168.1.10 get_version

# Download raw data
# ./innovusion_lidar_util <ip of LIDAR> start <file_name> <capture_size_in_MB>
./innovusion_lidar_util 172.168.1.10 start ./rawdata 12

# Modify lidar IP
# ./innovusion_lidar_util <ip of LIDAR> set_ip_address <new_ip_address>
./innovusion_lidar_util 172.168.1.10 set_ip_address 172.168.1.11
# soft reboot to enable new IP
#./innovusion_lidar_util <old ip address> soft_reboot
./innovusion_lidar_util 172.168.1.10 soft_reboot

# Modify network information
#./innovusion_lidar_util <ip of LIDAR> set_network <new_ip_address> <new_netmask_address> <new_gateway_address>
./innovusion_lidar_util 172.168.1.10 set_network 172.168.1.10 255.255.0.0 172.168.1.1

# View radar network information
# ./innovusion_lidar_util <ip of LIDAR> get_network
./innovusion_lidar_util 172.168.1.10 get_network

# view running time
# ./innovusion_lidar_util <ip of LIDAR> get_uptime
./innovusion_lidar_util 172.168.1.10 get_uptime

# Get frame rate
# ./innovusion_lidar_util <ip of LIDAR> get_framerate
./innovusion_lidar_util 172.168.1.10 get_framerate

# soft reboot
# ./innovusion_lidar_util <ip of LIDAR> soft_reboot
./innovusion_lidar_util 172.168.1.10 soft_reboot

# Get register parameters
# ./innovusion_lidar_util <ip of LIDAR>get_registers
./innovusion_lidar_util 172.168.1.10 get_registers

# Get temperature information
# ./innovusion_lidar_util <ip of LIDAR> get_temperature
./innovusion_lidar_util 172.168.1.10 get_temperature

# Get detector temperature information
# ./innovusion_lidar_util <ip of LIDAR> get_detector_temps
./innovusion_lidar_util 172.168.1.10 get_detector_temps

# Get polygon speed
# ./innovusion_lidar_util <ip of LIDAR> get_motor_speeds
./innovusion_lidar_util 172.168.1.10 get_motor_speeds

# Display all firmware log information
#./innovusion_lidar_util <ip of LIDAR> get_log
./innovusion_lidar_util 172.168.1.10 get_log

# Download firmware logs for radar
#./innovusion_lidar_util <ip of LIDAR> get_log_file <filename>
./innovusion_lidar_util 172.168.1.10 get_log_file log

# Get lidar sequence number
# ./innovusion_lidar_util <ip of LIDAR> get_sn
./innovusion_lidar_util 172.168.1.10 get_sn

# Get lidar status information
#./innovusion_lidar_util <ip of LIDAR> get_status
./innovusion_lidar_util 172.168.1.10 get_status


# Get lidar model information
#./innovusion_lidar_util <ip of LIDAR> get_model
./innovusion_lidar_util 172.168.1.10 get_model

# Download yaml file
# ./innovusion_lidar_util <ip of LIDAR> download_cal_file <file_id> <filename>
./innovusion_lidar_util 172.168.1.10 download_cal_file 1 test.yaml

# Upload yaml file
#./innovusion_lidar_util <ip of LIDAR> upload_cal_file <file_id> <filename>
./innovusion_lidar_util 172.168.1.10 upload_cal_file 1 test.yaml

# Set the internal parameters of the lidar
#./innovusion_lidar_util <ip of LIDAR>set_config <section> <value> <…>
./innovusion_lidar_util 172.168.1.10 set_config time ntp_en 1

# Get the internal parameters of the lidar
#./innovusion_lidar_util <ip of LIDAR> get_config [section [key]]
./innovusion_lidar_util 172.168.1.10 get_config

# View the modified internal parameters of lidar
#./innovusion_lidar_util <ip of LIDAR>get_config_changed [section [key]]
./innovusion_lidar_util 172.168.1.10 get_config_changed

# Get raw raw data
#./innovusion_lidar_util <ip of LIDAR> raw_capture <unit> <channel> <capture_size> <0|1>
./innovusion_lidar_util 172.168.1.10 raw_capture 1 5 0

# Stop lidar internal point cloud service
#./innovusion_lidar_util <ip of LIDAR> stop
./innovusion_lidar_util 172.168.1.10 stop

# Set the vertical position of the ROI area center
#./innovusion_lidar_util <ip of LIDAR> set_vertical_roi <degree>
./innovusion_lidar_util 172.168.1.10 set_vertical_roi 22

# Get state information of Lidar
#./innovusion_lidar_util <ip of LIDAR> get_state
./innovusion_lidar_util 172.168.1.10 get_state

# PTP TEST
#./innovusion_lidar_util <ip of LIDAR> ptp_diag [<status> | <trace> <period> <interval> | <stop> | <show>]
./innovusion_lidar_util 172.168.1.10 ptp_diag status

# download PCS/PTP configuration file
#./innovusion_lidar_util <ip of LIDAR> download_internal_file <file_id> <filename>
./innovusion_lidar_util 172.168.1.10 download_internal_file PCS_ENV test.yaml

# upload PCS/PTP configuration file
#./innovusion_lidar_util <ip of LIDAR> upload_internal_file <file_id> <filename>
./innovusion_lidar_util 172.168.1.10 upload_internal_file PCS_ENV test.yaml
```

## C. Location
If you want to get detial of innovusion_lidar_util, please press [here](../apps/tools/lidar_util/lidar_util.cpp)
