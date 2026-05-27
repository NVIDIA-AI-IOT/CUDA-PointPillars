# How to use client sdk

Usage of client sdk is mainly based on [inno_lidar_api.h](../src/sdk_common/inno_lidar_api.h), this docmentation will briefly introduce some API to help you form a simple code block. For more API usage, please refer to [inno_lidar_api.h](../src/sdk_common/inno_lidar_api.h)

**inno_lidar_api.h:**
inno_lidar_set_log_level()
inno_lidar_open_live()
inno_lidar_open_file()
inno_lidar_set_callbacks()
inno_lidar_start()
inno_lidar_stop()
inno_lidar_close()

## step 1: (OPTIONAL)
```shell
# Use inno_lidar_set_log_level() to specify the log level.
void inno_lidar_set_log_level(enum InnoLogLevel log_level)
```


## step 2: (REQUIRED)
```shell
# Use inno_lidar_open_live() to connect to a sensor.
int inno_lidar_open_live(const char *name, const char *lidar_ip, uint16_t port,
                          enum InnoLidarProtocol protocol, uint16_t udp_port)

OR

# Use inno_lidar_open_file() to open a raw data file.
int inno_lidar_open_file(const char *name, const char *filename, bool raw_format, int play_rate, int rewind,
                         int64_t skip)
```


## step 3: (REQUIRED)
```shell
# Use inno_lidar_set_callbacks() to specify the lidar_message, lidar_dataPacket and lidar_status callbacks.
# message_callback handles warning, error, and critical-error notifications that might occur.
# data_callback provides access to the lidar pointcloud data.
# status_callback provides status information of lidar.
int inno_lidar_set_callbacks(int handle, InnoMessageCallback message_callback, InnoDataPacketCallback data_callback,
                             InnoStatusPacketCallback status_callback, InnoHosttimeCallback get_host_time,
                             void *callback_context)
```

## step 4: (REQUIRED)
```shell
# Use inno_lidar_start() to start reading from live lidar or file.
int inno_lidar_start(int handle)
```

## step 5: (REQUIRED)
```shell
# Use inno_lidar_stop() to stop reading.
int inno_lidar_stop(int handle)
```
## step 6: (REQUIRED)
```shell
# Use inno_lidar_close() to close a lidar handle and release any associated resources.
int inno_lidar_close(int handle)
```
