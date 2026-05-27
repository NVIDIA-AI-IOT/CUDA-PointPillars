/*
 *  Copyright (C) 2022 Innovusion Inc.
 *
 *  License: BSD Software License Agreement
 *
 *  $Id$
 */

/**
 *  @file
 *
 *  Innovusion LIDAR SDK Header File
 *
 *  The file provides the data structures definition and exported functions of
 *  Innovusion LIDAR SDK.
 */

/**
 *  Simple 7-step introduction describing how to use the SDK to write
 *  an Innovusion Lidar driver.
 *
 *  step 1a: (OPTIONAL)
 *    Use inno_lidar_setup_sig_handler() to setup handling of SIGSEGV, SIGBUS,
 *        and SIGFPE signals.
 *
 *  step 1b: (OPTIONAL)
 *    Use inno_lidar_set_logs() to specify the log file fds (file descriptors)
 *        and callback.
 *    Use inno_lidar_set_log_level() to specify the log level.
 *
 *  step 2: (REQUIRED)
 *    Use inno_lidar_open_live() to connect to a sensor.
 *    Or use inno_lidar_open_file() to open a raw data file.
 *    Both methods return a lidar handle.
 *

 *  step 3: (REQUIRED)
 *    Use inno_lidar_set_callbacks() to specify the lidar_message and
 lidar_packet
 *        callbacks.
 *    Use the lidar_message callback to handle warning, error, and
 critical-error
 *        notifications that might occur.
 *    The lidar_packet callback provides access to the lidar pointcloud data.
 *
 *  step 4A: (REQUIRED)
 *    Use inno_lidar_start() to start reading from live lidar or file.
 *    Corresponding callbacks will be triggered.  The callbacks are called in
 *    the context of a thread dedicated to each lidar handle, i.e. one callback
 *    queue per lidar handle.  Please note that multiple callbacks may be
 *    called AT THE SAME TIME for different lidar handles.
 *
 *  step 5: (REQUIRED)
 *    Use inno_lidar_stop() to stop reading.  No further callbacks will be
 *    called once this call returns.
 *
 *  step 6: (REQUIRED)
 *    Use inno_lidar_close() to close a lidar handle and release any
 *    associated resources.
 *
 *  Threading:
 *    In inno_lidar_start(), the SDK library will use pthread library to spawn
 *    threads to read and process data and make callbacks.
 */

#ifndef APPS_INNO_LIDAR_API_WIN_EXAMPLE_INCLUDE_INNO_LIDAR_API_H_
#define APPS_INNO_LIDAR_API_WIN_EXAMPLE_INCLUDE_INNO_LIDAR_API_H_

#include <math.h>
#include <stdint.h>

#include "sdk_common/inno_lidar_packet.h"
#include "utils/inno_lidar_log.h"

#if defined(_WIN32) && !defined(__MINGW64__)
#define INNO_DLLEXPORT __declspec(dllexport)
#define INNO_DLLIMPORT __declspec(dllimport)
#else
#define INNO_DLLEXPORT
#define INNO_DLLIMPORT
#endif

#ifdef INNO_EXPORTS
#define INNO_API INNO_DLLEXPORT
#else
#define INNO_API INNO_DLLIMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/********************
 * exported functions
 ********************/
/*
 * @brief Open a lidar handle for a live Innovusion lidar unit.
 * @param name Contains the name of lidar, e.g. "forward-1"
 *        The name should be less than 32 characters.
 * @param lidar_ip Contains the lidar IP string, e.g. "172.168.1.10".
 * @param port Lidar PORT, e.g. 8010.
 * @param protocol Specify protocol,
 * @param udp_port UDP port to receive data, when
 *                 protocol is INNO_LIDAR_PROTOCOL_UDP
 * @return Return lidar handle
 */
INNO_API int inno_lidar_open_live(const char *name, const char *lidar_ip, uint16_t port,
                                  enum InnoLidarProtocol protocol, uint16_t udp_port);

/*
 * @brief Open a lidar handle from an Innovusion lidar proprietary
 *        data file
 * @param name Name of lidar.
 *        The name should be less than 32 characters.
 * @param filename The filename of an Innovusion lidar
 *        proprietary data file.
 * @param raw_format
 * @param play_rate The playback rate,
 *        == 0 means play as fast as possible
 *        <= 100 means MB/s, e.g. 15 means play at 15 MB/s
 *         > 100 means rate_multiplier = play_rate / 10000.0,
 *           e.g, 15000 means play at 1.5x speed
 * @param rewind How many times rewind before stop, 0 means no rewind,
 *        < 0 means rewind infinity times.
 * @return Return a lidar handle.
 */
INNO_API int inno_lidar_open_file(const char *name, const char *filename, bool raw_format, int play_rate, int rewind,
                                  int64_t skip);

/*
 * @brief Set callbacks for a lidar handle. Developers need to
 *        set callbacks before calling inno_lidar_start().
 *        For every lidar handle, inno_lidar_set_callbacks can
 *        only be called once.
 * @param handle Lidar handle (from either inno_lidar_open_live or
 *        inno_lidar_open_file).
 * @param message_callback This callback is called when some warning/error
 *                            happens.
 *                            For a given lidar handle, no message_callback or
 *                            frame_callback will be called until the
 *                            previous callback (for the same lidar handle)
 *                            has returned.
 *                            Multiple callbacks may happen at the same time
 *                            for different lidar handles.
 *                            Pass NULL pointer means no callback.
 * @param data_callback This callback is called when one frame or
 *                            sub-frame is available.  The callback happens
 *                            in a thread dedicated to that lidar_handle.
 *                            For a given lidar handle, no message_callback or
 *                            frame_callback will be called until the
 *                            previous callback (for the same lidar handle)
 *                            has returned.
 *                            Multiple callbacks may happen at the same time
 *                            for different lidar handles.
 *                            Pass NULL pointer means no callback.
 * @param status_callback
 * @param get_host_time This callback will get the current host time.
 *                           Pass NULL pointer means
 * clock_gettime(CLOCK_REALTIME) will be used.
 * @param callback_context Context passed in when callback is invoked.
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_set_callbacks(int handle, InnoMessageCallback message_callback,
                                      InnoDataPacketCallback data_callback, InnoStatusPacketCallback status_callback,
                                      InnoHosttimeCallback get_host_time, void *callback_context);

/*
 * @brief Start to read data from live lidar unit or from a data file.
 *        The message_callback, frame_callback and status_callback will
 *        be called only after this function is called.  They may
 *        be called before this function returns since they are in
 *        different threads.
 * @param handle Lidar handle.
 * @return 0 means success, otherwise failure (e.g. invalid handle).
 */
INNO_API int inno_lidar_start(int handle);

/*
 * @brief Stop reading data from live lidar unit or from a data file.
 *        The message_callback, frame_callback and status_callback will
 *        not be called once this function is returned.
 * @param handle Lidar handle.
 * @return 0 means success, otherwise failure (e.g. invalid handle)
 */
INNO_API int inno_lidar_stop(int handle);

/*
 * @brief Close the lidar handle and release all resources.
 *        This function can be called on a valid lidar handle only
 *        after inno_lidar_stop() call has returned,
 *        or inno_lidar_start() has never been called before.
 * @param handle Lidar handle.
 * @return 0 means success, otherwise failure (e.g. invalid handle).
 */
INNO_API int inno_lidar_close(int handle);

/*
 * @brief Change the meaning of the ref data in InnoChannelPoint
 *        structures to indicate either reflectance or intensity.
 * @param handle Lidar handle
 * @param mode Either INNO_REFLECTANCE_MODE_INTENSITY
 *        or INNO_REFLECTANCE_MODE_REFLECTIVITY
 *        default is INNO_REFLECTANCE_MODE_REFLECTIVITY
 *        since 1.5.0
 * @return 0 means success, otherwise failure (e.g. invalid handle).
 */
INNO_API int inno_lidar_set_reflectance_mode(int handle, enum InnoReflectanceMode mode);

/*
 * @brief Set lidar working mode
 * @param mode Target mode
 * @param mode_before_change Mode bofore this change
 * @param status_before_change Lidar Status before this change
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_set_mode(int handle, enum InnoLidarMode mode, enum InnoLidarMode *mode_before_change,
                                 enum InnoLidarStatus *status_before_change);

/*
 * @brief Query lidar unit's model
 * @param handle Lidar handle.
 * @param buffer Buffer to store the model.
 * @param buffer_len Length of buffer, the recommended buffer_len is
 *        32, the size of buffer needs to be >= buffer_len
 * @return 0 means success
 *        -1 invlid lidar handle
 *        -2 buffer_len is too small
 *        -3 source is file
 *        otherwise return the size of model string
 *        stored in buffer, not include the trailing zero
 */
INNO_API int inno_lidar_get_model(int handle, char *buffer, int buffer_len);

/*
 * @brief Get lidar current working mode and status
 * @param handle Lidar handle
 * @param mode Current work mode
 * @param pre_mode Previous work mode
 * @param status Work status
 * @param in_transition_mode_ms Time (in ms) stay in transition mode
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_get_mode_status(int handle, enum InnoLidarMode *mode, enum InnoLidarMode *pre_mode,
                                        enum InnoLidarStatus *status, uint64_t *in_transition_mode_ms);
/*
 * @brief Change the number & type of points returned each laser pulse.
 * @param handle Lidar handle.
 * @param ret_mode
 * @return 0 means success, otherwise failure (e.g. invalid handle).
 */
INNO_API int inno_lidar_set_return_mode(int handle, enum InnoMultipleReturnMode ret_mode);

/*
 * @brief Get the state of the lidar.
 * @param handle Lidar handle.
 * @return InnoLidarBase::State, -1 failure (e.g. invalid handle)
 */
INNO_API int inno_lidar_get_state(int handle);
/*
 * @brief Set ROI (center point)
 * @param handle Lidar handle.
 * @param horz_angle ROI horizontal center (in [-60, 60] degrees)
 * @param vert_angle ROI vertical center (in [-13, 13] degrees)
 *        Setting either horz_angle or vert_angle to
 *        kInnoNopROI, i.e. 10000.0
 *        will maintain the previous value for that angle.  Any other value
 *        outside the allowed range will result in a failure code return and
 *        no change in ROI center.
 * @return 0 means success, otherwise failure (e.g. invalid handle)
 */
INNO_API int inno_lidar_set_roi(int handle, double horz_angle, double vert_angle);

/*
 * @brief Get current ROI (center point)
 * @param handle Lidar handle.
 * @param horz_angle Store return ROI horizontal center (in [-60, 60] degrees)
 * @param vert_angle StoreROI vertical center (in [-13, 13] degree)
 * @return 0 means success, otherwise failure (e.g. invalid handle)
 */
INNO_API int inno_lidar_get_roi(int handle, double *horz_angle, double *vert_angle);

/*
 * @brief Set debug level for all lidar handles.
 *        This function can be called any time.
 *        It may be called multiple times, e.g. to decrease the log
 *        level before executing some code and then increasing the
 *        log level again after the section of code completes.
 * @param log_level Set new debug level.
 * @return Void.
 */
INNO_API void inno_lidar_set_log_level(enum InnoLogLevel log_level);

/*
 * @brief Get lidar anglehv table
 * @param handle Lidar handle
 * @param anglehv_table Buffer to store the table
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_get_anglehv_table(int handle, InnoDataPacket *anglehv_table);

/*
 * @brief Get lidar attribute like frame_rate
 * @param handle Lidar handle
 * @param attribute Name of the attribute
 * @param buffer Buffer to store the string value
 * @param Buffer_len Length of buffer, the recommended buffer_len is
 *                 1024, the size of buffer needs to be >= buffer_len.
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_get_attribute_string(int handle, const char *attribute, char *buffer, size_t buffer_len);

/*
 * @brief Set lidar attribute
 * @param handle Lidar handle
 * @param attribute Name of the attribute
 * @param buffer Buffer to store the string value
 * @return 0 means success, otherwise failure (e.g. invalid lidar handle)
 */
INNO_API int inno_lidar_set_attribute_string(int handle, const char *attribute, const char *buffer);

/*
 * @brief Set config name-value pair for a lidar handle.
 *        This function should be called before inno_lidar_start is called,
 *        and can be called multiple times.
 * @param handle Lidar handle
 * @param cfg_name Name of the config item.
 *        It must start with "Lidar_" and have two parts: section and key.
 *        separated by '/'. e.g. Lidar_section1/key1
 * @param cfg_value Value of the config item.
 * @return 0 means success
 *         1 invalid handle
 *         2 invalid name
 *         3 invalid value
 */
INNO_API int inno_lidar_set_config_name_value(int handle, const char *cfg_name, const char *cfg_value);

/*
 * @brief Description: Query lidar unit's state
 * @param handle Lidar handle.
 * @param state
 * @param error_code Address to store error code. NULL means
 *        do not store error code
 * @return 0 means success, -1 if handle is invalid
 */
INNO_API int inno_lidar_get_fw_state(int handle, enum InnoLidarState *state, int *error_code);

/*
 * @brief Query lidar unit's firmware version.
 * @param handle Lidar handle.
 * @param buffer Buffer to store the firmware version.
 * @param Buffer_len Length of buffer, the recommended buffer_len is
 *                 512, the size of buffer needs to be >= buffer_len.
 * @return 0 means success
 *        -1 invlid lidar handle
 *        -2 buffer_len is too small
 *        -3 source is file
 *        otherwise return the size of firmware version string
 *         stored in buffer, not include the trailing zero
 *        Sample content in the buffer:
 *           App Version: app-2.3.0-rc8.134
 *             build-time: 2019-08-14-18-19-25
 *           FPGA Datecode: 0x190814e2
 *             fpga-ver: 0x13
 *             fpga-rev: 0x0e
 *             board-rev: 0x2
 *           Firmware Version: 2.3.1-rc3-418.2019-08-15-17-41-40
 *             build-tag: 2.3.1-rc3-418
 *             build-time: 2019-08-15-17-41-40
 *             build-git-tag: 1.0.19
 */
INNO_API int inno_lidar_get_fw_version(int handle, char *buffer, int buffer_len);

/*
 * @brief Query lidar unit's S/N
 * @param handle Lidar handle.
 * @param buffer Buffer to store the S/N.
 * @param buffer_len Length of buffer, the recommended buffer_len is
 *                 128, the size of buffer needs to be >= buffer_len.
 * @return  0 means success
 *         -1 invlid lidar handle
 *         -2 buffer_len is too small
 *         -3 source is file
 *         otherwise return the size of S/N string
 *         stored in buffer, not include the trailing zero
 */
INNO_API int inno_lidar_get_sn(int handle, char *buffer, int buffer_len);


/*
 * @brief Get max number of points in the pkt. For
 *        INNO_ITEM_TYPE_SPHERE_POINTCLOUD pkt it is calculated
 *        based on number of blocks and number of returns.
 * @param pkt DataPacket
 * @return max number of points in the pkt
 */
INNO_API uint32_t inno_lidar_get_max_points_count(const InnoDataPacket *pkt);

/*
* @brief 
  [[Deprecated]]: not support compact pointcloud
  Convert SPHERE POINTCLOUD points the source
        data packet to XYZ_POINTCLOUD and add them
        to the destination data packet.
* @param src Source data packet
* @param dest Destination data packet
* @param dest_size Max size of the destination data packet
* @param crc_disable: true: not do crc for dest InnoDataPacket
* @return false if the src pkt is invalid, true otherwise
*/
INNO_API bool inno_lidar_convert_to_xyz_pointcloud(const InnoDataPacket *src, InnoDataPacket *dest, size_t dest_size,
                                                   bool crc_disable);

/*
* @brief Convert SPHERE POINTCLOUD points the source
        data packet to XYZ_POINTCLOUD and add them
        to the destination data packet.
* @param src Source data packet
* @param dest Destination data packet
* @param dest_size Max size of the destination data packet
* @param crc_disable: true: not do crc for dest InnoDataPacket
* @param hvangle_table: anglehv table for convert to sphere pointcloud
* hvangle table is get from inno_lidar_get_anglehv_table
* only InnoItemType is INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD need hvangle_table to convert to sphere pointcloud
* other type don't need hvangle_table, can set NULL
* @return false if the src pkt is invalid, true otherwise
*/
INNO_API bool inno_lidar_convert_to_xyz_pointcloud2(const InnoDataPacket *src, InnoDataPacket *dest, size_t dest_size,
                                                    bool crc_disable, const InnoDataPacket *hvangle_table);
/**
 * @brief check if occur fault
 * @param pkt StatusPacket
 * @return false if not occur fault, true otherwise
 */
INNO_API bool inno_lidar_check_status_packet_fault(const InnoStatusPacket *pkt);

/**
 * @brief check if occur fault specified by fault id
 * @param pkt StatusPacket
 * @param fid fault id
 * @return false if not occeur fault, true otherwise
 */
INNO_API bool inno_lidar_check_status_packet_fault_id(const InnoStatusPacket *pkt, int fid);

/*
 * @brief handle lidar open handle.
 * @param mode lidar current mode
 * @param scan_direction lidar scan direction
 * @param scan_id lidar scan id
 * @param ch lidar scan channel
 * @return ring id, if failed, return 0.
 */
INNO_API int inno_lidar_get_ring_id(int handle, enum InnoLidarMode mode, uint32_t scan_direction, uint32_t scan_id,
                                    uint32_t ch);

/*
* @brief Get ring id converter
* @param handle Lidar handle
* @return ring id converter
*/
INNO_API void *inno_lidar_get_ring_id_converter(int handle);

/*
 * @brief Get Innovusion lidar API version
 * @return Version string.
 */
INNO_API const char *inno_api_version(void);

/*
 * @brief Get Innovusion lidar API build tag.
 * @return Build tag string.
 */
INNO_API const char *inno_api_build_tag(void);

/*
 * @brief Get Innovusion lidar API build time.
 * @return Build time string.
 */
INNO_API const char *inno_api_build_time(void);

/**
 * @brief Setup sigaction for SIGSEGV SIGBUS SIGFPE (optional)
 *        The signal handler will print out the stack backtrace
 *        and then call old signal handler.
 * @return Void.
 */
INNO_API void inno_lidar_setup_sig_handler(void);

/**
 * @brief Set log files fds (file descriptors).
 *        This function can be only called once.
 * @param out_fd All log whose level is less or equals
 *        to INNO_LOG_LEVEL_INFO will be written to this file.
 *        -1 means do not write.
 * @param error_fd All log whose level is bigger or equals
 *        to INNO_LOG_LEVEL_WARNING will be written to this file.
 *        -1 means do not write.
 * @param rotate_file_base_file
 * @param rotate_file_number
 * @param rotate_file_size_limit
 * @param log_callback
 * @param callback_ctx
 * @param rotate_file_base_file_err
 * @param rotate_file_number_err
 * @param rotate_file_size_limit_err
 * @param flags (1 means use use_async)
 * @return Void.
 */
INNO_API void inno_lidar_set_logs(int out_fd, int error_fd, const char *rotate_file_base_file,
                                  uint32_t rotate_file_number, uint64_t rotate_file_size_limit,
                                  InnoLogCallback log_callback, void *callback_ctx,
                                  const char *rotate_file_base_file_err, uint32_t rotate_file_number_err,
                                  uint64_t rotate_file_size_limit_err, uint64_t flags);

/**
 * @brief Set Lidar log function
 * @param  log_callback : Callback function
 * @param  ctx :          Callback function context
 */
INNO_API void inno_lidar_log_callback(InnoLogCallback log_callback, void *ctx);

/*
 * @brief Download "uds.log","ila.log","inno_pc_server.txt" and "lidar-log.txt"
 * @param ip (e.g. 10.42.0.91)
 * @param Download path (e.g. /home/demo/Desktop/pipeline)
 * @return 0 means success, otherwise failure
 */
INNO_API int inno_lidar_download_logs(const char* ip, const char* path);

/**
 * @brief Set recorder callback
 * @param handle            Lidar handle
 * @param type              INNO_RECORDER_CALLBACK_TYPE_INNO_PC
 * @param callback          Callback function
 * @param callback_context  Callback function context
 * @return Return 0 for success, others for error
 */
INNO_API int inno_lidar_set_recorder_callback(int handle, enum InnoRecorderCallbackType type,
                                             InnoRecorderCallback callback, void *callback_context);
#ifdef __cplusplus
}
#endif

#endif  // APPS_INNO_LIDAR_API_WIN_EXAMPLE_INCLUDE_INNO_LIDAR_API_H_
