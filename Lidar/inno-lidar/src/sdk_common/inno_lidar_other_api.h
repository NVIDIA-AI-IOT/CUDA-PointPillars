/*
 *  Copyright (C) 2021 Innovusion Inc.
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

#ifndef SDK_COMMON_INNO_LIDAR_OTHER_API_H_
#define SDK_COMMON_INNO_LIDAR_OTHER_API_H_

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

extern "C" {
/********************
 * exported functions
 ********************/


/*
* @brief Return the data packet size according to the item type, count
        and return mode.
* @param type Data packet type
* @param item_count Item count
* @param mode Multi-return mode
* @return size of data packet in bytes. Return 0 if the type is invalid.
*/
size_t inno_lidar_get_data_packet_size(enum InnoItemType type, uint32_t item_count,
                                                enum InnoMultipleReturnMode mode);

/*
 * @brief Set the velocity and angular velocity for motion correction
 * @param handle Lidar handle.
 * @param velocity_m_s[3]: velocities in x, y, z axis, unit is m/s
 * @param angular_velocity_rad_s[3]: angular velocities in x, y, z axis, RAD/s
 * @return 0 means success, -1 if handle is invalid
 */
int inno_lidar_set_motion_compensation(int handle, double velocity_m_s[3], double angular_velocity_rad_s[3]);

/*
 * @brief Open a lidar handle from a InputParam
 * @param name Name of lidar.
 *        The name should be less than 32 characters.
 * @param ctx The InputParam contex.
 * @return Return a lidar handle.
 */
INNO_API int inno_lidar_open_ctx(const char *name, void *ctx);

/*
 * @brief Sanity check the integrity of a InnoDataPacketGet.
 * @param pkt DataPacket
 * @param size Size of pkt if it is received from network or
 *        read from file
 * @return false if the pkt is invalid, true otherwise
 */
INNO_API bool inno_lidar_check_data_packet(const InnoDataPacket *pkt, size_t size);

/*
 * @brief Sanity check the integrity of a InnoDataPacketGet.
 * @param pkt StatusPacket
 * @param size Size of pkt if it is received from network or
 *        read from file, 0 means don't check size
 * @return false if the pkt is invalid, true otherwise
 */
INNO_API bool inno_lidar_check_status_packet(const InnoStatusPacket *pkt, size_t size);

/*
 * @brief InnoStatusPacket formatted output.
 * @param pkt StatusPacket
 * @param buffer
 * @param buffer_size
 * @return Upon successful return, these functions return the
 *         number of characters printed (excluding the null byte
 *         used to end output to strings).
 *         If an output error is encountered, a negative value is returned.
 */
INNO_API int inno_lidar_printf_status_packet(const InnoStatusPacket *pkt, char *buffer, size_t buffer_size);

/*
 * @brief fix falconk/k24 IMU status.
 * @param pkt StatusPacket
 * @param pkt StatusPacket
 * @param is_wgs IMU physical direction, falcon-k: false falcon-k24: true
 * @return false if the pkt/out_pkt is invalid, true otherwise.
 */
INNO_API bool inno_lidar_correct_imu_status(const InnoStatusPacket *pkt, InnoStatusPacket *out_pkt,
                                            bool is_wgs = false);
/**
 * @brief Read data from PS resgiter, reserved for internal use
 * @param handle Lidar handle
 * @param off    Register offset
 * @param value  Buffer to store value read from register
 * @return Return 0 for success, others for error
 */
int inno_lidar_read_ps_reg(int handle, uint16_t off, uint32_t *value);

/**
 * @brief Read data from PL resgiter, reserved for internal use
 * @param handle Lidar handle
 * @param off    Register offset
 * @param value  Buffer to store value read from register
 * @return Return 0 for success, others for error
 */
int inno_lidar_read_pl_reg(int handle, uint16_t off, uint32_t *value);

/**
 * @brief Write data to PS resgiter, reserved for internal use
 * @param handle Lidar handle
 * @param off    Register offset
 * @param value  Value to be written
 * @return Return 0 for success, others for error
 */
int inno_lidar_write_ps_reg(int handle, uint16_t off, uint32_t value);

/**
 * @brief Write data to PL resgiter, reserved for internal use
 * @param handle Lidar handle
 * @param off    Register offset
 * @param value  Value to be written
 * @return Return 0 for success, others for error
 */
int inno_lidar_write_pl_reg(int handle, uint16_t off, uint32_t value);

/**
 * @brief Use lidar configuration parameter file for a lidar handle.
 *        This function should be called before inno_lidar_start is called.
 * @param handle Lidar handle.
 * @param lidar_model Lidar model name, e.g. "E" or "REV_E"
 * @param yaml_filename Full file path of the lidar configuration
 *        parameter file. The file is in yaml format.
 * @return 0 means success, otherwise failure (e.g. invalid handle)
 */
int inno_lidar_set_parameters(int handle, const char *lidar_model, const char *yaml_filename);
};

#endif  // SDK_COMMON_INNO_LIDAR_OTHER_API_H_
