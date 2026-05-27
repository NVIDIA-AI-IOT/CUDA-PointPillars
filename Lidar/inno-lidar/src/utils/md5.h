/*
 * Copyright [2019] No Copyright is Claimed
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * See md5.c for more information.
 */
#ifndef UTILS_MD5_H_
#define UTILS_MD5_H_

#ifdef HAVE_OPENSSL
#include <openssl/md5.h>
#else
#include <stddef.h>
#include <stdint.h>

/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD5_u32plus;
typedef struct MD5_result {
  unsigned char result[16];
} MD5_result;

typedef struct MD5_result_str {
  char str[33];
} MD5_result_string;

typedef struct {
  MD5_u32plus lo, hi;
  MD5_u32plus a, b, c, d;
  unsigned char buffer[64];
  MD5_u32plus block[16];
} MD5_CTX;

/**
 * @brief Initialize MD5_calculation
 * @param ctx MD5_CTX to be initialized
 */
extern void MD5_Init(MD5_CTX *ctx);
/**
 * @brief Update MD5_calculation
 * @param ctx MD5_CTX to store the updated result
 * @param data Data to update
 * @param size Size of the data
 */
extern void MD5_Update(MD5_CTX *ctx, const void *data, uint64_t size);
/**
 * @brief Calculate the final MD5 result
 * @param result Buffer to store the result
 * @param ctx MD5_CTX, which will be reset at the end of the calculation
 */
extern void MD5_Final(unsigned char *result, MD5_CTX *ctx);
/**
 * @brief Print the MD5 result into the given buffer
 * @param str    Buffer to store the result
 * @param size   Size of the buffer
 * @param result MD5 result
 */
extern void MD5_print(char *str, size_t size, const unsigned char *result);
#endif  // HAVE_OPENSSL
#else
#endif  // UTILS_MD5_H_
