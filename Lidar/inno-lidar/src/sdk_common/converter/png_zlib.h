/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

/**
 part of the source code are from
 https://github.com/lvgl/lv_lib_png/blob/master/lodepng.h

 Copyright (c) 2005-2020 Lode Vandevenne
 This software is provided 'as-is', without any express or implied
 warranty. In no event will the authors be held liable for any damages
 arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.
 3. This notice may not be removed or altered from any source
    distribution.
*/

#ifndef PCS_PNG_ZLIB_H_
#define PCS_PNG_ZLIB_H_

#include <string.h> /*for size_t*/
#include <vector>

/*Compress a buffer with deflate. See RFC 1951. Out buffer must be freed after
 * use.*/
namespace innovusion {
class PNGCompress {
 public:
  /*
    Compresses data with Zlib.
    Reallocates the out buffer and appends the data.
    Zlib adds a small header and trailer around the deflate data.
    The data is output in the format of the zlib specification.
    Either, *out must be NULL and *outsize must be 0, or, *out must be a valid
    buffer and *outsize its size in bytes. out must be freed by user after
    usage.
  */
  bool compress(std::vector<unsigned char> *out,
                const std::vector<unsigned char> &in) const;

 private:
  void save_file_header(std::vector<unsigned char> *out) const;
  void save_file_end(std::vector<unsigned char> *out,
                     const std::vector<unsigned char> &in) const;
};
}  // namespace innovusion

#endif  // PCS_PNG_ZLIB_H_
