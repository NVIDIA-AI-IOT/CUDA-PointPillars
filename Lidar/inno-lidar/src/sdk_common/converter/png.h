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

#ifndef PCS_PNG_H_
#define PCS_PNG_H_

#include <string.h> /*for size_t*/

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace innovusion {

  using PNGVector = std::vector<std::uint8_t>;

  // function to color image point
class PNGChunk;
  using SaveFunc = std::function<bool(PNGChunk &)>;

  //
  // The PNG color types (also used for raw image).
  //
  enum class PNGColorType {
    GREY = 0,         // grayscale: 8,16 bit
    RGB = 2,          // RGB: 8,16 bit
    PALETTE = 3,      // palette: 8 bit

    GREY_ALPHA = 4,   // grayscale with alpha: 8,16 bit
    RGBA = 6,         // RGB with alpha: 8,16 bit
  };

  //***************************************************************************

  //
  // Every chunk has the same structure:
  //    a 4-byte length
  //        (big-endian format, as with all integer values in PNG streams),
  //    a 4-byte chunk type,
  //    between 0 and 2,147,483,647 bytes of chunk data,
  //    and a 4-byte cyclic redundancy check value (CRC).
  //
class PNGChunk {
 private :
    explicit PNGChunk(int length, const char *type);
    ~PNGChunk();

    void uint2byte(unsigned val, int offset);

    void generate_crc();

 public :
    // The length field refers to the length of the data field alone,
    // not the chunk type or CRC length of the data of the chunk in bytes
    std::uint32_t length_;

    // ASCII a-z,A-Z only
    // PNG standard chunk ASCII naming conventions:
    // 1st byte: uppercase = critical, lowercase = ancillary
    // 2nd byte: uppercase = public, lowercase = private
    // 3rd byte: must be uppercase
    // 4th byte: uppercase = unsafe to copy, lowercase = safe to copy
    std::array<char, 4> type_;

    // length bytes of data
    PNGVector data_;

    // 4 bytes of CRC, computed on chunk name + data
    std::uint32_t crc_;

 public:
    void saveto_buffer(std::vector<char>* buf);
    void saveto_file(FILE *file);

 public:
    static bool save_IHDR(const SaveFunc &saveFunc,
                          unsigned w, unsigned h,
                          PNGColorType colortype,
                          unsigned bitdepth,
                          unsigned interlace_method);

    static bool save_PLTE(const SaveFunc &saveFunc, const PNGVector &palette);

    static bool save_IDAT(const SaveFunc &saveFunc, const PNGVector& raw);

    static bool save_IEND(const SaveFunc &saveFunc);
};

  //*****************************************************************************

class PNGWriter {
 public:
    PNGWriter(unsigned w, unsigned h,
              PNGColorType colortype,
              unsigned bitdepth);

    ~PNGWriter();

    //
    bool encode(int x, int y, unsigned char r, unsigned char g,
                unsigned char b);

    //
    bool saveto_buffer(std::vector<char>* buf) const;

    //
    bool saveto_file(const std::string &filename) const;

 private:
    //
    bool filter();
    bool save(const SaveFunc &saveFunc) const;

 private:
    PNGVector image_data_;

    unsigned width_;
    unsigned height_;

    PNGColorType colortype_;
    unsigned bitdepth_;

    unsigned compression_method;    // Always 0.
    unsigned filter_method;         // 0 ~ 4
    unsigned interlace_method;      // 0 = none, 1 = Adam7
};
}         // namespace innovusion

#endif    // PCS_PNG_H_
