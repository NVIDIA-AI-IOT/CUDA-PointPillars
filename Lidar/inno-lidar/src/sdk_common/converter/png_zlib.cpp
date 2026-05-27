/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

/**
 part of the source code are from
 https://github.com/lvgl/lv_lib_png/blob/master/lodepng.c

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

#include "sdk_common/converter/png_zlib.h"
// #include <stdlib.h> /* allocations */

namespace innovusion {
/*
  non compressed deflate block data: 
    1 bit BFINAL,
    2 bits BTYPE,
    (5 bits): it jumps to start of next byte, 
    2 bytes LEN, 
    2 bytes NLEN, 
    LEN bytes literal DATA
*/
static unsigned deflateNoCompression(std::vector<unsigned char>* out,
                                     const std::vector<unsigned char> &in) {
  size_t datasize = in.size();

  size_t block_size = 65535u;
  size_t block_count = (in.size() + block_size - 1) / block_size;

  unsigned datapos = 0;
  for (size_t i = 0; i != block_count; ++i) {
    unsigned BFINAL = (i == block_count - 1);
    unsigned BTYPE = 0;

    unsigned LEN = block_size;
    if (datasize - datapos < block_size) LEN = (unsigned)datasize - datapos;
    unsigned NLEN = block_size - LEN;

    unsigned char firstbyte =
        (unsigned char)(BFINAL + ((BTYPE & 1u) << 1u) + ((BTYPE & 2u) << 1u));
    out->push_back(firstbyte);
    out->push_back((unsigned char)(LEN & 255));
    out->push_back((unsigned char)(LEN >> 8u));
    out->push_back((unsigned char)(NLEN & 255));
    out->push_back((unsigned char)(NLEN >> 8u));

    out->insert(out->end(), in.data() + datapos, in.data() + datapos + LEN);
    datapos += LEN;
  }

  return 0;
}

/*Return the adler32 of the bytes data[0..len-1]*/
static unsigned update_adler32(unsigned adler,
                               const std::vector<unsigned char> &in) {
  unsigned s1 = adler & 0xffffu;
  unsigned s2 = (adler >> 16u) & 0xffffu;

  unsigned i = 0;
  for (unsigned char c : in) {
    s1 += c;
    s2 += s1;

    i++;

    /*at least 5552 sums can be done before the sums overflow, saving a lot of
     * module divisions*/
    if (i >= 5552u) {
      s1 %= 65521u;
      s2 %= 65521u;

      i = 0;
    }
  }

  if (i > 0) {
    s1 %= 65521u;
    s2 %= 65521u;
  }

  return (s2 << 16u) | s1;
}


/*  zlib data: 
      1 byte CMF (CM+CINFO), 
      1 byte FLG, 
      deflate data, 
      4 byte ADLER32
      checksum of the Decompressed data
  */
void PNGCompress::save_file_header(std::vector<unsigned char> *out) const {
  // 0b01111000: CM 8, CINFO 7. With CINFO 7
  // any window size up to 32768 can be used.
  unsigned CMF = 120;
  unsigned FLEVEL = 0;
  unsigned FDICT = 0;
  unsigned CMFFLG = 256 * CMF + FDICT * 32 + FLEVEL * 64;
  unsigned FCHECK = 31 - CMFFLG % 31;
  CMFFLG += FCHECK;

  out->push_back((unsigned char)(CMFFLG >> 8));
  out->push_back((unsigned char)(CMFFLG & 255));
}

void PNGCompress::save_file_end(std::vector<unsigned char> *out,
                                const std::vector<unsigned char> &in) const {
  unsigned ADLER32 = update_adler32(1u, in);

  // buffer must have at least 4 allocated bytes available
  out->push_back((unsigned char)((ADLER32 >> 24) & 0xff));
  out->push_back((unsigned char)((ADLER32 >> 16) & 0xff));
  out->push_back((unsigned char)((ADLER32 >> 8) & 0xff));
  out->push_back((unsigned char)((ADLER32)&0xff));
}

bool PNGCompress::compress(std::vector<unsigned char> *out,
                           const std::vector<unsigned char> &in) const {
  save_file_header(out);

  //
  //  int insize = in.size();

  //
  /*on PNGs, deflate blocks of 65-262k seem to give most dense encoding*/
  //  size_t blocksize = insize / 8u + 8;
  //
  //  if (blocksize < 65536) blocksize = 65536;
  //
  //  if (blocksize > 262144) blocksize = 262144;

  //
  unsigned error = 0;
  error = deflateNoCompression(out, in);

  //
  save_file_end(out, in);
  return error == 0;
}
}  // namespace innovusion
