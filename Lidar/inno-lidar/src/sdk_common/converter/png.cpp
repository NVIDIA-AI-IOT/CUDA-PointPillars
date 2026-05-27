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

#include "sdk_common/converter/png.h"

#include <limits.h> /* LONG_MAX */
#include <stdio.h>  /* file handling */
#include <stdlib.h> /* allocations */

#include "sdk_common/converter/png_zlib.h"
#include "utils/inno_lidar_log.h"

namespace innovusion {
// CRC polynomial: 0xedb88320
static unsigned png_crc32_table[256] = {
    0u,          1996959894u, 3993919788u, 2567524794u, 124634137u,
    1886057615u, 3915621685u, 2657392035u, 249268274u,  2044508324u,
    3772115230u, 2547177864u, 162941995u,  2125561021u, 3887607047u,
    2428444049u, 498536548u,  1789927666u, 4089016648u, 2227061214u,
    450548861u,  1843258603u, 4107580753u, 2211677639u, 325883990u,
    1684777152u, 4251122042u, 2321926636u, 335633487u,  1661365465u,
    4195302755u, 2366115317u, 997073096u,  1281953886u, 3579855332u,
    2724688242u, 1006888145u, 1258607687u, 3524101629u, 2768942443u,
    901097722u,  1119000684u, 3686517206u, 2898065728u, 853044451u,
    1172266101u, 3705015759u, 2882616665u, 651767980u,  1373503546u,
    3369554304u, 3218104598u, 565507253u,  1454621731u, 3485111705u,
    3099436303u, 671266974u,  1594198024u, 3322730930u, 2970347812u,
    795835527u,  1483230225u, 3244367275u, 3060149565u, 1994146192u,
    31158534u,   2563907772u, 4023717930u, 1907459465u, 112637215u,
    2680153253u, 3904427059u, 2013776290u, 251722036u,  2517215374u,
    3775830040u, 2137656763u, 141376813u,  2439277719u, 3865271297u,
    1802195444u, 476864866u,  2238001368u, 4066508878u, 1812370925u,
    453092731u,  2181625025u, 4111451223u, 1706088902u, 314042704u,
    2344532202u, 4240017532u, 1658658271u, 366619977u,  2362670323u,
    4224994405u, 1303535960u, 984961486u,  2747007092u, 3569037538u,
    1256170817u, 1037604311u, 2765210733u, 3554079995u, 1131014506u,
    879679996u,  2909243462u, 3663771856u, 1141124467u, 855842277u,
    2852801631u, 3708648649u, 1342533948u, 654459306u,  3188396048u,
    3373015174u, 1466479909u, 544179635u,  3110523913u, 3462522015u,
    1591671054u, 702138776u,  2966460450u, 3352799412u, 1504918807u,
    783551873u,  3082640443u, 3233442989u, 3988292384u, 2596254646u,
    62317068u,   1957810842u, 3939845945u, 2647816111u, 81470997u,
    1943803523u, 3814918930u, 2489596804u, 225274430u,  2053790376u,
    3826175755u, 2466906013u, 167816743u,  2097651377u, 4027552580u,
    2265490386u, 503444072u,  1762050814u, 4150417245u, 2154129355u,
    426522225u,  1852507879u, 4275313526u, 2312317920u, 282753626u,
    1742555852u, 4189708143u, 2394877945u, 397917763u,  1622183637u,
    3604390888u, 2714866558u, 953729732u,  1340076626u, 3518719985u,
    2797360999u, 1068828381u, 1219638859u, 3624741850u, 2936675148u,
    906185462u,  1090812512u, 3747672003u, 2825379669u, 829329135u,
    1181335161u, 3412177804u, 3160834842u, 628085408u,  1382605366u,
    3423369109u, 3138078467u, 570562233u,  1426400815u, 3317316542u,
    2998733608u, 733239954u,  1555261956u, 3268935591u, 3050360625u,
    752459403u,  1541320221u, 2607071920u, 3965973030u, 1969922972u,
    40735498u,   2617837225u, 3943577151u, 1913087877u, 83908371u,
    2512341634u, 3803740692u, 2075208622u, 213261112u,  2463272603u,
    3855990285u, 2094854071u, 198958881u,  2262029012u, 4057260610u,
    1759359992u, 534414190u,  2176718541u, 4139329115u, 1873836001u,
    414664567u,  2282248934u, 4279200368u, 1711684554u, 285281116u,
    2405801727u, 4167216745u, 1634467795u, 376229701u,  2685067896u,
    3608007406u, 1308918612u, 956543938u,  2808555105u, 3495958263u,
    1231636301u, 1047427035u, 2932959818u, 3654703836u, 1088359270u,
    936918000u,  2847714899u, 3736837829u, 1202900863u, 817233897u,
    3183342108u, 3401237130u, 1404277552u, 615818150u,  3134207493u,
    3453421203u, 1423857449u, 601450431u,  3009837614u, 3294710456u,
    1567103746u, 711928724u,  3020668471u, 3272380065u, 1510334235u,
    755167117u};

/*****************************************************************************/
// color mode

// static unsigned png_get_bpp(PNGColorType colortype, unsigned bitdepth)
// {
//   int channel = 0;
//   switch (colortype)
//   {
//   case PNGColorType::PALETTE:
//     channel = 1;
//     break;
//   case PNGColorType::GREY:
//     channel = 1;
//     break;
//   case PNGColorType::GREY_ALPHA:
//     channel = 2;
//     break;
//   case PNGColorType::RGB:
//     channel = 3;
//     break;
//   case PNGColorType::RGBA:
//     channel = 4;
//     break;
//   default:
//     return 0; /*invalid color type*/
//   }

//   /*bits per pixel is amount of channels * bits per channel*/
//   return channel * bitdepth;
// }

/*****************************************************************************/
// PNGChunk

PNGChunk::PNGChunk(int length, const char *type) {
  this->length_ = length;
  if (this->length_ > this->data_.capacity()) this->data_.reserve(length);

  for (int i = 0; i < 4; i++) type_[i] = type[i];
}

PNGChunk::~PNGChunk() {}

void PNGChunk::generate_crc() {
  unsigned r = 0xffffffffu;

  // type
  for (int i = 0; i < 4; i++) {
    r = png_crc32_table[(r ^ type_[i]) & 0xffu] ^ (r >> 8u);
  }

  // date
  for (auto val : data_) {
    r = png_crc32_table[(r ^ val) & 0xffu] ^ (r >> 8u);
  }

  this->crc_ = r ^ 0xffffffffu;
}

void PNGChunk::uint2byte(unsigned val, int offset) {
  data_[offset + 0] = (unsigned char)((val >> 24) & 0xff);
  data_[offset + 1] = (unsigned char)((val >> 16) & 0xff);
  data_[offset + 2] = (unsigned char)((val >> 8) & 0xff);
  data_[offset + 3] = (unsigned char)((val)&0xff);
}

//
//
void PNGChunk::saveto_buffer(std::vector<char>* buf) {
  if (buf == nullptr) return;

  // length
  {
    std::uint8_t val[4];
    val[0] = (unsigned char)((this->length_ >> 24) & 0xff);
    val[1] = (unsigned char)((this->length_ >> 16) & 0xff);
    val[2] = (unsigned char)((this->length_ >> 8) & 0xff);
    val[3] = (unsigned char)((this->length_) & 0xff);

    buf->insert(buf->end(), val, val + 4);
  }

  // type
  buf->insert(buf->end(), this->type_.begin(), this->type_.end());

  // data
  if (this->length_ > 0)
    buf->insert(buf->end(), this->data_.begin(), this->data_.end());

  // crc
  this->generate_crc();
  {
    std::uint8_t val[4];
    val[0] = (unsigned char)((this->crc_ >> 24) & 0xff);
    val[1] = (unsigned char)((this->crc_ >> 16) & 0xff);
    val[2] = (unsigned char)((this->crc_ >> 8) & 0xff);
    val[3] = (unsigned char)((this->crc_) & 0xff);

    buf->insert(buf->end(), val, val + 4);
  }
}

//
//
void PNGChunk::saveto_file(FILE *file) {
  // length
  {
    std::uint8_t val[4];
    val[0] = (unsigned char)((this->length_ >> 24) & 0xff);
    val[1] = (unsigned char)((this->length_ >> 16) & 0xff);
    val[2] = (unsigned char)((this->length_ >> 8) & 0xff);
    val[3] = (unsigned char)((this->length_) & 0xff);

    fwrite(val, 1, 4, file);
  }

  // type
  fwrite(this->type_.data(), 1, this->type_.size(), file);

  // data
  if (this->length_ > 0) fwrite(this->data_.data(), 1, this->length_, file);

  // crc
  this->generate_crc();
  {
    std::uint8_t val[4];
    val[0] = (unsigned char)((this->crc_ >> 24) & 0xff);
    val[1] = (unsigned char)((this->crc_ >> 16) & 0xff);
    val[2] = (unsigned char)((this->crc_ >> 8) & 0xff);
    val[3] = (unsigned char)((this->crc_) & 0xff);

    fwrite(val, 1, 4, file);
  }
}

bool PNGChunk::save_IHDR(const SaveFunc &saveFunc, unsigned w, unsigned h,
                         PNGColorType colortype, unsigned bitdepth,
                         unsigned interlace_method) {
  PNGChunk chunk(13, "IHDR");
  chunk.data_.resize(13);

  chunk.uint2byte(w, 0);  // width
  chunk.uint2byte(h, 4);  // height

  chunk.data_[8] = (unsigned char)bitdepth;   // bit depth
  chunk.data_[9] = (unsigned char)colortype;  // color type
  chunk.data_[10] = 0;                        // compression method
  chunk.data_[11] = 0;                        // filter method
  chunk.data_[12] = interlace_method;         // interlace method

  return saveFunc(chunk);
}

bool PNGChunk::save_PLTE(const SaveFunc &saveFunc, const PNGVector &palette) {
  PNGChunk chunk(palette.size(), "PLTE");

  // size_t j = 8;
  // for (size_t i = 0; i != info->palette.size(); ++i)
  // {
  //   /*add all channels except alpha channel*/
  //   chunk[j++] = info->palette[i * 4 + 0];
  //   chunk[j++] = info->palette[i * 4 + 1];
  //   chunk[j++] = info->palette[i * 4 + 2];
  // }

  return saveFunc(chunk);
}

bool PNGChunk::save_IDAT(const SaveFunc &saveFunc, const PNGVector& raw) {
  PNGChunk chunk(0, "IDAT");

  PNGCompress pngCompress;
  pngCompress.compress(&chunk.data_, raw);

  chunk.length_ = chunk.data_.size();
  inno_log_info("IDAT chunk.data : %u", chunk.length_);

  return saveFunc(chunk);
}

bool PNGChunk::save_IEND(const SaveFunc &saveFunc) {
  PNGChunk chunk(0, "IEND");

  return saveFunc(chunk);
}

/*****************************************************************************/
// PNGWriter

// filter type
constexpr int FILTER_NONE = 0;

//
constexpr int png_signature_size = 8;
const unsigned char png_signature[] = {0x89, 0x50, 0x4e, 0x47,
                                       0x0d, 0x0a, 0x1a, 0x0a};

PNGWriter::PNGWriter(unsigned w, unsigned h, PNGColorType colortype,
                     unsigned bitdepth) {
  this->width_ = w;
  this->height_ = h;

  this->colortype_ = colortype;
  this->bitdepth_ = bitdepth;

  this->interlace_method = 0;
  this->compression_method = 0;
  this->filter_method = 0;

  int size = width_ * height_ * 3;
  image_data_.reserve(size + height_);
}

PNGWriter::~PNGWriter() {}

//
//
//
bool PNGWriter::encode(int x, int y, unsigned char r, unsigned char g,
                       unsigned char b) {
  if (x == 0) image_data_.push_back(FILTER_NONE);

  image_data_.push_back(r);
  image_data_.push_back(g);
  image_data_.push_back(b);

  return true;
}

//
//
//
bool PNGWriter::save(const SaveFunc &func) const {
  inno_log_info("image_data_ size : %" PRI_SIZELU,
                image_data_.size());

  //
  PNGChunk::save_IHDR(func, width_, height_, colortype_, bitdepth_,
                      interlace_method);

  //
  PNGChunk::save_IDAT(func, this->image_data_);

  //
  PNGChunk::save_IEND(func);

  return true;
}


//
//
//
bool PNGWriter::saveto_buffer(std::vector<char>* buf) const {
  inno_log_info("save begin -- buff size : %" PRI_SIZELU,
                buf->size());

  // write signature
  buf->insert(buf->end(), png_signature, png_signature + png_signature_size);

  SaveFunc func = [&buf](PNGChunk &chunk) {
    chunk.saveto_buffer(buf);
    return true;
  };

  save(func);

  inno_log_info("save end -- buff size : %" PRI_SIZELU,
                buf->size());
  return true;
}


//
//
//
bool PNGWriter::saveto_file(const std::string &filename) const {
  FILE *file = fopen(filename.c_str(), "wb");

  if (!file) {
    inno_log_error("failed to open file for writing");
    return false;
  }

  // write signature
  fwrite(png_signature, 1, png_signature_size, file);

  SaveFunc func = [&file](PNGChunk &chunk) {
    chunk.saveto_file(file);
    return true;
  };

  save(func);

  fclose(file);
  return true;
}
}  // namespace innovusion
