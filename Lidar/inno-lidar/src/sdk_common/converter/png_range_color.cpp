/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include <cmath>
#include <vector>

#include "sdk_common/converter/png_range_color.h"

namespace innovusion {
RangeColor::RangeColor(ColorType color_type) { initBGYR(); }

RangeColor::~RangeColor() {}

void RangeColor::initBGYR() {
  ColorKey keys[4];
  keys[0] = ColorKey(0, 0, 255, 0);            // blue
  keys[1] = ColorKey(0, 255, 0, 1.0 / 3.0);    // green
  keys[2] = ColorKey(255, 255, 0, 2.0 / 3.0);  // yello
  keys[3] = ColorKey(255, 0, 0, 1);            // red

  initColorStep(keys, 4);
}

//
// Gradient Algorithm
//
void RangeColor::initColorStep(ColorKey *keys, int key_count) {
  unsigned j = 0;  // current interval

  for (std::size_t i = 0; i < MAX_STEPS; i++) {
    const double relativePos = static_cast<double>(i) / (MAX_STEPS - 1);

    // forward to the right intervale
    while (keys[j + 1].pos < relativePos) {
      // std::cout << keys[j + 1].pos << " " << relativePos << " " << i << " "
      // << j <<"\n";
      ++j;
    }

    // interpolation coef
    const double alpha =
        (relativePos - keys[j].pos) / (keys[j + 1].pos - keys[j].pos);

    // linear interpolation
    color_step[i] = Color(static_cast<unsigned char>(
                              keys[j].r + (keys[j + 1].r - keys[j].r) * alpha),
                          static_cast<unsigned char>(
                              keys[j].g + (keys[j + 1].g - keys[j].g) * alpha),
                          static_cast<unsigned char>(
                              keys[j].b + (keys[j + 1].b - keys[j].b) * alpha));

    // std::cout << i << " " << j << ":"
    //           << (int)color_step[i].r << " "
    //           << (int)color_step[i].g << " "
    //           << (int)color_step[i].b << "                    ";
    // if( i% 5 == 4)
    //   std::cout <<"\n";
  }
}

void RangeColor::getColor(float range, float reflectance, unsigned char* r,
                          unsigned char* g, unsigned char* b) {
  if (std::isinf(range)) {
    if (range > 0.0f) {
      *r = bkBlue.r;
      *g = bkBlue.g;
      *b = bkBlue.b;
    } else {
      *r = colorBlack.r;
      *g = colorBlack.g;
      *b = colorBlack.b;
    }
    return;
  }

  if (!std::isfinite(range)) {
    *r = bkRed.r;
    *g = bkRed.g;
    *b = bkRed.b;
    return;
  }

  //
  int index = -1;
  if (reflectance >= 0.0 && reflectance <= 1.0)
    index = static_cast<int>(reflectance * (MAX_STEPS - 1));

  if (index >= 0 && index < MAX_STEPS) {
    *r = color_step[index].r;
    *g = color_step[index].g;
    *b = color_step[index].b;
  } else {
    *r = colorBlack.r;
    *g = colorBlack.g;
    *b = colorBlack.b;
  }
}
}  // namespace innovusion
