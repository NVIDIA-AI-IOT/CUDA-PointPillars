/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef CONVERTER_PNG_RANGE_COLOR_H_
#define CONVERTER_PNG_RANGE_COLOR_H_

#include <array>

namespace innovusion {
//
//
//
struct Color {
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;

  Color() : r(0), g(0), b(0) {}

  Color(unsigned char r, unsigned char g, unsigned char b) : r(r), g(g), b(b) {}
};

//
//
//
struct ColorKey : Color {
  double pos;

  ColorKey() : Color(), pos(0) {}

  ColorKey(unsigned char r, unsigned char g, unsigned char b, double p)
      : Color(r, g, b), pos(p) {}
};

//
//
//
enum class ColorType {
  BGYR = 0  // Blue > Green > Yellow > Red
};

constexpr int MAX_STEPS = 1024;

//
//
//
class RangeColor {
 public:
  explicit RangeColor(ColorType color_type);
  ~RangeColor();

  // Get RGB color values for a given float in [0, 1]
  void getColor(float range, float ref, unsigned char* r, unsigned char* g,
                unsigned char* b);

 private:
  Color colorBlack = Color(0, 0, 0);    //  -INFINITY
  Color bkBlue = Color(150, 150, 200);  //   INFINITY(light blue),
  Color bkRed = Color(200, 150, 150);   //   NAN(light red)

  //
  std::array<Color, MAX_STEPS> color_step;

  //
  void initBGYR();

  //
  void initColorStep(ColorKey *keys, int key_count);
};
}  // namespace innovusion

#endif  // CONVERTER_PNG_RANGE_COLOR_H_
