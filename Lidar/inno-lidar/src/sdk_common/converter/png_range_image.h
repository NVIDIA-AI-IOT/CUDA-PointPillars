/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef CONVERTER_PNG_RANGE_IMAGE_H_
#define CONVERTER_PNG_RANGE_IMAGE_H_

#include <functional>
#include <vector>
#include <cstdint>
#if !defined(__MINGW64__) && defined(_WIN32)
#define ssize_t __int64
#define __attribute__(X)

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif
#endif

namespace innovusion {
//
struct RangePoint {
  float range;
  float reflectance;
};

using RangePointVector = std::vector<RangePoint>;

// function to color image point
using ColorFunc = std::function<void(float, float, unsigned char*,
                                     unsigned char*, unsigned char*)>;

// function to save image point
using ImageFunc =
    std::function<void(int, int, unsigned char, unsigned char, unsigned char)>;

class RangeImage {
  // True if no points are invalid
  bool is_dense = true;

 public:
  int32_t width = 0;
  int32_t height = 0;

 public:
  RangeImage(float angular_resolution_x, float angular_resolution_y,
             float max_angle_width, float max_angle_height);

  ~RangeImage();

  //
  void start(float noise_level = 0.0f, float min_range = 0.0f,
             int border_size = 0);

  // add point into the current range image using a z-buffer
  void insert_point(float x, float y, float z, float reflectance);

  //
  void stop();

  // Check if a point is inside of the image
  inline bool is_in_image(int x, int y) const;

  //
  void get_image(ColorFunc color_func, ImageFunc image_func);

 private:
  int crop_top = -1;
  int crop_bottom = -1;
  int crop_left = -1;
  int crop_right = -1;

  float noise_level = 0.0f;
  float min_range = 0.0f;
  int borderSize = 0;

  std::vector<int> counters_;

  //
  RangePointVector points;

  // This point is used to be able to return a reference to a non-existing point
  RangePoint unobserved_point;

  // Angular resolution of the range image in x/y direction in radians per pixel
  float angular_resolution_x_;
  float angular_resolution_y_;

  // provided for better performance of multiplication compared to division
  float angular_resolution_x_reciprocal_;  // 1.0/angular_resolution_x_
  float angular_resolution_y_reciprocal_;  // 1.0/angular_resolution_y_

  // Position of the top left corner of the range image
  // compared to an image of full size (360x180 degrees)
  int image_offset_x_;
  int image_offset_y_;

  // static
  static const int lookup_table_size;
  static std::vector<float> asin_lookup_table;
  static std::vector<float> atan_lookup_table;
  static std::vector<float> cos_lookup_table;

  // Create lookup tables for trigonometric functions
  static void createLookupTables();

  // Query the asin lookup table
  static inline float asinLookUp(float value);

  // Query the std::atan2 lookup table
  static inline float atan2LookUp(float y, float x);

  // Query the cos lookup table
  static inline float cosLookUp(float value);
};

}  // namespace innovusion

#endif  // CONVERTER_PNG_RANGE_IMAGE_H_
