/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include <string.h>
#include <cmath>
#include <limits>
#include <algorithm>

#include "sdk_common/converter/png_range_image.h"
#include "utils/inno_lidar_log.h"

#define inno_lrintf(x) (lrintf(static_cast<float>(x)))
#define inno_lrint(x) (lrint(static_cast<double>(x)))

namespace innovusion {

inline float rad2deg(float alpha) { return (alpha * 57.29578f); }

inline float deg2rad(float alpha) { return (alpha * 0.017453293f); }

inline double rad2deg(double alpha) { return (alpha * 57.29578); }

inline double deg2rad(double alpha) { return (alpha * 0.017453293); }

const int RangeImage::lookup_table_size = 20001;

std::vector<float> RangeImage::asin_lookup_table;
std::vector<float> RangeImage::atan_lookup_table;
std::vector<float> RangeImage::cos_lookup_table;

RangeImage::RangeImage(float angular_resolution_x, float angular_resolution_y,
                       float max_angle_width, float max_angle_height) {
  createLookupTables();

  //
  unobserved_point.range = -std::numeric_limits<float>::infinity();
  unobserved_point.reflectance = 0;

  // set Angular Resolution
  angular_resolution_x_ = angular_resolution_x;
  angular_resolution_x_reciprocal_ = 1.0f / angular_resolution_x_;

  angular_resolution_y_ = angular_resolution_y;
  angular_resolution_y_reciprocal_ = 1.0f / angular_resolution_y_;

  //
  width = static_cast<uint32_t>(inno_lrint(
      std::floor(max_angle_width * angular_resolution_x_reciprocal_)));
  height = static_cast<uint32_t>(inno_lrint(
      std::floor(max_angle_height * angular_resolution_y_reciprocal_)));

  int full_width = static_cast<int>(inno_lrint(
      std::floor(deg2rad(360.0f) * angular_resolution_x_reciprocal_)));
  int full_height = static_cast<int>(inno_lrint(
      std::floor(deg2rad(180.0f) * angular_resolution_y_reciprocal_)));

  image_offset_x_ = (full_width - static_cast<int>(width)) / 2;
  image_offset_y_ = (full_height - static_cast<int>(height)) / 2;

  //
  unsigned int size = width * height;
  this->points.clear();
  this->points.resize(size, unobserved_point);
}

RangeImage::~RangeImage() {}

void RangeImage::start(float noise_level_, float min_range_, int border_size_) {
  is_dense = false;

  crop_top = height;
  crop_bottom = -1;
  crop_left = width;
  crop_right = -1;

  this->counters_.clear();
  this->counters_.resize(width * height, 0);
}

void RangeImage::insert_point(float x, float y, float z, float reflectance) {
  //
  bool is_finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  if (!is_finite)  // Check for NAN etc
    return;

  float transformedPoint[3] = {y, -x, z};
  float range_of_current_point =
      std::sqrt(std::pow(y, 2) + std::pow(x, 2) + std::pow(z, 2));

  float angle_x = atan2LookUp(transformedPoint[0], transformedPoint[2]);
  float angle_y = asinLookUp(transformedPoint[1] / range_of_current_point);

  // Get the image point corresponding to the given angles
  float x_real = (angle_x * cosLookUp(angle_y) + static_cast<float>(M_PI)) *
                     angular_resolution_x_reciprocal_ -
                 static_cast<float>(image_offset_x_);
  float y_real = (angle_y + 0.5f * static_cast<float>(M_PI)) *
                     angular_resolution_y_reciprocal_ -
                 static_cast<float>(image_offset_y_);

  // real2D to Int2D
  int x_image = static_cast<int>(inno_lrintf(x_real));
  int y_image = static_cast<int>(inno_lrintf(y_real));

  if (range_of_current_point < min_range || !is_in_image(x_image, y_image))
    return;

  // Do some minor interpolation by checking the three closest neighbors to the
  // point, that are not filled yet.
  int floor_x = inno_lrint(std::floor(x_real));
  int floor_y = inno_lrint(std::floor(y_real));

  int ceil_x = inno_lrint(std::ceil(x_real));
  int ceil_y = inno_lrint(std::ceil(y_real));

  int neighbor_x[4], neighbor_y[4];
  neighbor_x[0] = floor_x;
  neighbor_y[0] = floor_y;
  neighbor_x[1] = floor_x;
  neighbor_y[1] = ceil_y;
  neighbor_x[2] = ceil_x;
  neighbor_y[2] = floor_y;
  neighbor_x[3] = ceil_x;
  neighbor_y[3] = ceil_y;
  // std::cout << x_real<<","<<y_real<<": ";

  for (int i = 0; i < 4; ++i) {
    int n_x = neighbor_x[i];
    int n_y = neighbor_y[i];

    // std::cout << n_x<<","<<n_y<<" ";
    if (n_x == x_image && n_y == y_image) continue;

    if (is_in_image(n_x, n_y)) {
      int neighbor_array_pos = n_y * width + n_x;
      if (this->counters_[neighbor_array_pos] == 0) {
        this->points[neighbor_array_pos].reflectance = reflectance;

        float &neighbor_range = this->points[neighbor_array_pos].range;
        neighbor_range =
            (std::isinf(neighbor_range)
                 ? range_of_current_point
                 : (std::min)(neighbor_range, range_of_current_point));

        crop_top = (std::min)(crop_top, n_y);
        crop_right = (std::max)(crop_right, n_x);
        crop_bottom = (std::max)(crop_bottom, n_y);
        crop_left = (std::min)(crop_left, n_x);
      }
    }
  }

  // The point itself
  int arrayPos = y_image * width + x_image;
  float &range_at_image_point = this->points[arrayPos].range;
  int &counter = this->counters_[arrayPos];

  bool addCurrentPoint = false;
  bool replace_with_current_point = false;

  if (counter == 0) {
    replace_with_current_point = true;
  } else {
    if (range_of_current_point < range_at_image_point - noise_level) {
      replace_with_current_point = true;
    } else if (std::fabs(range_of_current_point - range_at_image_point) <=
               noise_level) {
      addCurrentPoint = true;
    }
  }

  if (replace_with_current_point) {
    counter = 1;

    this->points[arrayPos].reflectance = reflectance;
    range_at_image_point = range_of_current_point;

    crop_top = (std::min)(crop_top, y_image);
    crop_right = (std::max)(crop_right, x_image);
    crop_bottom = (std::max)(crop_bottom, y_image);
    crop_left = (std::min)(crop_left, x_image);
    // std::cout << "Adding point "<<x<<","<<y<<"\n";
  } else if (addCurrentPoint) {
    ++counter;
    range_at_image_point +=
        (range_of_current_point - range_at_image_point) / counter;
  }
}

void RangeImage::stop() {
  inno_log_info("point size : %" PRI_SIZELU,
                this->points.size());

  //
  bool topIsDone = true, bottomIsDone = true;
  bool leftIsDone = true, rightIsDone = true;

  if (crop_top < 0) {
    crop_top = -1;
    topIsDone = false;
  }
  if (crop_right < 0) {
    crop_right = static_cast<int>(width);
    rightIsDone = false;
  }
  if (crop_bottom < 0) {
    crop_bottom = static_cast<int>(height);
    bottomIsDone = false;
  }
  if (crop_left < 0) {
    crop_left = -1;
    leftIsDone = false;
  }

  // Find top border
  while (!topIsDone && crop_top <= crop_bottom) {
    ++crop_top;
    int lineStart = crop_top * width;
    int min_x = std::max(0, crop_left);
    int max_x = std::min(static_cast<int>(width) - 1, crop_right);

    for (int x = min_x; x <= max_x && !topIsDone; ++x)
      if (std::isfinite(this->points[lineStart + x].range)) topIsDone = true;
  }

  // Check if range image is empty
  if (crop_top >= static_cast<int>(height)) {
    this->points.clear();
    width = height = 0;
    return;
  }

  // Find right border
  while (!rightIsDone) {
    --crop_right;
    int min_y = std::max(0, crop_top);
    int max_y = std::min(static_cast<int>(height) - 1, crop_bottom);

    for (int y = min_y; y <= max_y && !rightIsDone; ++y)
      if (std::isfinite(this->points[y * width + crop_right].range))
        rightIsDone = true;
  }

  // Find bottom border
  while (!bottomIsDone) {
    --crop_bottom;
    int lineStart = crop_bottom * width;
    int min_x = std::max(0, crop_left);
    int max_x = std::min(static_cast<int>(width) - 1, crop_right);

    for (int x = min_x; x <= max_x && !bottomIsDone; ++x)
      if (std::isfinite(this->points[lineStart + x].range)) bottomIsDone = true;
  }

  // Find left border
  while (!leftIsDone) {
    ++crop_left;
    int min_y = std::max(0, crop_top);
    int max_y = std::min(static_cast<int>(height) - 1, crop_bottom);

    for (int y = min_y; y <= max_y && !leftIsDone; ++y)
      if (std::isfinite(this->points[y * width + crop_left].range))
        leftIsDone = true;
  }

  crop_left -= borderSize;
  crop_top -= borderSize;
  crop_right += borderSize;
  crop_bottom += borderSize;

  // Create copy without copying the old points - vector::swap only copies a few
  // pointers, not the content
  RangePointVector tmpPoints;
  this->points.swap(tmpPoints);

  RangeImage oldRangeImage = *this;
  tmpPoints.swap(oldRangeImage.points);

  width = crop_right - crop_left + 1;
  height = crop_bottom - crop_top + 1;
  image_offset_x_ = crop_left + oldRangeImage.image_offset_x_;
  image_offset_y_ = crop_top + oldRangeImage.image_offset_y_;
  this->points.resize(width * height);

  // std::cout << oldRangeImage.width<<"x"<<oldRangeImage.height<<" ->
  // "<<width<<"x"<<height<<"\n";

  // Copy points
  for (int y = 0, oldY = crop_top; y < static_cast<int>(height); ++y, ++oldY) {
    for (int x = 0, oldX = crop_left; x < static_cast<int>(width);
         ++x, ++oldX) {
      RangePoint &currentPoint = this->points[y * width + x];
      if (oldX < 0 || oldX >= static_cast<int>(oldRangeImage.width) ||
          oldY < 0 || oldY >= static_cast<int>(oldRangeImage.height)) {
        currentPoint = unobserved_point;
        continue;
      }
      currentPoint = oldRangeImage.points[oldY * oldRangeImage.width + oldX];
    }
  }
}

bool RangeImage::is_in_image(int x, int y) const {
  return (x >= 0 && x < static_cast<int>(width) && y >= 0 &&
          y < static_cast<int>(height));
}

void RangeImage::createLookupTables() {
  if (!asin_lookup_table.empty()) return;

  asin_lookup_table.resize(lookup_table_size);
  for (int i = 0; i < lookup_table_size; ++i) {
    float value = static_cast<float>(i - (lookup_table_size - 1) / 2) /
                  static_cast<float>((lookup_table_size - 1) / 2);
    asin_lookup_table[i] = asinf(value);
  }

  atan_lookup_table.resize(lookup_table_size);
  for (int i = 0; i < lookup_table_size; ++i) {
    float value = static_cast<float>(i - (lookup_table_size - 1) / 2) /
                  static_cast<float>((lookup_table_size - 1) / 2);
    atan_lookup_table[i] = std::atan(value);
  }

  cos_lookup_table.resize(lookup_table_size);

  for (int i = 0; i < lookup_table_size; ++i) {
    float value = static_cast<float>(i) * 2.0f * static_cast<float>(M_PI) /
                  static_cast<float>(lookup_table_size - 1);
    cos_lookup_table[i] = std::cos(value);
  }
}

void RangeImage::get_image(ColorFunc color_func, ImageFunc image_func) {
  // get min, max range
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();

  for (const auto &point : this->points) {
    float range = point.range;
    if (!std::isfinite(range)) continue;

    min_value = (std::min)(min_value, range);
    max_value = (std::max)(max_value, range);
  }

  //
  float factor = 1.0f / (max_value - min_value);
  float offset = -min_value;

  int i = 0;
  unsigned char r, g, b;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      i = y * width + x;

      // unsigned char &r = *(dataPtr++), &g = *(dataPtr++), &b = *(dataPtr++);
      float range = this->points[i].range;
      float ref = this->points[i].reflectance / 255;

      if (std::isfinite(range)) {
        // Normalize value to [0, 1]
        range = std::max(0.0f, std::min(1.0f, factor * (range + offset)));
        color_func(range, ref, &r, &g, &b);
      } else {
        color_func(range, ref, &r, &g, &b);
      }

      // save
      image_func(x, y, r, g, b);
    }
  }
}

inline float RangeImage::asinLookUp(float value) {
  return (asin_lookup_table[static_cast<int>(
      static_cast<float>(inno_lrintf(
          (static_cast<float>(lookup_table_size - 1) / 2.0f) * value)) +
      static_cast<float>(lookup_table_size - 1) / 2.0f)]);
}

inline float RangeImage::atan2LookUp(float y, float x) {
  if (x == 0 && y == 0) return 0;

  float ret;
  if (std::abs(x) < std::abs(y)) {
    ret = atan_lookup_table[static_cast<int>(
        static_cast<float>(inno_lrintf(
            (static_cast<float>(lookup_table_size - 1) / 2.0f) * (x / y))) +
        static_cast<float>(lookup_table_size - 1) / 2.0f)];
    ret = static_cast<float>(x * y > 0 ? M_PI / 2 - ret : -M_PI / 2 - ret);
  } else {
    ret = atan_lookup_table[static_cast<int>(
        static_cast<float>(inno_lrintf(
            (static_cast<float>(lookup_table_size - 1) / 2.0f) * (y / x))) +
        static_cast<float>(lookup_table_size - 1) / 2.0f)];
  }

  if (x < 0) ret = static_cast<float>(y < 0 ? ret - M_PI : ret + M_PI);

  return (ret);
}

inline float RangeImage::cosLookUp(float value) {
  int cell_idx = static_cast<int>(
      inno_lrintf((static_cast<float>(lookup_table_size - 1)) *
                  std::abs(value) / (2.0f * static_cast<float>(M_PI))));
  return (cos_lookup_table[cell_idx]);
}

}  // namespace innovusion
